"""
Main inference engine that orchestrates all components.
Runs the inference loop at 16 Hz and coordinates input capture and action output.

Aligned with final_model/ architecture:
  - TemporalCrossTransformer: d_model=384, 81 tokens (with audio) or 65 (without)
  - Scene: 64 frames → YOLO embed → (B, 64, 512) → ModalityCompressor 64→16
  - Radar: 16 tokens → RadarEncoder → (B, 16, 512)
  - Audio: stereo 16sec → StereoAudioEncoder → (B, 32, 512) → linspace 16
  - Detection: (B, 16, 100)
  - Action: (B, 64, 22) → ModalityCompressor 64→16
  - State: (B, 100)
"""

import sys
import time
import threading
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch

from ..config import Config, ACTION_KEYS, ACTION_NAMES, MODEL_DIMS
from ..capture.screen_capture import ScreenCapture
from ..capture.audio_capture import AudioCapture, get_audio_capture
from ..buffers.ring_buffers import (
    FrameBuffer, StereoAudioBuffer, ActionHistoryBuffer,
    StateBuffer, DetectionBuffer,
)
from ..models.model_loader import load_models, ModelBundle
from .preprocessor import Preprocessor
from .embedding_cache import GPUEmbeddingCache, AudioEmbeddingCache


class MillisecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            s = ct.strftime("%H:%M:%S")
        return f"{s}.{int(record.msecs):03d}"


def setup_inference_logger(log_dir: str = "./inference_logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"inference_{timestamp}.log"

    logger = logging.getLogger("inference")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(MillisecondFormatter('%(asctime)s | %(message)s'))
    logger.addHandler(fh)

    logger.debug("=== Inference Log Started ===")
    print(f"[Engine] Logging to: {log_file}")
    return logger


@dataclass
class InferenceResult:
    """Result from a single inference step."""
    timestamp: float

    mouse_delta: tuple       # (yaw_delta, pitch_delta) in degrees (denormalized)
    mouse_delta_raw: tuple   # raw normalized model output: delta / (T * MOUSE_SCALE)
    key_logits: np.ndarray   # (20,) raw logits
    key_probs: np.ndarray    # (20,) after sigmoid
    value: float

    mouse_pixels: tuple  # (dx, dy) in pixels
    pressed_keys: set  # Set of key names to press

    inference_time_ms: float
    fps: float


class InferenceEngine:
    """
    Main inference engine for the CS2 VLA Agent.

    Data flow per frame:
    1. Scene frame (640×640) → YOLO extract_features + embed → scene cache (64 entries, 512-dim)
    2. Scene frame → YOLO forward_detections_only → postprocess → detection buffer (16 entries)
    3. Radar frame (224×224) → RadarEncoder → radar cache (16 entries, 512-dim)
    4. Stereo audio (2, 256000) → StereoAudioEncoder → (32, 512) → linspace → (16, 512)
    5. Action history (64, 22) from action buffer
    6. State vector (100,) from GSI
    7. TemporalCrossTransformer → mouse_embed (384), keys (20), value (1)
    8. FlowActionHead.sample(mouse_embed) → mouse_delta (2,)
    """

    def __init__(
        self,
        config: Config,
        checkpoint_path: str,
        device: str = "cuda",
        use_audio: bool = True,
        use_buy: bool = False,
    ):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.use_audio = use_audio
        self.use_buy = use_buy

        self._running = False
        self._inference_thread: Optional[threading.Thread] = None
        self._inference_count = 0
        self._last_inference_time = 0.0
        self._current_fps = 0.0

        # Components (initialized in start())
        self.models: Optional[ModelBundle] = None
        self.screen_capture: Optional[ScreenCapture] = None
        self.audio_capture: Optional[AudioCapture] = None
        self.preprocessor: Optional[Preprocessor] = None

        # Buffers — aligned with final_model/ dimensions
        self.scene_buffer = FrameBuffer(max_size=config.scene_buffer_size)    # 64 frames
        self.radar_buffer = FrameBuffer(max_size=config.radar_buffer_size)    # 32 frames
        self.audio_buffer = StereoAudioBuffer(
            sample_rate=config.audio_sample_rate,
            duration=config.audio_buffer_duration,
            channels=config.audio_channels,
        )
        self.action_history = ActionHistoryBuffer(max_size=64)       # 64 action history
        self.state_buffer = StateBuffer(max_size=16, state_dim=100)  # state_dim=100
        self.detection_buffer = DetectionBuffer(max_size=16, detection_dim=100)

        # Embedding caches (initialized in start())
        self.radar_cache: Optional[GPUEmbeddingCache] = None
        self.scene_cache: Optional[GPUEmbeddingCache] = None
        self.audio_cache: Optional[AudioEmbeddingCache] = None
        self._cache_enabled = True

        # Current game state (updated by GSI callback)
        self._current_state_vec: Optional[np.ndarray] = None
        self._state_lock = threading.Lock()

        # Callbacks
        self._result_callbacks: list = []

        # Logger
        self._logger: Optional[logging.Logger] = None

        # Controllers (set externally)
        self.mouse_controller = None
        self.keyboard_controller = None
        self.overlay = None
        self.buy_executor = None

    def set_mouse_controller(self, controller):
        self.mouse_controller = controller

    def set_keyboard_controller(self, controller):
        self.keyboard_controller = controller

    def set_overlay(self, overlay):
        self.overlay = overlay

    def set_buy_executor(self, executor):
        self.buy_executor = executor

    def register_result_callback(self, callback: Callable[[InferenceResult], None]):
        self._result_callbacks.append(callback)

    def update_game_state(self, state_vec: np.ndarray):
        """
        Update current game state (called by GSI adapter).

        Args:
            state_vec: (100,) state vector
        """
        with self._state_lock:
            self._current_state_vec = state_vec.copy()
            self.state_buffer.add(time.time(), state_vec)

    def start(self):
        if self._running:
            print("[Engine] Already running")
            return

        print("[Engine] Starting...")

        self._logger = setup_inference_logger()

        # Load models from final_model/
        print("[Engine] Loading models...")
        self.models = load_models(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            use_audio=self.use_audio,
            use_trt=self.config.use_trt,
            trt_dir=self.config.trt_dir,
            final_model_path=self.config.final_model_path,
        )

        self._verify_model_modes()

        # Initialize preprocessor with correct dimensions
        self.preprocessor = Preprocessor(
            device=self.device,
            radar_seq_len=MODEL_DIMS["radar_seq"],       # 16
            scene_seq_len=MODEL_DIMS["scene_seq"],        # 64
            action_seq_len=MODEL_DIMS["actions_seq"],     # 64
            state_dim=MODEL_DIMS["state_dim"],            # 100
        )

        # Initialize embedding caches
        if self._cache_enabled:
            print("[Engine] Initializing embedding caches...")

            self.radar_cache = GPUEmbeddingCache(
                capacity=256,
                emb_dim=MODEL_DIMS["radar_dim"],   # 512
                device=self.device,
                enable_validation=False
            )

            self.scene_cache = GPUEmbeddingCache(
                capacity=256,
                emb_dim=MODEL_DIMS["scene_dim"],   # 512
                device=self.device,
                enable_validation=False
            )

            if self.use_audio:
                self.audio_cache = AudioEmbeddingCache(
                    embedding_duration_ms=500,
                    sample_rate=self.config.audio_sample_rate,
                    device=self.device
                )

            radar_stats = self.radar_cache.get_stats()
            scene_stats = self.scene_cache.get_stats()
            print(f"  Radar cache: {radar_stats['memory_mb']:.1f} MB (dim={MODEL_DIMS['radar_dim']})")
            print(f"  Scene cache: {scene_stats['memory_mb']:.1f} MB (dim={MODEL_DIMS['scene_dim']})")
            if self.use_audio:
                print(f"  Audio cache: on-demand (stereo 16sec → 32 embeds → linspace 16)")

        # Start screen capture
        print("[Engine] Starting screen capture...")
        self.screen_capture = ScreenCapture(
            screen_width=self.config.screen_width,
            screen_height=self.config.screen_height,
            radar_crop_box=self.config.radar_crop_box,
            radar_size=self.config.radar_size,
            target_fps=self.config.capture_fps,
            monitor_index=self.config.monitor_index,
        )
        self.screen_capture.set_scene_callback(self._on_scene_frame)
        self.screen_capture.set_radar_callback(self._on_radar_frame)
        self.screen_capture.start()

        # Start audio capture (optional, stereo)
        if self.use_audio:
            print("[Engine] Starting stereo audio capture...")
            self.audio_capture = get_audio_capture(
                sample_rate=self.config.audio_sample_rate,
                buffer_duration=self.config.audio_buffer_duration,
                channels=self.config.audio_channels,
            )
            self.audio_capture.set_callback(self._on_audio_chunk)
            self.audio_capture.start()

        # Start inference thread
        self._running = True
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

        print(f"[Engine] Started at {self.config.inference_rate} Hz inference rate")

    def stop(self):
        print("[Engine] Stopping...")
        self._running = False

        if self.screen_capture:
            self.screen_capture.stop()

        if self.audio_capture:
            self.audio_capture.stop()

        if self._inference_thread:
            self._inference_thread.join(timeout=2.0)

        if self.keyboard_controller:
            self.keyboard_controller.release_all()

        print(f"[Engine] Stopped. Total inferences: {self._inference_count}")

    def _on_scene_frame(self, timestamp: float, frame: np.ndarray):
        self.scene_buffer.add(timestamp, frame)

    def _on_radar_frame(self, timestamp: float, frame: np.ndarray):
        self.radar_buffer.add(timestamp, frame)

    def _on_audio_chunk(self, chunk: np.ndarray):
        self.audio_buffer.add(chunk)

    def _inference_loop(self):
        target_interval = 1.0 / self.config.inference_rate
        fps_window = []

        while self._running:
            loop_start = time.time()

            try:
                result = self._run_inference()

                if result is not None:
                    if self.config.apply_actions:
                        self._apply_actions(result)

                    self.action_history.add(
                        np.array(result.mouse_delta_raw, dtype=np.float32),
                        result.key_probs
                    )

                    for callback in self._result_callbacks:
                        try:
                            callback(result)
                        except Exception as e:
                            print(f"[Engine] Callback error: {e}")

                    if self.overlay:
                        self._update_overlay(result)

                    self._log_inference(result)
                    self._inference_count += 1

            except Exception as e:
                print(f"[Engine] Inference error: {e}")
                import traceback
                traceback.print_exc()

            elapsed = time.time() - loop_start
            fps_window.append(elapsed)
            if len(fps_window) > 30:
                fps_window.pop(0)
            self._current_fps = 1.0 / (sum(fps_window) / len(fps_window)) if fps_window else 0

            sleep_time = max(0, target_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _should_invalidate_cache(self, new_state_vec: Optional[np.ndarray]) -> bool:
        # TODO: Detect round restart, respawn, etc.
        return False

    def _run_inference(self) -> Optional[InferenceResult]:
        """Run a single inference step matching final_model/ architecture."""
        start_time = time.time()

        if len(self.scene_buffer) < 1:
            return None

        # Get current state
        with self._state_lock:
            state_vec = self._current_state_vec

        # Cache invalidation check
        if self._should_invalidate_cache(state_vec):
            if self._logger:
                self._logger.debug("[CACHE] Invalidated (round restart detected)")
            if self.radar_cache:
                self.radar_cache.invalidate()
            if self.scene_cache:
                self.scene_cache.invalidate()
            if self.audio_cache:
                self.audio_cache.invalidate()
            self.detection_buffer.clear()

        # Log diagnostics
        if self._inference_count % 50 == 0:
            self._log_diagnostics(
                scene_len=len(self.scene_buffer),
                radar_len=len(self.radar_buffer),
                has_state=(state_vec is not None),
                has_audio=(self.use_audio and self.audio_capture is not None)
            )

        with torch.no_grad():
            # === 1. Scene: encode latest frame → scene_cache (64, 512) ===
            latest_scene = self.scene_buffer.get_latest()
            if latest_scene is not None:
                ts, frame = latest_scene
                # Prepare frame for YOLO: (1, 3, 640, 640)
                if frame.dtype == np.uint8:
                    frame_f = frame.astype(np.float32) / 255.0
                else:
                    frame_f = frame.copy()
                if frame_f.shape[0] != 640 or frame_f.shape[1] != 640:
                    import cv2
                    frame_f = cv2.resize(frame_f, (640, 640), interpolation=cv2.INTER_LINEAR)
                frame_tensor = torch.from_numpy(
                    np.transpose(frame_f, (2, 0, 1))
                ).unsqueeze(0).to(self.device)

                # Get scene embedding via YOLO
                p3, p5 = self.models.yolo.extract_features(frame_tensor)
                scene_emb = self.models.yolo.embed_from_features(p3, p5)  # (1, 512)
                self.scene_cache.add_embedding(scene_emb.squeeze(0), timestamp=ts)

                # Get detections
                det_raw = self.models.yolo.forward_detections_only(frame_tensor)
                det_vec = self.preprocessor._postprocess_detections(
                    det_raw, self.models.yolo, max_det=20
                )
                self.detection_buffer.add(det_vec)

            # Get scene sequence: last 64 embeddings → (1, 64, 512)
            scene_embeds = self.scene_cache.get_sequence(
                MODEL_DIMS["scene_seq"]   # 64
            ).unsqueeze(0)  # (1, 64, 512)

            # === 2. Radar: encode latest frame → radar_cache (16, 512) ===
            latest_radar = self.radar_buffer.get_latest()
            if latest_radar is not None:
                ts, frame = latest_radar
                if frame.dtype == np.uint8:
                    frame_f = frame.astype(np.float32) / 255.0
                else:
                    frame_f = frame.copy()
                frame_tensor = torch.from_numpy(
                    np.transpose(frame_f, (2, 0, 1))
                ).unsqueeze(0).to(self.device)

                radar_emb = self.models.radar_encoder(frame_tensor)  # (1, 512)
                self.radar_cache.add_embedding(radar_emb.squeeze(0), timestamp=ts)

            # Get radar sequence: last 16 embeddings → (1, 16, 512)
            radar_embeds = self.radar_cache.get_sequence(
                MODEL_DIMS["radar_seq"]   # 16
            ).unsqueeze(0)  # (1, 16, 512)

            # === 3. Audio: stereo (1, 2, 256000) → (1, 16, 512) ===
            audio_embeds = None
            if self.use_audio and self.models.audio_encoder is not None:
                audio_data = self.audio_capture.get_buffer() if self.audio_capture else \
                    np.zeros((2, 256000), dtype=np.float32)
                audio_tensor = self.preprocessor.prepare_audio(audio_data)  # (1, 2, 256000)

                if self._cache_enabled and self.audio_cache is not None:
                    current_position = self.audio_capture.total_samples if self.audio_capture else 0
                    audio_embeds = self.audio_cache.get_embeddings(
                        audio_tensor, current_position, self.models.audio_encoder
                    )  # (1, 16, 512)
                else:
                    audio_embeds = self.preprocessor.encode_audio(
                        self.models.audio_encoder, audio_tensor
                    )  # (1, 16, 512)

            # === 4. Detection sequence: (1, 16, 100) ===
            det_sequence = self.detection_buffer.get_sequence()  # (16, 100)
            detection_tensor = self.preprocessor.prepare_detections(det_sequence)  # (1, 16, 100)

            # === 5. Action history: (1, 64, 22) ===
            action_tensor = self.preprocessor.prepare_action_history(
                self.action_history.get_history()
            )  # (1, 64, 22)

            # === 6. State vector: (1, 100) ===
            state_tensor = self.preprocessor.prepare_state_vector(state_vec)  # (1, 100)

            # === 7. TemporalCrossTransformer forward ===
            mouse_embed, policy_keys, value = self.models.temporal_model(
                radar_seq=radar_embeds,          # (1, 16, 512)
                scene_seq=scene_embeds,          # (1, 64, 512) → compressed to 16
                detection_seq=detection_tensor,  # (1, 16, 100)
                action_seq=action_tensor,        # (1, 64, 22) → compressed to 16
                state_vec=state_tensor,          # (1, 100)
                audio_seq=audio_embeds,          # (1, 16, 512) or None
            )

            # === 8. FlowActionHead: sample mouse delta ===
            if hasattr(self.models.flow_head, 'trt_engine'):
                policy_mouse = mouse_embed  # TRT already returns final
            else:
                policy_mouse = self.models.flow_head.sample(mouse_embed)  # (1, 2)

            # Extract outputs
            # mouse_delta_normalized: raw model output = delta_degrees / (T * MOUSE_SCALE)
            mouse_delta_normalized = policy_mouse[0].cpu().numpy()  # (2,)
            key_logits = policy_keys[0].cpu().numpy()  # (20,)
            key_probs = torch.sigmoid(policy_keys[0]).cpu().numpy()
            value_pred = value[0, 0].item() if value.dim() > 1 else value[0].item()

            # Debug first 5 inferences
            if self._inference_count < 5:
                print(f"\n[DEBUG] Inference #{self._inference_count}")
                print(f"  Input shapes: radar={radar_embeds.shape}, scene={scene_embeds.shape}, "
                      f"det={detection_tensor.shape}, action={action_tensor.shape}, state={state_tensor.shape}")
                if audio_embeds is not None:
                    print(f"  Audio: {audio_embeds.shape}")
                print(f"  Raw mouse output (normalized): {mouse_delta_normalized}")
                print(f"  Key probs top-5: {sorted(enumerate(key_probs), key=lambda x: -x[1])[:5]}")

        # Denormalize mouse: training target = delta_degrees / (actual_ticks * MOUSE_SCALE)
        # At 16 Hz inference and 64 tick/sec server: T_INFERENCE = 64 / 16 = 4
        _MOUSE_SCALE = 25.0
        _T_INFERENCE = max(1, round(64 / self.config.inference_rate))
        yaw_degrees = float(mouse_delta_normalized[0]) * _T_INFERENCE * _MOUSE_SCALE
        pitch_degrees = float(mouse_delta_normalized[1]) * _T_INFERENCE * _MOUSE_SCALE

        # Convert degrees to pixels for display/logging
        dx = int(yaw_degrees / self.config.degrees_per_pixel * self.config.mouse_sensitivity)
        dy = int(pitch_degrees / self.config.degrees_per_pixel * self.config.mouse_sensitivity)

        # Determine which keys to press
        pressed_keys = set()
        for idx, prob in enumerate(key_probs):
            if prob > self.config.key_threshold:
                key = ACTION_KEYS.get(idx)
                if key:
                    pressed_keys.add(key)

        inference_time = (time.time() - start_time) * 1000

        return InferenceResult(
            timestamp=time.time(),
            mouse_delta=(yaw_degrees, pitch_degrees),
            mouse_delta_raw=tuple(mouse_delta_normalized.tolist()),
            key_logits=key_logits,
            key_probs=key_probs,
            value=value_pred,
            mouse_pixels=(dx, dy),
            pressed_keys=pressed_keys,
            inference_time_ms=inference_time,
            fps=self._current_fps,
        )

    def _apply_actions(self, result: InferenceResult):
        if self.mouse_controller:
            self.mouse_controller.apply_delta(
                result.mouse_delta[0],
                result.mouse_delta[1]
            )

        if self.keyboard_controller:
            self.keyboard_controller.apply_actions(
                torch.from_numpy(result.key_logits)
            )

    def _update_overlay(self, result: InferenceResult):
        if not self.overlay:
            return

        key_probs_dict = {
            ACTION_NAMES[i]: float(result.key_probs[i])
            for i in range(min(len(ACTION_NAMES), len(result.key_probs)))
        }

        self.overlay.update(
            mouse_delta=result.mouse_delta,
            mouse_pixels=result.mouse_pixels,
            key_probs=key_probs_dict,
            pressed_keys=result.pressed_keys,
            fps=result.fps,
            game_state={
                "value": result.value,
                "inference_ms": result.inference_time_ms,
            }
        )

    def _log_inference(self, result: InferenceResult):
        if not self._logger:
            return

        key_probs_str = " | ".join([
            f"{ACTION_NAMES[i]}:{result.key_probs[i]:.3f}"
            for i in range(min(len(ACTION_NAMES), len(result.key_probs)))
        ])

        pressed_str = ",".join(sorted(result.pressed_keys)) if result.pressed_keys else "none"

        log_line = (
            f"FPS:{result.fps:.1f} | "
            f"mouse_delta:({result.mouse_delta[0]:+.3f}, {result.mouse_delta[1]:+.3f}) | "
            f"mouse_px:({result.mouse_pixels[0]:+d}, {result.mouse_pixels[1]:+d}) | "
            f"pressed:[{pressed_str}] | "
            f"keys: {key_probs_str}"
        )

        self._logger.debug(log_line)

    def _log_diagnostics(self, scene_len: int, radar_len: int, has_state: bool, has_audio: bool):
        if not self._logger:
            return

        diag_parts = [
            f"scene_frames={scene_len}/{self.config.scene_buffer_size}",
            f"radar_frames={radar_len}/{self.config.radar_buffer_size}",
            f"state={'YES' if has_state else 'NO'}",
            f"audio={'YES' if has_audio else 'NO'}",
            f"det_buffer={self.detection_buffer.filled_count}/16",
            f"action_buf={self.action_history.filled_count}/64"
        ]

        if self._cache_enabled:
            if self.radar_cache:
                rs = self.radar_cache.get_stats()
                diag_parts.append(f"radar_cache={rs['count']}/{rs['capacity']} ({rs['filled_ratio']*100:.0f}%)")

            if self.scene_cache:
                ss = self.scene_cache.get_stats()
                diag_parts.append(f"scene_cache={ss['count']}/{ss['capacity']} ({ss['filled_ratio']*100:.0f}%)")

            if self.audio_cache and self.use_audio:
                aus = self.audio_cache.get_stats()
                diag_parts.append(f"audio_hit_rate={aus['hit_rate']*100:.0f}%")

        self._logger.debug(f"DIAG: {' | '.join(diag_parts)}")

    def _verify_model_modes(self):
        print("\n[Engine] === Model Mode Verification ===")

        models_info = [
            ("radar_encoder", self.models.radar_encoder),
            ("yolo", self.models.yolo),
            ("temporal_model", self.models.temporal_model),
        ]
        if self.models.audio_encoder is not None:
            models_info.append(("audio_encoder", self.models.audio_encoder))

        all_eval = True
        for name, model in models_info:
            if hasattr(model, 'training'):
                is_training = model.training
                status = "TRAIN" if is_training else "EVAL"
                if is_training:
                    all_eval = False
                    print(f"  [!] {name}: {status} <-- PROBLEM!")
                else:
                    print(f"  [OK] {name}: {status}")
            else:
                print(f"  [OK] {name}: (no training attr)")

        if not all_eval:
            print("[Engine] WARNING: Some models are NOT in eval mode!")
            print("[Engine] Forcing eval mode...")
            self.models.eval()
        else:
            print("[Engine] All models in EVAL mode - OK")

        print("[Engine] ================================\n")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "inference_count": self._inference_count,
            "fps": self._current_fps,
            "scene_buffer_size": len(self.scene_buffer),
            "radar_buffer_size": len(self.radar_buffer),
            "audio_buffer_filled": self.audio_buffer.get_filled_ratio() if self.audio_capture else 0,
            "running": self._running,
        }
