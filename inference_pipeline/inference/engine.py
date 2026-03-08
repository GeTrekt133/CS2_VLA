"""
Main inference engine — event-driven, single-threaded.

Architecture:
  capture → YOLO embed + detection → (radar @ 1Hz) → (audio @ 2Hz) →
  (alive @ 2Hz) → transformer → action → next capture

No fixed FPS — runs as fast as the pipeline allows.

Aligned with final_model/ architecture:
  - TemporalCrossTransformer: d_model=384, 81 tokens (with audio) or 65 (without)
  - Scene: 64 embeddings → ModalityCompressor 64→16
  - Radar: 16 embeddings → (B, 16, 512)
  - Audio: stereo 16sec → 32 embeds → linspace 16
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

# Alive digit detector (lazy import)
_AliveDigitDetector = None

def _get_alive_detector_class():
    global _AliveDigitDetector
    if _AliveDigitDetector is None:
        from alive_digit.inference import AliveDigitDetector
        _AliveDigitDetector = AliveDigitDetector
    return _AliveDigitDetector


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

    mouse_delta: tuple       # (yaw_delta, pitch_delta) in degrees
    mouse_delta_raw: tuple   # raw normalized model output
    key_logits: np.ndarray   # (20,) raw logits
    key_probs: np.ndarray    # (20,) after sigmoid
    value: float

    mouse_pixels: tuple  # (dx, dy) in pixels
    pressed_keys: set

    inference_time_ms: float
    fps: float


class InferenceEngine:
    """
    Event-driven inference engine.

    Single-threaded main loop:
      grab frame → YOLO → (radar @ 1Hz) → (audio @ 2Hz) →
      (alive @ 2Hz) → transformer → action → repeat
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
        self._current_fps = 0.0

        # Components (initialized in start())
        self.models: Optional[ModelBundle] = None
        self.screen_capture: Optional[ScreenCapture] = None
        self.audio_capture: Optional[AudioCapture] = None
        self.preprocessor: Optional[Preprocessor] = None

        # Buffers
        self.action_history = ActionHistoryBuffer(max_size=64)
        self.state_buffer = StateBuffer(max_size=16, state_dim=100)
        self.detection_buffer = DetectionBuffer(max_size=16, detection_dim=100)

        # Embedding caches (initialized in start())
        self.radar_cache: Optional[GPUEmbeddingCache] = None
        self.scene_cache: Optional[GPUEmbeddingCache] = None
        self.audio_cache: Optional[AudioEmbeddingCache] = None

        # Current game state (updated by GSI callback from another thread)
        self._current_state_vec: Optional[np.ndarray] = None
        self._state_lock = threading.Lock()

        # Rate limiters (timestamps of last processing)
        self._last_radar_time = 0.0
        self._last_audio_time = 0.0
        self._last_alive_time = 0.0

        # Alive digit detector
        self.alive_detector = None
        self._alive_ct = 0
        self._alive_t = 0

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
        """Update current game state (called by GSI server thread)."""
        with self._state_lock:
            self._current_state_vec = state_vec.copy()
            self.state_buffer.add(time.time(), state_vec)

    def start(self):
        if self._running:
            print("[Engine] Already running")
            return

        print("[Engine] Starting...")

        self._logger = setup_inference_logger()

        # Load models
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

        # Preprocessor
        self.preprocessor = Preprocessor(
            device=self.device,
            radar_seq_len=MODEL_DIMS["radar_seq"],
            scene_seq_len=MODEL_DIMS["scene_seq"],
            action_seq_len=MODEL_DIMS["actions_seq"],
            state_dim=MODEL_DIMS["state_dim"],
        )

        # Embedding caches
        print("[Engine] Initializing embedding caches...")
        self.radar_cache = GPUEmbeddingCache(
            capacity=256,
            emb_dim=MODEL_DIMS["radar_dim"],
            device=self.device,
            enable_validation=False,
        )
        self.scene_cache = GPUEmbeddingCache(
            capacity=256,
            emb_dim=MODEL_DIMS["scene_dim"],
            device=self.device,
            enable_validation=False,
        )
        if self.use_audio:
            self.audio_cache = AudioEmbeddingCache(
                embedding_duration_ms=500,
                sample_rate=self.config.audio_sample_rate,
                device=self.device,
            )

        radar_stats = self.radar_cache.get_stats()
        scene_stats = self.scene_cache.get_stats()
        print(f"  Radar cache: {radar_stats['memory_mb']:.1f} MB (dim={MODEL_DIMS['radar_dim']})")
        print(f"  Scene cache: {scene_stats['memory_mb']:.1f} MB (dim={MODEL_DIMS['scene_dim']})")
        if self.use_audio:
            print(f"  Audio cache: on-demand (stereo 16sec → 32 embeds → linspace 16)")

        # Screen capture (synchronous)
        print("[Engine] Initializing screen capture...")
        self.screen_capture = ScreenCapture(
            screen_width=self.config.screen_width,
            screen_height=self.config.screen_height,
            radar_crop_box=self.config.radar_crop_box,
            radar_size=self.config.radar_size,
            monitor_index=self.config.monitor_index,
        )
        self.screen_capture.open()

        # Alive digit detector
        alive_path = Path(self.config.alive_digit_model_path)
        if alive_path.exists():
            try:
                AliveDigitDetector = _get_alive_detector_class()
                self.alive_detector = AliveDigitDetector(str(alive_path), device='cpu')
                print(f"[Engine] Alive digit detector loaded from {alive_path}")
            except Exception as e:
                print(f"[Engine] Failed to load alive detector: {e}")
        else:
            print(f"[Engine] Alive digit model not found at {alive_path} — state_vec[4,5] will be 0")

        # Audio capture (needs its own thread for WASAPI loopback)
        if self.use_audio:
            print("[Engine] Starting stereo audio capture...")
            self.audio_capture = get_audio_capture(
                sample_rate=self.config.audio_sample_rate,
                buffer_duration=self.config.audio_buffer_duration,
                channels=self.config.audio_channels,
            )
            self.audio_capture.start()

        # Start main loop in a thread (so main thread can handle shutdown)
        self._running = True
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

        print("[Engine] Started (event-driven, no FPS cap)")

    def stop(self):
        print("[Engine] Stopping...")
        self._running = False

        if self._inference_thread:
            self._inference_thread.join(timeout=2.0)

        if self.screen_capture:
            self.screen_capture.close()

        if self.audio_capture:
            self.audio_capture.stop()

        if self.keyboard_controller:
            self.keyboard_controller.release_all()

        print(f"[Engine] Stopped. Total inferences: {self._inference_count}")

    def _inference_loop(self):
        """Event-driven main loop: grab → process → act → repeat."""
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

            # FPS tracking (no sleep — run as fast as possible)
            elapsed = time.time() - loop_start
            fps_window.append(elapsed)
            if len(fps_window) > 30:
                fps_window.pop(0)
            self._current_fps = 1.0 / (sum(fps_window) / len(fps_window)) if fps_window else 0

    def _should_invalidate_cache(self, new_state_vec: Optional[np.ndarray]) -> bool:
        # TODO: Detect round restart, respawn, etc.
        return False

    def _run_inference(self) -> Optional[InferenceResult]:
        """Single inference step: grab → encode → transformer → action."""
        start_time = time.time()

        # === 1. Grab frame (blocking, as fast as possible) ===
        ts, scene_frame, radar_frame = self.screen_capture.grab()

        # === 2. Get current state (from GSI thread) ===
        with self._state_lock:
            state_vec = self._current_state_vec

        # === 3. Alive digit detector @ configured rate ===
        if (self.alive_detector is not None
                and ts - self._last_alive_time >= 1.0 / self.config.alive_digit_rate):
            try:
                self._alive_ct, self._alive_t = self.alive_detector.predict(scene_frame)
            except Exception:
                pass
            self._last_alive_time = ts

        # Inject alive counts into state vector
        if state_vec is not None and self.alive_detector is not None:
            state_vec = state_vec.copy()
            state_vec[4] = self._alive_ct / 5.0
            state_vec[5] = self._alive_t / 5.0

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

        # Log diagnostics every 50 frames
        if self._inference_count % 50 == 0:
            self._log_diagnostics(
                has_state=(state_vec is not None),
                has_audio=(self.use_audio and self.audio_capture is not None),
            )

        with torch.no_grad():
            # === 4. Scene: YOLO embed + detection (every frame) ===
            frame_tensor = self._prepare_frame(scene_frame)

            p3, p5 = self.models.yolo.extract_features(frame_tensor)
            scene_emb = self.models.yolo.embed_from_features(p3, p5)  # (1, 512)
            self.scene_cache.add_embedding(scene_emb.squeeze(0), timestamp=ts)

            det_raw = self.models.yolo.forward_detections_only(frame_tensor)
            det_vec = self.preprocessor._postprocess_detections(
                det_raw, self.models.yolo, max_det=20
            )
            self.detection_buffer.add(det_vec)

            # Scene sequence: last 64 embeddings → (1, 64, 512)
            scene_embeds = self.scene_cache.get_sequence(
                MODEL_DIMS["scene_seq"]
            ).unsqueeze(0)

            # === 5. Radar: encode @ radar_rate Hz ===
            if ts - self._last_radar_time >= 1.0 / self.config.radar_rate:
                radar_tensor = self._prepare_frame(radar_frame)
                radar_emb = self.models.radar_encoder(radar_tensor)  # (1, 512)
                self.radar_cache.add_embedding(radar_emb.squeeze(0), timestamp=ts)
                self._last_radar_time = ts

            # Radar sequence: last 16 embeddings → (1, 16, 512)
            radar_embeds = self.radar_cache.get_sequence(
                MODEL_DIMS["radar_seq"]
            ).unsqueeze(0)

            # === 6. Audio: encode @ audio_rate Hz ===
            audio_embeds = None
            if self.use_audio and self.models.audio_encoder is not None:
                if ts - self._last_audio_time >= 1.0 / self.config.audio_rate:
                    audio_data = self.audio_capture.get_buffer() if self.audio_capture else \
                        np.zeros((2, 256000), dtype=np.float32)
                    audio_tensor = self.preprocessor.prepare_audio(audio_data)

                    if self.audio_cache is not None:
                        current_position = self.audio_capture.total_samples if self.audio_capture else 0
                        audio_embeds = self.audio_cache.get_embeddings(
                            audio_tensor, current_position, self.models.audio_encoder
                        )
                    else:
                        audio_embeds = self.preprocessor.encode_audio(
                            self.models.audio_encoder, audio_tensor
                        )
                    self._last_audio_time = ts
                    self._last_audio_embeds = audio_embeds
                else:
                    # Reuse last audio embeddings between encode intervals
                    audio_embeds = getattr(self, '_last_audio_embeds', None)

            # === 7. Detection sequence: (1, 16, 100) ===
            det_sequence = self.detection_buffer.get_sequence()
            detection_tensor = self.preprocessor.prepare_detections(det_sequence)

            # === 8. Action history: (1, 64, 22) ===
            action_tensor = self.preprocessor.prepare_action_history(
                self.action_history.get_history()
            )

            # === 9. State vector: (1, 100) ===
            state_tensor = self.preprocessor.prepare_state_vector(state_vec)

            # === 10. TemporalCrossTransformer forward ===
            mouse_embed, policy_keys, value = self.models.temporal_model(
                radar_seq=radar_embeds,
                scene_seq=scene_embeds,
                detection_seq=detection_tensor,
                action_seq=action_tensor,
                state_vec=state_tensor,
                audio_seq=audio_embeds,
            )

            # === 11. FlowActionHead: sample mouse delta ===
            if hasattr(self.models.flow_head, 'trt_engine'):
                policy_mouse = mouse_embed
            else:
                policy_mouse = self.models.flow_head.sample(mouse_embed)

            # Extract outputs
            mouse_delta_normalized = policy_mouse[0].cpu().numpy()
            key_logits = policy_keys[0].cpu().numpy()
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

        # Denormalize mouse
        # T_INFERENCE = ticks between inferences. With event-driven, use actual elapsed time.
        _MOUSE_SCALE = 25.0
        # Estimate T from actual FPS: T = 64 / actual_fps (64 tick server)
        actual_fps = max(1.0, self._current_fps)
        _T_INFERENCE = max(1, round(64 / actual_fps))
        yaw_degrees = float(mouse_delta_normalized[0]) * _T_INFERENCE * _MOUSE_SCALE
        pitch_degrees = float(mouse_delta_normalized[1]) * _T_INFERENCE * _MOUSE_SCALE

        dx = int(yaw_degrees / self.config.degrees_per_pixel * self.config.mouse_sensitivity)
        dy = int(pitch_degrees / self.config.degrees_per_pixel * self.config.mouse_sensitivity)

        pressed_keys = set()
        for idx, prob in enumerate(key_probs):
            if prob > self.config.key_threshold:
                key = ACTION_KEYS.get(idx)
                if key:
                    pressed_keys.add(key)

        inference_time = (time.time() - start_time) * 1000

        return InferenceResult(
            timestamp=ts,
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

    def _prepare_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Convert uint8 HWC frame to float32 NCHW tensor on device."""
        if frame.dtype == np.uint8:
            frame_f = frame.astype(np.float32) / 255.0
        else:
            frame_f = frame.copy()
        return torch.from_numpy(
            np.transpose(frame_f, (2, 0, 1))
        ).unsqueeze(0).to(self.device)

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

    def _log_diagnostics(self, has_state: bool, has_audio: bool):
        if not self._logger:
            return

        diag_parts = [
            f"fps={self._current_fps:.1f}",
            f"state={'YES' if has_state else 'NO'}",
            f"audio={'YES' if has_audio else 'NO'}",
            f"det_buffer={self.detection_buffer.filled_count}/16",
            f"action_buf={self.action_history.filled_count}/64",
        ]

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
            "running": self._running,
        }
