"""
Main inference engine that orchestrates all components.
Runs the inference loop at 16 Hz and coordinates input capture and action output.
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

from ..config import Config, ACTION_KEYS, ACTION_NAMES
from ..capture.screen_capture import ScreenCapture
from ..capture.audio_capture import AudioCapture, get_audio_capture
from ..buffers.ring_buffers import FrameBuffer, AudioBuffer, ActionHistoryBuffer, StateBuffer
from ..models.model_loader import load_models, ModelBundle
from .preprocessor import Preprocessor
from .embedding_cache import GPUEmbeddingCache, AudioEmbeddingCache


class MillisecondFormatter(logging.Formatter):
    """Formatter with milliseconds support."""
    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            s = ct.strftime("%H:%M:%S")
        return f"{s}.{int(record.msecs):03d}"


def setup_inference_logger(log_dir: str = "./inference_logs") -> logging.Logger:
    """Setup logger for inference results."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"inference_{timestamp}.log"

    logger = logging.getLogger("inference")
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicates on restart
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(MillisecondFormatter('%(asctime)s | %(message)s'))

    logger.addHandler(fh)

    # Write header
    logger.debug("=== Inference Log Started ===")

    print(f"[Engine] Logging to: {log_file}")
    return logger


@dataclass
class InferenceResult:
    """Result from a single inference step."""
    timestamp: float

    # Raw model outputs
    mouse_delta: tuple  # (yaw_delta, pitch_delta) in degrees
    key_logits: np.ndarray  # (20,) raw logits
    key_probs: np.ndarray  # (20,) after sigmoid
    value: float

    # Processed outputs
    mouse_pixels: tuple  # (dx, dy) in pixels
    pressed_keys: set  # Set of key names to press

    # Debug info
    inference_time_ms: float
    fps: float


class InferenceEngine:
    """
    Main inference engine for the CS2 VLA Agent.

    Coordinates:
    - Screen capture (scene + radar frames)
    - Audio capture (30 sec rolling buffer)
    - GSI game state (from separate server)
    - Model inference at 16 Hz
    - Action output (mouse + keyboard)
    """

    def __init__(
        self,
        config: Config,
        checkpoint_path: str,
        device: str = "cuda",
        use_audio: bool = True,
        use_buy: bool = False,
    ):
        """
        Initialize inference engine.

        Args:
            config: Configuration object
            checkpoint_path: Path to model checkpoint
            device: Compute device
            use_audio: Whether to use audio encoder
            use_buy: Whether to enable auto-buy
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.use_audio = use_audio
        self.use_buy = use_buy

        # State
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

        # Buffers
        self.scene_buffer = FrameBuffer(max_size=config.scene_buffer_size)
        self.radar_buffer = FrameBuffer(max_size=config.radar_buffer_size)
        self.audio_buffer = AudioBuffer(
            sample_rate=config.audio_sample_rate,
            duration=config.audio_buffer_duration
        )
        self.action_history = ActionHistoryBuffer(max_size=16)
        self.state_buffer = StateBuffer(max_size=16)

        # Embedding caches (initialized in start())
        self.radar_cache: Optional[GPUEmbeddingCache] = None
        self.scene_cache: Optional[GPUEmbeddingCache] = None
        self.audio_cache: Optional[AudioEmbeddingCache] = None
        self._cache_enabled = True  # Can be disabled for validation

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
        """Set mouse controller for action output."""
        self.mouse_controller = controller

    def set_keyboard_controller(self, controller):
        """Set keyboard controller for action output."""
        self.keyboard_controller = controller

    def set_overlay(self, overlay):
        """Set debug overlay."""
        self.overlay = overlay

    def set_buy_executor(self, executor):
        """Set buy executor for auto-buy."""
        self.buy_executor = executor

    def register_result_callback(self, callback: Callable[[InferenceResult], None]):
        """Register callback for inference results."""
        self._result_callbacks.append(callback)

    def update_game_state(self, state_vec: np.ndarray):
        """
        Update current game state (called by GSI adapter).

        Args:
            state_vec: (95,) state vector
        """
        with self._state_lock:
            self._current_state_vec = state_vec.copy()
            self.state_buffer.add(time.time(), state_vec)

    def start(self):
        """Start inference engine."""
        if self._running:
            print("[Engine] Already running")
            return

        print("[Engine] Starting...")

        # Setup logger
        self._logger = setup_inference_logger()

        # Load models
        print("[Engine] Loading models...")
        self.models = load_models(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            use_audio=self.use_audio,
            use_trt=self.config.use_trt,
            trt_dir=self.config.trt_dir,
            src_path=self.config.src_path,
            audio_src_path=self.config.audio_src_path,
        )

        # Verify models are in eval mode
        self._verify_model_modes()

        # Initialize preprocessor
        self.preprocessor = Preprocessor(
            device=self.device,
            radar_seq_len=self.config.radar_buffer_size,
            scene_seq_len=self.config.scene_buffer_size,
        )

        # Initialize embedding caches
        if self._cache_enabled:
            print("[Engine] Initializing embedding caches...")

            self.radar_cache = GPUEmbeddingCache(
                capacity=256,  # ~16 sec @ 16 FPS
                emb_dim=512,
                device=self.device,
                enable_validation=False  # Disable for production
            )

            self.scene_cache = GPUEmbeddingCache(
                capacity=128,  # ~8 sec @ 16 FPS
                emb_dim=2048,
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
            print(f"  Radar cache: {radar_stats['memory_mb']:.1f} MB")
            print(f"  Scene cache: {scene_stats['memory_mb']:.1f} MB")
            if self.use_audio:
                print(f"  Audio cache: 0.12 MB (on-demand)")

        # Initialize screen capture
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

        # Initialize audio capture (optional)
        if self.use_audio:
            print("[Engine] Starting audio capture...")
            self.audio_capture = get_audio_capture(
                sample_rate=self.config.audio_sample_rate,
                buffer_duration=self.config.audio_buffer_duration,
            )
            self.audio_capture.set_callback(self._on_audio_chunk)
            self.audio_capture.start()

        # Start inference thread
        self._running = True
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

        print(f"[Engine] Started at {self.config.inference_rate} Hz inference rate")

    def stop(self):
        """Stop inference engine."""
        print("[Engine] Stopping...")

        self._running = False

        # Stop components
        if self.screen_capture:
            self.screen_capture.stop()

        if self.audio_capture:
            self.audio_capture.stop()

        # Wait for inference thread
        if self._inference_thread:
            self._inference_thread.join(timeout=2.0)

        # Release all keys
        if self.keyboard_controller:
            self.keyboard_controller.release_all()

        print(f"[Engine] Stopped. Total inferences: {self._inference_count}")

    def _on_scene_frame(self, timestamp: float, frame: np.ndarray):
        """Callback for new scene frame."""
        self.scene_buffer.add(timestamp, frame)

    def _on_radar_frame(self, timestamp: float, frame: np.ndarray):
        """Callback for new radar frame."""
        self.radar_buffer.add(timestamp, frame)

    def _on_audio_chunk(self, chunk: np.ndarray):
        """Callback for new audio chunk."""
        self.audio_buffer.add(chunk)

    def _inference_loop(self):
        """Main inference loop running at target Hz."""
        target_interval = 1.0 / self.config.inference_rate
        fps_window = []

        while self._running:
            loop_start = time.time()

            try:
                result = self._run_inference()

                if result is not None:
                    # Apply actions if enabled
                    if self.config.apply_actions:
                        self._apply_actions(result)

                    # Update action history
                    self.action_history.add(
                        np.array(result.mouse_delta, dtype=np.float32),
                        result.key_probs
                    )

                    # Dispatch to callbacks
                    for callback in self._result_callbacks:
                        try:
                            callback(result)
                        except Exception as e:
                            print(f"[Engine] Callback error: {e}")

                    # Update overlay
                    if self.overlay:
                        self._update_overlay(result)

                    # Log to file
                    self._log_inference(result)

                    self._inference_count += 1

            except Exception as e:
                print(f"[Engine] Inference error: {e}")
                import traceback
                traceback.print_exc()

            # Calculate FPS
            elapsed = time.time() - loop_start
            fps_window.append(elapsed)
            if len(fps_window) > 30:
                fps_window.pop(0)
            self._current_fps = 1.0 / (sum(fps_window) / len(fps_window)) if fps_window else 0

            # Sleep to maintain target rate
            sleep_time = max(0, target_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _should_invalidate_cache(self, new_state_vec: Optional[np.ndarray]) -> bool:
        """
        Determine if cache should be invalidated due to game state change.

        Invalidate on:
        - Round restart (detected by state vec changes)
        - Large positional jump (teleport, respawn)
        - Game restart

        Args:
            new_state_vec: Current state vector or None

        Returns:
            True if cache should be invalidated
        """
        # TODO: Implement proper invalidation logic based on state_vec parsing
        # For now, never invalidate to maximize cache hit rate
        # Future improvements:
        # - Parse state_vec for round number (detect round change)
        # - Detect HP going from 0 -> 100 (respawn)
        # - Detect freeze_time changes
        return False

    def _run_inference(self) -> Optional[InferenceResult]:
        """Run a single inference step."""
        start_time = time.time()

        # Check if we have enough data
        if len(self.scene_buffer) < 1:
            return None

        # Get current state
        with self._state_lock:
            state_vec = self._current_state_vec

        # Check if cache should be invalidated
        if self._should_invalidate_cache(state_vec):
            if self._logger:
                self._logger.debug("[CACHE] Invalidated (round restart detected)")
            if self.radar_cache:
                self.radar_cache.invalidate()
            if self.scene_cache:
                self.scene_cache.invalidate()
            if self.audio_cache:
                self.audio_cache.invalidate()

        # Prepare inputs
        scene_frames = self.scene_buffer.get_recent(self.config.scene_buffer_size)
        radar_frames = self.radar_buffer.get_recent(self.config.radar_buffer_size)

        # Log diagnostics every 50 inferences
        if self._inference_count % 50 == 0:
            self._log_diagnostics(
                scene_len=len(scene_frames),
                radar_len=len(radar_frames),
                has_state=(state_vec is not None),
                has_audio=(self.use_audio and self.audio_capture is not None)
            )

        # Prepare tensors
        scene_tensor = self.preprocessor.prepare_scene_sequence(scene_frames)
        radar_tensor = self.preprocessor.prepare_radar_sequence(radar_frames)
        state_tensor = self.preprocessor.prepare_state_vector(state_vec)
        action_tensor = self.preprocessor.prepare_action_history(
            self.action_history.get_history()
        )

        with torch.no_grad():
            # === Encode radar with cache ===
            if self._cache_enabled and self.radar_cache is not None:
                # Extract only the latest frame from radar_tensor (1, T, 3, 224, 224)
                latest_radar = radar_tensor[:, -1]  # (1, 3, 224, 224)

                # Encode 1 frame + retrieve cached sequence
                radar_seq_cached = self.radar_cache.add_and_get_sequence(
                    latest_radar,
                    self.models.radar_encoder,
                    seq_length=self.config.radar_buffer_size,
                    timestamp=time.time()
                )  # (seq_length, 512)

                # Add batch dimension for temporal_model compatibility
                radar_embeds = radar_seq_cached.unsqueeze(0)  # (1, seq_length, 512)
            else:
                # Fallback to full encoding (for validation or if cache disabled)
                radar_embeds = self.preprocessor.encode_radar_sequence(
                    self.models.radar_encoder, radar_tensor
                )

            # === Encode scene with cache ===
            if self._cache_enabled and self.scene_cache is not None:
                # Extract only the latest scene frame
                latest_scene = scene_tensor[:, -1]  # (1, 3, H, W)

                # Get embeddings from YOLO for caching
                # We need to use yolo.embeds() method (returns only embeddings)
                def get_scene_embedding(frame):
                    """Extract scene embedding from YOLO."""
                    # YOLO forward returns (detections, embeddings)
                    _, embeds = self.models.yolo(frame)
                    return embeds

                scene_seq_cached = self.scene_cache.add_and_get_sequence(
                    latest_scene,
                    get_scene_embedding,
                    seq_length=self.config.scene_buffer_size,
                    timestamp=time.time()
                )  # (seq_length, 2048)

                # Add batch dimension
                scene_embeds = scene_seq_cached.unsqueeze(0)  # (1, seq_length, 2048)

                # Get detections from latest frame only
                # Detections are always from the most recent frame
                detections = torch.zeros(1, 1, 100, device=self.device)
            else:
                # Fallback to full encoding
                scene_embeds, detections = self.preprocessor.encode_scene_sequence(
                    self.models.yolo, scene_tensor
                )

            # === Encode audio with cache ===
            audio_embeds = None
            if self.use_audio and self.models.audio_encoder is not None:
                audio_data = self.audio_capture.get_buffer() if self.audio_capture else np.zeros(480000)
                audio_tensor = self.preprocessor.prepare_audio(audio_data)

                if self._cache_enabled and self.audio_cache is not None:
                    # Use audio cache
                    # Get total samples processed (current position in audio stream)
                    current_position = self.audio_buffer._total_samples if self.audio_capture else 0

                    audio_embeds = self.audio_cache.get_embeddings(
                        audio_tensor,
                        current_position,
                        self.models.audio_encoder
                    )
                else:
                    # Fallback to direct encoding
                    audio_embeds = self.preprocessor.encode_audio(
                        self.models.audio_encoder, audio_tensor
                    )

            # Run temporal transformer
            if self.use_audio and audio_embeds is not None:
                mouse_embed, policy_keys, value = self.models.temporal_model(
                    radar_seq=radar_embeds,
                    scene_seq=scene_embeds,
                    detection_seq=detections,
                    action_seq=action_tensor,
                    state_vec=state_tensor,
                    audio_seq=audio_embeds,
                )
            else:
                mouse_embed, policy_keys, value = self.models.temporal_model(
                    radar_seq=radar_embeds,
                    scene_seq=scene_embeds,
                    detection_seq=detections,
                    action_seq=action_tensor,
                    state_vec=state_tensor,
                )

            # Sample mouse action via FlowActionHead
            # Note: If using TRT, this is already done inside the engine
            if hasattr(self.models.flow_head, 'trt_engine'):
                # TRT version already returns final policy_mouse
                policy_mouse = mouse_embed
            else:
                # PyTorch version needs flow matching sampling
                policy_mouse = self.models.flow_head.sample(mouse_embed)

            # Extract outputs
            mouse_delta = policy_mouse[0].cpu().numpy()  # (2,)
            key_logits = policy_keys[0].cpu().numpy()  # (20,)
            key_probs = torch.sigmoid(policy_keys[0]).cpu().numpy()
            value_pred = value[0, 0].item() if value.dim() > 1 else value[0].item()

            # Debug: Log raw model outputs for first 5 inferences
            if self._inference_count < 5:
                print(f"\n[DEBUG] Inference #{self._inference_count}")
                print(f"  Input shapes: radar={radar_embeds.shape}, scene={scene_embeds.shape}")
                print(f"  Raw mouse output: {mouse_delta}")
                print(f"  Raw key logits (first 5): {key_logits[:5]}")
                print(f"  Key logits stats: min={key_logits.min():.4f}, max={key_logits.max():.4f}, mean={key_logits.mean():.4f}")
                print(f"  Key probs stats: min={key_probs.min():.4f}, max={key_probs.max():.4f}")

        # Convert mouse delta to pixels
        yaw_delta, pitch_delta = mouse_delta[0], mouse_delta[1]
        dx = int(yaw_delta / self.config.degrees_per_pixel * self.config.mouse_sensitivity)
        dy = int(pitch_delta / self.config.degrees_per_pixel * self.config.mouse_sensitivity)

        # Determine which keys to press
        pressed_keys = set()
        for idx, prob in enumerate(key_probs):
            if prob > self.config.key_threshold:
                key = ACTION_KEYS.get(idx)
                if key:
                    pressed_keys.add(key)

        inference_time = (time.time() - start_time) * 1000  # ms

        return InferenceResult(
            timestamp=time.time(),
            mouse_delta=(float(yaw_delta), float(pitch_delta)),
            key_logits=key_logits,
            key_probs=key_probs,
            value=value_pred,
            mouse_pixels=(dx, dy),
            pressed_keys=pressed_keys,
            inference_time_ms=inference_time,
            fps=self._current_fps,
        )

    def _apply_actions(self, result: InferenceResult):
        """Apply inference result as game inputs."""
        # Apply mouse movement
        if self.mouse_controller:
            self.mouse_controller.apply_delta(
                result.mouse_delta[0],
                result.mouse_delta[1]
            )

        # Apply keyboard actions
        if self.keyboard_controller:
            self.keyboard_controller.apply_actions(
                torch.from_numpy(result.key_logits)
            )

    def _update_overlay(self, result: InferenceResult):
        """Update debug overlay with inference result."""
        if not self.overlay:
            return

        # Build key probs dict
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
        """Log inference result to file."""
        if not self._logger:
            return

        # Format key probabilities
        key_probs_str = " | ".join([
            f"{ACTION_NAMES[i]}:{result.key_probs[i]:.3f}"
            for i in range(min(len(ACTION_NAMES), len(result.key_probs)))
        ])

        # Format pressed keys
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
        """Log diagnostic info about inputs."""
        if not self._logger:
            return

        diag_parts = [
            f"scene_frames={scene_len}",
            f"radar_frames={radar_len}",
            f"state={'YES' if has_state else 'NO'}",
            f"audio={'YES' if has_audio else 'NO'}"
        ]

        # Add cache statistics if enabled
        if self._cache_enabled:
            if self.radar_cache:
                radar_stats = self.radar_cache.get_stats()
                diag_parts.append(
                    f"radar_cache={radar_stats['count']}/{radar_stats['capacity']} "
                    f"({radar_stats['filled_ratio']*100:.0f}%)"
                )

            if self.scene_cache:
                scene_stats = self.scene_cache.get_stats()
                diag_parts.append(
                    f"scene_cache={scene_stats['count']}/{scene_stats['capacity']} "
                    f"({scene_stats['filled_ratio']*100:.0f}%)"
                )

            if self.audio_cache and self.use_audio:
                audio_stats = self.audio_cache.get_stats()
                hit_rate = audio_stats['hit_rate'] * 100
                diag_parts.append(
                    f"audio_cache_hits={audio_stats['cache_hits']}/{audio_stats['cache_hits']+audio_stats['cache_misses']} "
                    f"({hit_rate:.0f}%)"
                )

        self._logger.debug(f"DIAG: {' | '.join(diag_parts)}")

    def _verify_model_modes(self):
        """Verify all models are in eval mode and log diagnostics."""
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
            is_training = model.training
            status = "TRAIN" if is_training else "EVAL"
            if is_training:
                all_eval = False
                print(f"  [!] {name}: {status} <-- PROBLEM!")
            else:
                print(f"  [OK] {name}: {status}")

        if not all_eval:
            print("[Engine] WARNING: Some models are NOT in eval mode!")
            print("[Engine] Forcing eval mode...")
            self.models.eval()
            # Verify again
            for name, model in models_info:
                if model.training:
                    print(f"  [FAIL] {name} still in training mode!")
        else:
            print("[Engine] All models in EVAL mode - OK")

        print("[Engine] ================================\n")

    @property
    def is_running(self) -> bool:
        """Whether engine is running."""
        return self._running

    @property
    def stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "inference_count": self._inference_count,
            "fps": self._current_fps,
            "scene_buffer_size": len(self.scene_buffer),
            "radar_buffer_size": len(self.radar_buffer),
            "audio_buffer_filled": self.audio_buffer.get_filled_ratio() if self.audio_capture else 0,
            "running": self._running,
        }


def test_engine():
    """Test engine initialization (requires checkpoint)."""
    print("Testing InferenceEngine...")

    config = Config()
    config.apply_actions = False  # Don't actually send inputs

    try:
        engine = InferenceEngine(
            config=config,
            checkpoint_path="./checkpoints2/test/epoch_1.pth",
            device="cpu",
            use_audio=False,
        )
        print("Engine initialized successfully")
    except Exception as e:
        print(f"Engine initialization failed (expected without checkpoint): {e}")


if __name__ == "__main__":
    test_engine()
