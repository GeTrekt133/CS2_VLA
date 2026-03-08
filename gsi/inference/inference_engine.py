"""
Inference Engine - real-time model inference using GSI data.

Loads trained models and runs predictions during gameplay.
"""
import sys
import time
import threading
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2
import torch

from ..server.http_server import GSIServer, GSIPayload
from ..adapter.state_adapter import GSIStateAdapter, GameStateSnapshot
from ..buffer.state_buffer import StateBuffer, FrameBuffer
from ..config.gsi_config import GSIConfig


@dataclass
class InferenceResult:
    """Model inference result."""

    timestamp: float

    # Mouse prediction (yaw_delta, pitch_delta)
    mouse_delta: tuple  # (yaw, pitch)

    # Key predictions (probabilities for 20 actions)
    key_probs: Dict[str, float]

    # Most likely actions
    predicted_keys: List[str]

    # Value estimate
    value: float

    # Game state info
    hp: int
    round_num: int
    phase: str


# Action names matching model output
ACTION_NAMES = [
    "fire", "jump", "crouch", "forward", "back",
    "use", "left", "right", "second_fire", "reload",
    "molotov", "tab", "interact",
    "weapon1", "weapon2", "weapon3",
    "he", "flash", "smoke", "decoy"
]


class GSIInferenceEngine:
    """
    Real-time inference using GSI data.

    Loads trained models and runs predictions during gameplay.

    Usage:
        engine = GSIInferenceEngine(
            checkpoint_path="./checkpoints2/run_xxx/epoch_10.pth"
        )
        engine.register_callback(on_prediction)
        engine.start()
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[GSIConfig] = None,
        device: str = "cuda",
    ):
        self.config = config or GSIConfig()
        self.device = device
        self.checkpoint_path = checkpoint_path

        # Components
        self.server = GSIServer(
            host=self.config.host,
            port=self.config.port,
            auth_token=self.config.auth_token,
        )
        self.adapter = GSIStateAdapter()
        self.state_buffer = StateBuffer(size=self.config.state_buffer_size)
        self.frame_buffer = FrameBuffer(size=self.config.frame_buffer_size)

        # Models (loaded on start)
        self.radar_encoder = None
        self.yolo = None
        self.temporal_model = None
        self._models_loaded = False

        # Callbacks
        self._callbacks: List[Callable[[InferenceResult], None]] = []

        # State
        self._running = False
        self._last_inference_time = 0
        self._inference_interval = 1 / 16  # 16 Hz inference rate
        self._inference_count = 0

        # Screen capture
        self._capture_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None
        self._sct = None

        # Current state
        self._current_snapshot: Optional[GameStateSnapshot] = None

    def _load_models(self):
        """Load trained models from checkpoint."""
        print(f"[Inference] Loading models from {self.checkpoint_path}")

        # Add src to path
        src_path = Path(__file__).parent.parent.parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        try:
            from RadarEncoder import RadarEncoderEffB0
            from Yolo import DetectionModel
            from TemporalTransformer import TemporalCrossTransformer

            # Load checkpoint
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False
            )

            # Initialize models
            self.radar_encoder = RadarEncoderEffB0(
                pretrained=False, in_ch=3, out_ch=64, embed_dim=512
            ).to(self.device)
            self.radar_encoder.load_state_dict(
                checkpoint["radar_encoder"], strict=False
            )
            self.radar_encoder.eval()

            # YOLO
            yolo_cfg = src_path / "yolo11n.yaml"
            self.yolo = DetectionModel(cfg=str(yolo_cfg)).to(self.device)
            self.yolo.load_state_dict(checkpoint["yolo"], strict=False)
            self.yolo.eval()

            # Temporal model
            self.temporal_model = TemporalCrossTransformer().to(self.device)
            self.temporal_model.load_state_dict(
                checkpoint["temporal_model"], strict=False
            )
            self.temporal_model.eval()

            self._models_loaded = True
            print("[Inference] Models loaded successfully")

        except Exception as e:
            print(f"[Inference] Failed to load models: {e}")
            raise

    def _init_screen_capture(self):
        """Initialize screen capture."""
        try:
            import mss
            self._sct = mss.mss()
            return True
        except ImportError:
            print("[Inference] WARNING: mss not installed")
            return False

    def register_callback(self, callback: Callable[[InferenceResult], None]):
        """Register callback for inference results."""
        self._callbacks.append(callback)

    def start(self):
        """Start inference engine."""
        if self._running:
            print("[Inference] Already running")
            return

        # Load models
        if not self._models_loaded:
            self._load_models()

        # Initialize screen capture
        self._init_screen_capture()

        # Start GSI server
        self.server.register_callback(self._on_gsi_update)
        self.server.start()

        # Start threads
        self._running = True

        if self._sct:
            self._capture_thread = threading.Thread(
                target=self._capture_loop, daemon=True
            )
            self._capture_thread.start()

        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self._inference_thread.start()

        print("[Inference] Engine started")
        print(f"[Inference] Listening on port {self.config.port}")

    def _on_gsi_update(self, payload: GSIPayload):
        """Handle GSI updates."""
        if not self._running:
            return

        snapshot = self.adapter.parse(payload)
        if snapshot is None:
            return

        self._current_snapshot = snapshot
        state_vec = self.adapter.to_state_vec(snapshot)
        self.state_buffer.add(snapshot, state_vec)

    def _capture_loop(self):
        """Screen capture loop for visual input."""
        if self._sct is None:
            return

        monitor = self._sct.monitors[1]
        target_interval = 1.0 / self.config.capture_fps

        while self._running:
            try:
                start = time.time()

                screenshot = self._sct.grab(monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                frame = cv2.resize(
                    frame,
                    (self.config.screen_width, self.config.screen_height)
                )

                self.frame_buffer.add(time.time(), frame)

                elapsed = time.time() - start
                sleep_time = max(0, target_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                print(f"[Inference] Capture error: {e}")
                time.sleep(0.1)

    def _inference_loop(self):
        """Main inference loop."""
        while self._running:
            current_time = time.time()

            # Rate limit inference
            if current_time - self._last_inference_time < self._inference_interval:
                time.sleep(0.001)
                continue

            try:
                result = self._run_inference()
                if result:
                    self._dispatch_result(result)
                    self._last_inference_time = current_time
                    self._inference_count += 1

            except Exception as e:
                print(f"[Inference] Error: {e}")

            time.sleep(0.01)

    def _run_inference(self) -> Optional[InferenceResult]:
        """Run a single inference step."""
        # Get current state
        current_state = self.state_buffer.get_current()
        if current_state is None:
            return None

        snapshot = current_state.snapshot

        # Get recent frames
        recent_frames = self.frame_buffer.get_recent(16)
        if len(recent_frames) < 1:
            return None

        # Get state history
        state_vecs, _ = self.state_buffer.get_recent(16)
        if len(state_vecs) < 1:
            return None

        with torch.no_grad():
            # Prepare scene sequence (16 frames)
            scene_frames = []
            for ts, frame in recent_frames[-16:]:
                frame_tensor = torch.from_numpy(frame).float() / 255.0
                scene_frames.append(frame_tensor)

            # Pad if needed
            while len(scene_frames) < 16:
                if scene_frames:
                    scene_frames.insert(0, scene_frames[0])
                else:
                    scene_frames.append(torch.zeros(
                        self.config.screen_height,
                        self.config.screen_width,
                        3
                    ))

            # (B, T, H, W, C) -> (B, T, C, H, W)
            scene_seq = torch.stack(scene_frames).unsqueeze(0)
            scene_seq = scene_seq.permute(0, 1, 4, 2, 3).to(self.device)

            # Prepare radar sequence (crop from scene)
            radar_frames = []
            l, t, r, b = self.config.radar_crop_box

            for ts, frame in recent_frames[-129:]:
                radar = frame[t:b, l:r, :]
                radar = cv2.resize(radar, self.config.radar_size)
                radar_tensor = torch.from_numpy(radar).float() / 255.0
                radar_frames.append(radar_tensor)

            # Pad to 129 frames
            while len(radar_frames) < 129:
                if radar_frames:
                    radar_frames.insert(0, radar_frames[0])
                else:
                    radar_frames.append(torch.zeros(224, 224, 3))

            radar_seq = torch.stack(radar_frames).unsqueeze(0)
            radar_seq = radar_seq.permute(0, 1, 4, 2, 3).to(self.device)

            # State vector
            state_vec = torch.from_numpy(state_vecs[-1]).unsqueeze(0).to(self.device)

            # Encode radar (batch process)
            B, T_radar = radar_seq.shape[:2]
            radar_flat = radar_seq.view(B * T_radar, 3, 224, 224)
            radar_embeds_flat = self.radar_encoder(radar_flat)
            radar_embeds = radar_embeds_flat.view(B, T_radar, -1)

            # Encode scene
            B, T_scene = scene_seq.shape[:2]
            scene_flat = scene_seq.view(B * T_scene, 3, self.config.screen_height, self.config.screen_width)
            _, scene_embeds_flat = self.yolo(scene_flat)
            scene_embeds = scene_embeds_flat.view(B, T_scene, -1)

            # Detection (simplified - zeros)
            detect_embeds = torch.zeros(1, 1, 100).to(self.device)

            # Action history (zeros for now)
            action_seq = torch.zeros(1, 16, 22).to(self.device)

            # Run temporal model
            policy_mouse, policy_keys, value = self.temporal_model(
                radar_seq=radar_embeds,
                scene_seq=scene_embeds,
                detection_seq=detect_embeds,
                action_seq=action_seq,
                state_vec=state_vec
            )

            # Process outputs
            mouse_delta = policy_mouse[0].cpu().numpy()
            key_probs = torch.sigmoid(policy_keys[0]).cpu().numpy()
            value_pred = value[0, 0].item()

            # Get predicted keys (threshold 0.5)
            predicted_keys = [
                ACTION_NAMES[i]
                for i, prob in enumerate(key_probs)
                if prob > 0.5 and i < len(ACTION_NAMES)
            ]

            # Create result
            result = InferenceResult(
                timestamp=time.time(),
                mouse_delta=(float(mouse_delta[0]), float(mouse_delta[1])),
                key_probs={
                    name: float(key_probs[i])
                    for i, name in enumerate(ACTION_NAMES)
                    if i < len(key_probs)
                },
                predicted_keys=predicted_keys,
                value=value_pred,
                hp=snapshot.hp,
                round_num=snapshot.round_num,
                phase=snapshot.phase,
            )

            return result

    def _dispatch_result(self, result: InferenceResult):
        """Send result to all callbacks."""
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                print(f"[Inference] Callback error: {e}")

    def stop(self):
        """Stop inference engine."""
        self._running = False
        self.server.stop()

        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self._inference_thread:
            self._inference_thread.join(timeout=2.0)

        print(f"[Inference] Engine stopped. Total inferences: {self._inference_count}")

    @property
    def current_state(self) -> Optional[GameStateSnapshot]:
        """Get current game state."""
        return self._current_snapshot

    @property
    def stats(self) -> Dict:
        """Get engine statistics."""
        return {
            "inference_count": self._inference_count,
            "state_buffer_size": len(self.state_buffer),
            "frame_buffer_size": len(self.frame_buffer),
            "models_loaded": self._models_loaded,
            "running": self._running,
        }
