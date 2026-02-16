"""
Configuration constants for the CS2 VLA Agent inference pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from pathlib import Path


@dataclass
class Config:
    """Main configuration for the inference pipeline."""

    # === Inference settings ===
    inference_rate: int = 16  # Hz (inferences per second)
    device: str = "cuda"

    # === Checkpoint ===
    checkpoint_path: str = "./checkpoints2/run_xxx/epoch_10.pth"

    # === Screen capture ===
    screen_width: int = 640
    screen_height: int = 480
    capture_fps: int = 60
    monitor_index: int = 1  # Primary monitor

    # === Radar crop (from full screenshot) ===
    # (left, top, right, bottom) in pixels
    radar_crop_box: Tuple[int, int, int, int] = (10, 25, 140, 170)
    radar_size: Tuple[int, int] = (224, 224)  # Resize target

    # === Buffer sizes ===
    scene_buffer_size: int = 16  # Number of scene frames to keep
    # NOTE: 129 frames causes ~2 FPS. Use 32 for ~8 FPS or 16 for ~16 FPS
    radar_buffer_size: int = 32  # Number of radar frames to keep (reduced for perf)
    state_buffer_size: int = 16  # Number of game states to keep

    # === Audio settings ===
    audio_sample_rate: int = 16000  # Hz
    audio_buffer_duration: float = 30.0  # seconds
    audio_channels: int = 1  # mono

    # === Mouse control ===
    mouse_sensitivity: float = 1.0  # Multiplier for mouse delta
    # CS2 default: ~0.022 degrees per pixel at sens 1.0
    degrees_per_pixel: float = 0.022

    # === Keyboard control ===
    key_threshold: float = 0.5  # Probability threshold for key press

    # === GSI (Game State Integration) ===
    gsi_host: str = "127.0.0.1"
    gsi_port: int = 3000
    gsi_auth_token: str = "cs2nn_secret_token"

    # === Features ===
    use_audio: bool = True
    use_buy: bool = False
    use_overlay: bool = True
    apply_actions: bool = True  # Set False for debug/observation mode

    # === TensorRT ===
    use_trt: bool = False  # Use TensorRT FP16 engines for faster inference
    trt_dir: str = "./trt_engines"  # Directory containing .trt engine files

    # === Buy agent ===
    buy_model_path: str = "./buy_models/buy_predictor_v1"

    # === Paths ===
    src_path: Optional[Path] = None
    audio_src_path: Optional[Path] = None

    def __post_init__(self):
        """Set derived paths."""
        base = Path(__file__).parent.parent
        self.src_path = base / "src"
        self.audio_src_path = base / "audio_adaptation" / "src"


# Key mapping: model output index -> key name
# Based on DatasetIntent.py action encoding
ACTION_KEYS: Dict[int, str] = {
    0: "mouse1",      # fire
    1: "mouse2",      # second_fire (scope/alt fire)
    2: "w",           # forward
    3: "s",           # back
    4: "a",           # left
    5: "d",           # right
    6: "space",       # jump
    7: "ctrl",        # crouch
    8: "shift",       # walk (shift)
    9: "1",           # weapon1 (primary)
    10: "2",          # weapon2 (secondary/pistol)
    11: "3",          # weapon3 (knife)
    12: "5",          # c4 (bomb)
    13: "r",          # reload
    14: "4",          # HE grenade
    15: "mouse4",     # molotov (side button, rebindable)
    16: "mouse5",     # smoke (side button, rebindable)
    17: "f",          # flash
    18: "6",          # decoy
    19: "e",          # use/interact
}

# Action names for logging/display
ACTION_NAMES = [
    "fire", "second_fire", "forward", "back", "left", "right",
    "jump", "crouch", "shift", "weapon1", "weapon2", "weapon3",
    "c4", "reload", "he", "molotov", "smoke", "flash", "decoy", "use"
]


# Scan codes for keyboard input (more reliable for games than virtual key codes)
SCAN_CODES: Dict[str, int] = {
    # Movement
    'w': 0x11,       # W
    's': 0x1F,       # S
    'a': 0x1E,       # A
    'd': 0x20,       # D
    'space': 0x39,   # Space
    'ctrl': 0x1D,    # Left Ctrl
    'shift': 0x2A,   # Left Shift

    # Actions
    'r': 0x13,       # R (reload)
    'e': 0x12,       # E (use)
    'f': 0x21,       # F (flash)

    # Weapons/Numbers
    '1': 0x02,
    '2': 0x03,
    '3': 0x04,
    '4': 0x05,
    '5': 0x06,
    '6': 0x07,

    # Console
    '`': 0x29,       # Tilde (console)
}


# Model dimensions (from CLAUDE.md)
MODEL_DIMS = {
    "radar_dim": 512,
    "radar_seq": 32,  # Reduced from 129 for inference performance
    "scene_dim": 2048,
    "scene_seq": 16,
    "audio_dim": 512,
    "audio_seq": 60,
    "detection_dim": 100,  # 20 detections * 5 features
    "detection_seq": 1,
    "actions_dim": 22,  # 2 mouse + 20 keys
    "actions_seq": 16,
    "state_dim": 95,  # 9 scalars + 2 side + 42 weapon + 42 weapon_list
    "d_model": 512,
}


# State vector indices (95 total)
STATE_INDICES = {
    "hp": 0,
    "armor": 1,
    "helmet": 2,
    "ammo": 3,
    "ct_alive": 4,
    "t_alive": 5,
    "round_time_left": 6,
    "bomb_planted": 7,
    "freeze_time": 8,
    "side_start": 9,      # 2 one-hot
    "weapon_start": 11,   # 42 one-hot
    "weapon_list_start": 53,  # 42 multi-hot
}
