"""
Configuration constants for the CS2 VLA Agent inference pipeline.

Aligned with final_model/ architecture:
  - TemporalCrossTransformer: d_model=384, depth=4
  - StereoAudioEncoder: 16 sec stereo, 32 embeddings → linspace 16
  - YOLO l-scale: scene_dim=512, nc=2
  - State vector: 100 dim (12 scalars + 2 side + 43 weapon + 43 weapon_list)
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from pathlib import Path


@dataclass
class Config:
    """Main configuration for the inference pipeline."""

    # === Inference settings ===
    device: str = "cuda"

    # === Checkpoint ===
    checkpoint_path: str = "./checkpoints_final/latest/epoch_best.pth"

    # === Screen capture (synchronous, no FPS target) ===
    screen_width: int = 640
    screen_height: int = 640   # square — YOLO expects 640×640
    monitor_index: int = 1  # Primary monitor

    # === Radar crop (from full screenshot) ===
    # (left, top, right, bottom) in pixels
    radar_crop_box: Tuple[int, int, int, int] = (10, 25, 140, 170)
    radar_size: Tuple[int, int] = (224, 224)  # Resize target

    # === Buffer sizes ===
    scene_buffer_size: int = 64  # 64 scene frames (~4 sec @ 16 FPS) → passed to transformer
    radar_buffer_size: int = 32  # 32 radar frames (@1Hz) → linspace to 16 in cache
    state_buffer_size: int = 16  # Number of game states to keep

    # === Modality rates ===
    radar_rate: float = 1.0    # Hz — encode radar every 1 sec
    audio_rate: float = 2.0    # Hz — encode new 0.5 sec audio chunk every 0.5 sec

    # === Audio settings ===
    audio_sample_rate: int = 16000  # Hz
    audio_buffer_duration: float = 16.0  # seconds (StereoAudioEncoder expects 16 sec)
    audio_channels: int = 2  # stereo — CS2 HRTF spatial audio

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
    buy_model_path: str = "./buy_models/buy_v2"

    # === Alive digit detector (MNIST-like CNN for alive counts) ===
    alive_digit_model_path: str = "./alive_digit/best.pt"
    alive_digit_rate: float = 2.0  # Hz — run detector every 0.5s

    # === Paths ===
    final_model_path: Optional[Path] = None

    def __post_init__(self):
        """Set derived paths."""
        base = Path(__file__).parent.parent
        self.final_model_path = base / "final_model"


# Key mapping: model output index -> key name
# Matches DatasetIntent.py intent vector (20 dims)
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


# Model dimensions (aligned with final_model/ architecture)
MODEL_DIMS = {
    "radar_dim": 512,
    "radar_seq": 16,       # 32 raw → linspace 16 → encode → (16, 512)
    "scene_dim": 512,      # YOLO embed_from_features → 512 (NOT 2048)
    "scene_seq": 64,       # 64 raw → ModalityCompressor 64→16 in transformer
    "audio_dim": 512,
    "audio_seq": 16,       # StereoAudioEncoder→32 → linspace→16
    "detection_dim": 100,  # 20 detections * 5 features
    "detection_seq": 16,   # 16 tokens (already linspaced)
    "actions_dim": 22,     # 2 mouse + 20 keys
    "actions_seq": 64,     # 64 raw → ModalityCompressor 64→16 in transformer
    "state_dim": 100,      # 12 scalars + 2 side + 43 weapon + 43 weapon_list
    "d_model": 384,        # transformer hidden dim
}


# State vector indices (100 total)
# 12 scalars + 2 side + 43 weapon + 43 weapon_list = 100
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
    "defuser": 9,
    "score_ct": 10,
    "score_t": 11,
    "side_start": 12,          # 2 one-hot [CT, T]
    "weapon_start": 14,        # 43 one-hot
    "weapon_list_start": 57,   # 43 multi-hot
}


# Weapon list (43 items, matches DatasetIntent.py)
WEAPONS = [
    'p2000', 'usp-s', 'p250', 'five-seven', 'glock-18', 'tec-9',
    'cz75-auto', 'dual berettas', 'desert eagle', 'm249',
    'r8 revolver', 'mp9', 'mac-10', 'pp-bizon', 'mp7',
    'ump-45', 'p90', 'mp5-sd', 'famas', 'galil ar', 'sawed-off',
    'm4a4', 'm4a1-s', 'ak-47', 'aug', 'sg 553', 'ssg 08',
    'awp', 'scar-20', 'g3sg1', 'nova', 'xm1014', 'mag-7',
    'negev', 'knife', 'high explosive grenade', 'flashbang',
    'smoke grenade', 'decoy grenade', 'molotov',
    'incendiary grenade', 'c4 explosive', 'None'
]
