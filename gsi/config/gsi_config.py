"""
GSI Configuration and constants.
"""
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class GSIConfig:
    """CS2 GSI Configuration."""

    # Server settings
    host: str = "127.0.0.1"
    port: int = 3000
    auth_token: str = "cs2nn_secret_token"

    # Timing (match CS2 GSI config)
    timeout: float = 5.0
    buffer_delay: float = 0.1
    throttle: float = 0.1  # Min time between updates (seconds)
    heartbeat: float = 10.0

    # Data collection settings
    output_dir: str = "./gsi_data"
    save_screenshots: bool = True
    screenshot_interval: int = 4  # Every N ticks

    # Inference settings
    checkpoint_path: Optional[str] = None
    device: str = "cuda"

    # Buffer settings
    state_buffer_size: int = 256
    frame_buffer_size: int = 64

    # Screen capture
    screen_width: int = 640
    screen_height: int = 640
    capture_fps: int = 64

    # Radar crop (from DatasetIntent.py)
    radar_crop_box: tuple = (10, 25, 140, 170)
    radar_size: tuple = (224, 224)

    # CS2 paths (default Steam location)
    cs2_cfg_dir: str = r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg"


# ==================== CONSTANTS ====================
# Match exactly with src/DatasetIntent.py

KEYS = [
    "MOUSE_LEFT", "SPACE", "CTRL", "W", "S", "E", "A", "D",
    "MOUSE_RIGHT", "R", "MOLOTOV", "TAB", "F",
    "WEAPON1", "WEAPON2", "WEAPON3",
    "HE", "FLASH", "SMOKE", "DECOY", "C4", "SHIFT"
]

WEAPONS = [
    'p2000', 'p250', 'five-seven', 'glock-18', 'tec-9',
    'cz75-auto', 'dual berettas', 'desert eagle', 'm249',
    'r8 revolver', 'mp9', 'mac-10', 'pp-bizon', 'mp7',
    'ump-45', 'p90', 'mp5-sd', 'famas', 'galil ar', 'sawed-off',
    'm4a4', 'm4a1-s', 'ak-47', 'aug', 'sg 553', 'ssg 08',
    'awp', 'scar-20', 'g3sg1', 'nova', 'xm1014', 'mag-7',
    'negev', 'knife', 'high explosive grenade', 'flashbang',
    'smoke grenade', 'decoy grenade', 'molotov',
    'incendiary grenade', 'c4 explosive', 'None'
]

SIDES = ["CT", "T"]

# GSI weapon name -> project weapon name mapping
GSI_WEAPON_MAP = {
    # Pistols
    "weapon_hkp2000": "p2000",
    "weapon_usp_silencer": "p2000",
    "weapon_p250": "p250",
    "weapon_fiveseven": "five-seven",
    "weapon_glock": "glock-18",
    "weapon_tec9": "tec-9",
    "weapon_cz75a": "cz75-auto",
    "weapon_elite": "dual berettas",
    "weapon_deagle": "desert eagle",
    "weapon_revolver": "r8 revolver",

    # SMGs
    "weapon_mp9": "mp9",
    "weapon_mac10": "mac-10",
    "weapon_bizon": "pp-bizon",
    "weapon_mp7": "mp7",
    "weapon_ump45": "ump-45",
    "weapon_p90": "p90",
    "weapon_mp5sd": "mp5-sd",

    # Rifles
    "weapon_famas": "famas",
    "weapon_galilar": "galil ar",
    "weapon_m4a1": "m4a4",
    "weapon_m4a1_silencer": "m4a1-s",
    "weapon_ak47": "ak-47",
    "weapon_aug": "aug",
    "weapon_sg556": "sg 553",
    "weapon_ssg08": "ssg 08",
    "weapon_awp": "awp",
    "weapon_scar20": "scar-20",
    "weapon_g3sg1": "g3sg1",

    # Shotguns
    "weapon_nova": "nova",
    "weapon_xm1014": "xm1014",
    "weapon_mag7": "mag-7",
    "weapon_sawedoff": "sawed-off",

    # Machine guns
    "weapon_m249": "m249",
    "weapon_negev": "negev",

    # Knives (all variants)
    "weapon_knife": "knife",
    "weapon_knife_t": "knife",
    "weapon_bayonet": "knife",
    "weapon_knife_flip": "knife",
    "weapon_knife_gut": "knife",
    "weapon_knife_karambit": "knife",
    "weapon_knife_m9_bayonet": "knife",
    "weapon_knife_tactical": "knife",
    "weapon_knife_falchion": "knife",
    "weapon_knife_survival_bowie": "knife",
    "weapon_knife_butterfly": "knife",
    "weapon_knife_push": "knife",
    "weapon_knife_ursus": "knife",
    "weapon_knife_gypsy_jackknife": "knife",
    "weapon_knife_stiletto": "knife",
    "weapon_knife_widowmaker": "knife",
    "weapon_knife_css": "knife",
    "weapon_knife_cord": "knife",
    "weapon_knife_canis": "knife",
    "weapon_knife_outdoor": "knife",
    "weapon_knife_skeleton": "knife",
    "weapon_knife_kukri": "knife",

    # Grenades
    "weapon_hegrenade": "high explosive grenade",
    "weapon_flashbang": "flashbang",
    "weapon_smokegrenade": "smoke grenade",
    "weapon_decoy": "decoy grenade",
    "weapon_molotov": "molotov",
    "weapon_incgrenade": "incendiary grenade",

    # Other
    "weapon_c4": "c4 explosive",
    "weapon_taser": "None",  # Zeus not in list
}


def get_cs2_cfg_path() -> Path:
    """Get CS2 cfg directory path."""
    # Try common Steam locations
    paths = [
        Path(r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg"),
        Path(r"C:\Program Files\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg"),
        Path.home() / ".steam/steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/cfg",
    ]

    for p in paths:
        if p.exists():
            return p

    return paths[0]  # Return default even if not found
