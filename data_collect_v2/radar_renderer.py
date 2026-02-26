"""
Radar Renderer — generates minimap images from demo player positions.

Port of legacy pos_matching.py into a clean OOP interface.
Renders: player dots, direction triangles, FOV cone, bomb icons, death X marks.

Usage:
    renderer = RadarRenderer("de_mirage", target_steam_id=76561198..., assets_dir="assets")
    renderer.setup_round(positions_df, round_start_tick)
    for tick in range(start, end):
        img = renderer.render_tick(tick)  # BGR numpy array
"""

import os
import math

import cv2
import numpy as np
import pandas as pd


# ── Map coordinate data ──────────────────────────────────────────────────── #

MAP_DATA = {
    "de_mirage": {
        "radar_image": "de_mirage_radar.jpg",
        "crop": (8, 425, 10, 574),   # y1, y2, x1, x2
        "left_bot": (-3200, -3300),
        "right_up": (1850, 1700),
    },
    # Add more maps here:
    # "de_dust2": { ... },
    # "de_inferno": { ... },
}

# 5 teammate colors (BGR) matching CS2 spectator colors
TEAM_COLORS = [
    (211, 0, 148),   # purple
    (0, 255, 255),    # yellow
    (0, 100, 220),    # orange
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
]
ENEMY_COLOR = (0, 0, 255)  # red

# Color → bomb icon filename mapping
COLOR_TO_BOMB = {
    (211, 0, 148): "orange_to_violet.png",
    (0, 255, 255): "orange_to_yellow.png",
    (0, 100, 220): "bomb_dropped_crop.jpg",
    (0, 255, 0):   "orange_to_green.png",
    (255, 0, 0):   "orange_to_cyan.png",
    (0, 0, 255):   "orange_to_red.png",
}

# Rendering parameters
MARKER_SIZE = 15       # direction triangle tip distance
MARKER_BASE = 20       # direction triangle base width
FOV_ANGLE = 70         # FOV cone angle (degrees)
FOV_LENGTH = 40        # FOV cone max distance (pixels)
FOV_ALPHA_MAX = 1.0    # FOV cone max opacity
CIRCLE_RADIUS = 10     # player dot radius
DEATH_X_SIZE = 8       # death cross half-size
DEATH_X_THICK = 5      # death cross line thickness
DEATH_FADE_TICKS = 10  # ticks to show death marker


# ── Helper ───────────────────────────────────────────────────────────────── #

def _angle_in_cone(angle, left, right):
    """Check if angle falls within [left, right] arc (handles wrap-around)."""
    a = (angle - left) % (2 * math.pi)
    b = (right - left) % (2 * math.pi)
    return a <= b


# ── Renderer ─────────────────────────────────────────────────────────────── #

class RadarRenderer:
    """Renders minimap radar from player position data."""

    def __init__(self, map_name: str, target_steam_id: int, assets_dir: str):
        if map_name not in MAP_DATA:
            raise ValueError(f"Unknown map: {map_name}. Supported: {list(MAP_DATA.keys())}")

        self._map = MAP_DATA[map_name]
        self._target_steam = target_steam_id
        self._assets_dir = assets_dir

        # Load radar background
        radar_path = os.path.join(assets_dir, self._map["radar_image"])
        raw = cv2.imread(radar_path)
        if raw is None:
            raise FileNotFoundError(f"Radar template not found: {radar_path}")
        y1, y2, x1, x2 = self._map["crop"]
        self._base_img = raw[y1:y2, x1:x2]

        # Coordinate transform coefficients
        lb = self._map["left_bot"]
        ru = self._map["right_up"]
        self._left_bot = lb
        h, w = self._base_img.shape[:2]
        self._coefX = (abs(ru[0]) + abs(lb[0])) / w
        self._coefY = (abs(ru[1]) + abs(lb[1])) / h

        # Load bomb icons
        self._bomb_icons = {}
        for color, filename in COLOR_TO_BOMB.items():
            path = os.path.join(assets_dir, filename)
            icon = cv2.imread(path)
            if icon is not None:
                self._bomb_icons[color] = icon
        # White bomb (for teammate dropped/planted)
        white_path = os.path.join(assets_dir, "orange_to_white.png")
        self._bomb_white = cv2.imread(white_path)

        # Per-round state (set by setup_round)
        self._df: pd.DataFrame = pd.DataFrame()
        self._teammates: list = []
        self._teammates_steamid: list = []
        self._dict_colors: dict = {}
        self._cd_dead: dict = {}

    def _world_to_pixel(self, wx: float, wy: float) -> tuple:
        """Convert world coordinates to radar pixel coordinates."""
        h = self._base_img.shape[0]
        px = round(wx / self._coefX - self._left_bot[0] / self._coefX)
        py = h - round(wy / self._coefY - self._left_bot[1] / self._coefY)
        return px, py

    def setup_round(self, positions_df: pd.DataFrame, round_start_tick: int):
        """
        Initialize renderer state for a new round.

        Determines teams, assigns colors, resets death cooldowns.
        Must be called before render_tick() for each round.
        """
        self._df = positions_df

        # Find team info at round start
        tmp = positions_df[positions_df['tick'] == round_start_tick]
        if tmp.empty:
            # Fallback: find nearest tick
            ticks = positions_df['tick'].unique()
            nearest = ticks[np.argmin(np.abs(ticks - round_start_tick))]
            tmp = positions_df[positions_df['tick'] == nearest]

        target_row = tmp[tmp['steamid'] == self._target_steam]
        if target_row.empty:
            print(f"[RadarRenderer] WARNING: target steam {self._target_steam} "
                  f"not found at tick {round_start_tick}")
            self._teammates = []
            self._teammates_steamid = []
            self._dict_colors = {-1: ENEMY_COLOR}
            self._cd_dead = {}
            return

        target_team = target_row['team_num'].iloc[0]
        team_df = tmp[tmp['team_num'] == target_team]
        self._teammates = list(team_df['user_id'])
        self._teammates_steamid = list(map(int, team_df['steamid']))

        # Assign colors to teammates
        self._dict_colors = {}
        for i, uid in enumerate(self._teammates):
            self._dict_colors[uid] = TEAM_COLORS[i % len(TEAM_COLORS)]
        self._dict_colors[-1] = ENEMY_COLOR

        # Reset death cooldowns
        self._cd_dead = {}
        for uid in tmp['user_id']:
            self._cd_dead[uid] = [DEATH_FADE_TICKS, True]

    def render_tick(self, tick: int) -> np.ndarray:
        """
        Render radar for a single tick.

        Returns BGR image (same size as cropped radar template).
        """
        img = self._base_img.copy()
        tick_df = self._df[self._df['tick'] == tick].sort_values('user_id')

        if tick_df.empty:
            return img

        # Put target player last (drawn on top)
        row_target = tick_df[tick_df['steamid'] == self._target_steam]
        tick_df = tick_df[tick_df['steamid'] != self._target_steam]
        tick_df = pd.concat([tick_df, row_target]).reset_index(drop=True)

        for i in range(len(tick_df)):
            row = tick_df.iloc[i]
            x, y = self._world_to_pixel(row['X'], row['Y'])
            raw_uid = row['user_id']
            steamid = row['steamid']
            uid = raw_uid if raw_uid in self._teammates else -1

            is_alive = bool(row['is_alive'])
            is_spotted = bool(row['spotted']) if not pd.isna(row['spotted']) else False
            is_teammate = uid != -1

            # Visibility: teammates always visible, enemies only when spotted
            visible = (is_alive and is_teammate) or (is_spotted and not is_teammate and is_alive)

            if visible:
                self._cd_dead[raw_uid] = [DEATH_FADE_TICKS, True]
                color = self._dict_colors[uid]

                # Draw bomb icon if player carries bomb
                drew_bomb = self._try_draw_bomb_carrier(img, row, uid, steamid, x, y, color)

                if not drew_bomb:
                    # Draw player dot
                    cv2.circle(img, (x, y), CIRCLE_RADIUS, color=color, thickness=-1)

                # Draw direction triangle for teammates
                if is_teammate:
                    self._draw_direction_triangle(img, row, x, y)

                    # Draw FOV cone for target player
                    if steamid == self._target_steam:
                        self._draw_fov_cone(img, row, x, y)

            # Death X marker with fade
            if not is_alive and raw_uid in self._cd_dead:
                cd = self._cd_dead[raw_uid]
                if cd[0] > 0 and cd[1]:
                    cd[0] -= 1
                    color = self._dict_colors[uid]
                    cv2.line(img, (x - DEATH_X_SIZE, y - DEATH_X_SIZE),
                             (x + DEATH_X_SIZE, y + DEATH_X_SIZE), color, DEATH_X_THICK)
                    cv2.line(img, (x - DEATH_X_SIZE, y + DEATH_X_SIZE),
                             (x + DEATH_X_SIZE, y - DEATH_X_SIZE), color, DEATH_X_THICK)
                elif cd[0] == 0:
                    cd[1] = False

            # Dropped/planted bomb (teammate perspective)
            self._try_draw_dropped_bomb(img, row)

        return img

    def _try_draw_bomb_carrier(self, img, row, uid, steamid, x, y, color):
        """Draw bomb icon on carrier. Returns True if bomb was drawn instead of dot."""
        owner = row['owner']
        bomb_dropped = bool(row['bomb_dropped']) if not pd.isna(row.get('bomb_dropped', pd.NA)) else False
        bomb_planted = bool(row['bomb_planted']) if not pd.isna(row.get('bomb_planted', pd.NA)) else False

        if pd.isna(owner):
            return False

        # Player doesn't carry bomb (it's dropped or planted, or someone else has it)
        if steamid != owner or bomb_dropped or bomb_planted:
            return False

        # This player carries the bomb
        # Check if it's a teammate with bomb → show bomb icon
        if (owner in self._teammates_steamid or
            (bool(row['spotted']) if not pd.isna(row.get('spotted', pd.NA)) else False
             and owner not in self._teammates_steamid)):
            if not bomb_planted:
                bomb_icon = self._bomb_icons.get(color)
                if bomb_icon is not None:
                    bx, by = self._world_to_pixel(row['bombX'], row['bombY'])
                    self._paste_icon(img, bomb_icon, bx, by)
                    return True

        return False

    def _try_draw_dropped_bomb(self, img, row):
        """Draw white bomb icon when bomb is dropped or planted (from teammate info)."""
        owner = row['owner']
        bomb_dropped = bool(row['bomb_dropped']) if not pd.isna(row.get('bomb_dropped', pd.NA)) else False
        bomb_planted = bool(row['bomb_planted']) if not pd.isna(row.get('bomb_planted', pd.NA)) else False

        if pd.isna(owner):
            return

        if owner not in [int(s) for s in self._teammates_steamid]:
            return

        if not (bomb_dropped or bomb_planted):
            return

        if self._bomb_white is None:
            return

        bx, by = self._world_to_pixel(row['bombX'], row['bombY'])
        self._paste_icon(img, self._bomb_white, bx, by)

    def _paste_icon(self, img, icon, cx, cy):
        """Paste a small icon centered at (cx, cy) with bounds checking."""
        ih, iw = icon.shape[:2]
        y1 = cy - 8
        x1 = cx - 12
        y2 = y1 + ih
        x2 = x1 + iw

        # Bounds check
        h, w = img.shape[:2]
        if y1 < 0 or x1 < 0 or y2 > h or x2 > w:
            return

        img[y1:y2, x1:x2] = icon

    def _draw_direction_triangle(self, img, row, x, y):
        """Draw direction-pointing triangle for a player."""
        yaw_rad = math.pi * 2 - math.radians(row['yaw'])

        tip_x = x + math.cos(yaw_rad) * MARKER_SIZE
        tip_y = y + math.sin(yaw_rad) * MARKER_SIZE

        angle_offset = math.radians(30)
        left_x = x + math.cos(yaw_rad + angle_offset) * (MARKER_BASE / 2)
        left_y = y + math.sin(yaw_rad + angle_offset) * (MARKER_BASE / 2)
        right_x = x + math.cos(yaw_rad - angle_offset) * (MARKER_BASE / 2)
        right_y = y + math.sin(yaw_rad - angle_offset) * (MARKER_BASE / 2)

        triangle = np.array([
            [int(tip_x), int(tip_y)],
            [int(left_x), int(left_y)],
            [int(right_x), int(right_y)],
        ])
        cv2.fillPoly(img, [triangle], color=(230, 230, 230))

    def _draw_fov_cone(self, img, row, x, y):
        """Draw FOV visibility cone for target player."""
        yaw_rad = math.pi * 2 - math.radians(row['yaw'])
        half_fov = math.radians(FOV_ANGLE / 2)
        left_bound = yaw_rad - half_fov
        right_bound = yaw_rad + half_fov

        h, w = img.shape[:2]
        Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        dx = X - x
        dy = Y - y
        distances = np.sqrt(dx ** 2 + dy ** 2)
        angles = np.arctan2(dy, dx)

        mask = (distances <= FOV_LENGTH) & _angle_in_cone(angles, left_bound, right_bound)
        alpha = np.zeros_like(distances, dtype=np.float32)
        alpha[mask] = FOV_ALPHA_MAX * (1 - (distances[mask] / FOV_LENGTH))

        overlay = np.full_like(img, (230, 230, 230), dtype=np.uint8)

        for c in range(3):
            img[:, :, c] = (img[:, :, c] * (1 - alpha) + overlay[:, :, c] * alpha).astype(np.uint8)
