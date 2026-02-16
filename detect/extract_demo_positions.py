"""
Extract realistic player screen-space positions from CS2 demo files.

Parses .dem files, projects enemy positions to screen-space using 3D->2D
projection, and saves normalized position distributions for use in
synthetic YOLO dataset generation.

Output:
  - positions.npz  — (N, 4) array [cx_norm, cy_norm, h_norm, w_norm]
  - empty_ticks.json — ticks with no visible enemies (clean backgrounds)

Usage:
  python detect/extract_demo_positions.py \
    --demo-dir "D:\\demos" \
    --steam-id 76561198386265483 \
    --output positions.npz \
    --fov 90 --width 1920 --height 1080
"""

import argparse
import json
import numpy as np
from pathlib import Path
from demoparser2 import DemoParser


# CS2 player dimensions (Source 2 units)
PLAYER_HEIGHT = 72
PLAYER_WIDTH = 32


# ============================================================================
# Projection functions (from bbox_markup.py)
# ============================================================================

def world_to_local(vec, yaw, pitch):
    """Convert world-space vector to camera-local coordinates.

    CS2 convention:
      yaw  — rotation around Z (horizontal)
      pitch — rotation around Y (vertical, up/down)
    """
    yaw = np.radians(yaw)
    pitch = np.radians(-pitch)  # invert pitch

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    forward = np.array([cp * cy, cp * sy, sp])
    right   = np.array([-sy, cy, 0])
    up      = np.cross(right, forward)

    mat = np.vstack([right, up, forward]).T
    return mat.T @ vec


def project_to_screen(local, fov_deg, width, height):
    """Project camera-local 3D point to 2D screen coordinates."""
    fov = np.radians(fov_deg)
    aspect = width / height
    fx = 1 / np.tan(fov / 2)
    fy = fx * aspect

    x, y, z = local
    if z <= 0:
        return None

    ndc_x = (x / z) * fx
    ndc_y = (y / z) * fy

    screen_x = width  / 2 * (1 + ndc_x)
    screen_y = height / 2 * (1 + ndc_y)
    return np.array([screen_x, screen_y])


def project_player_bbox(observer_pos, observer_yaw, observer_pitch,
                        target_pos, fov, width, height):
    """Project a player's bounding box to screen-space.

    Projects 4 key points (head, feet, left, right) and returns
    the screen-space bounding box.

    Returns:
        (cx, cy, w, h) in pixels, or None if not visible.
    """
    delta = target_pos - observer_pos

    # 4 key points in world space
    pos_head  = delta + np.array([0, 0, PLAYER_HEIGHT])
    pos_feet  = delta.copy()
    pos_left  = delta + np.array([0, -PLAYER_WIDTH / 2, PLAYER_HEIGHT / 2])
    pos_right = delta + np.array([0,  PLAYER_WIDTH / 2, PLAYER_HEIGHT / 2])

    # Project all 4
    pts = []
    for pt in [pos_head, pos_feet, pos_left, pos_right]:
        local = world_to_local(pt, observer_yaw, observer_pitch)
        screen = project_to_screen(local, fov, width, height)
        if screen is None:
            return None
        pts.append(screen)

    head_2d, feet_2d, left_2d, right_2d = pts

    x_min = min(left_2d[0], right_2d[0])
    x_max = max(left_2d[0], right_2d[0])
    y_min = min(head_2d[1], feet_2d[1])  # head is usually higher (smaller y)
    y_max = max(head_2d[1], feet_2d[1])

    # Clamp to screen
    x_min = max(0, x_min)
    x_max = min(width, x_max)
    y_min = max(0, y_min)
    y_max = min(height, y_max)

    bw = x_max - x_min
    bh = y_max - y_min

    if bw < 4 or bh < 8:
        return None

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    return cx, cy, bw, bh


# ============================================================================
# Demo parsing
# ============================================================================

def parse_demo(demo_path, steam_id, fov, width, height, tick_step=16):
    """Parse a single demo file and extract screen-space positions.

    Args:
        demo_path: path to .dem file
        steam_id: POV player's Steam ID
        fov: horizontal FOV in degrees
        width: screen width in pixels
        height: screen height in pixels
        tick_step: sample every N ticks (default 16)

    Returns:
        positions: list of (cx_norm, cy_norm, h_norm, w_norm)
        empty_ticks: list of ticks with no visible enemies
    """
    parser = DemoParser(str(demo_path))

    df = parser.parse_ticks([
        'tick', 'steamid', 'X', 'Y', 'Z',
        'yaw', 'pitch', 'is_alive', 'team_num',
    ])

    if df.empty:
        print(f'  [WARN] Empty data for {demo_path}')
        return [], []

    # Sample every tick_step ticks
    unique_ticks = sorted(df['tick'].unique())
    sampled_ticks = unique_ticks[::tick_step]

    positions = []
    empty_ticks = []

    for tick in sampled_ticks:
        tick_data = df[df['tick'] == tick]

        # Find observer
        obs_rows = tick_data[tick_data['steamid'] == steam_id]
        if obs_rows.empty:
            continue

        obs = obs_rows.iloc[0]
        if not obs['is_alive']:
            continue

        obs_pos = np.array([obs['X'], obs['Y'], obs['Z']], dtype=np.float64)
        obs_yaw = float(obs['yaw'])
        obs_pitch = float(obs['pitch'])
        obs_team = obs['team_num']

        # Find enemies (alive, other team)
        enemies = tick_data[
            (tick_data['steamid'] != steam_id) &
            (tick_data['is_alive'] == True) &
            (tick_data['team_num'] != obs_team)
        ]

        tick_has_visible = False

        for _, enemy in enemies.iterrows():
            target_pos = np.array([enemy['X'], enemy['Y'], enemy['Z']],
                                  dtype=np.float64)

            result = project_player_bbox(
                obs_pos, obs_yaw, obs_pitch,
                target_pos, fov, width, height,
            )

            if result is not None:
                cx, cy, bw, bh = result
                # Normalize to [0, 1]
                positions.append([
                    cx / width,
                    cy / height,
                    bh / height,
                    bw / width,
                ])
                tick_has_visible = True

        if not tick_has_visible:
            empty_ticks.append(int(tick))

    return positions, empty_ticks


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description='Extract realistic player positions from CS2 demos')

    p.add_argument('--demo-dir', required=True,
                   help='Directory with .dem files (or single .dem file)')
    p.add_argument('--steam-id', type=int, required=True,
                   help='Steam ID of the POV player')
    p.add_argument('--output', default='positions.npz',
                   help='Output .npz file path')
    p.add_argument('--fov', type=float, default=90.0,
                   help='Horizontal FOV in degrees')
    p.add_argument('--width', type=int, default=1920,
                   help='Screen width')
    p.add_argument('--height', type=int, default=1080,
                   help='Screen height')
    p.add_argument('--tick-step', type=int, default=16,
                   help='Sample every N ticks')

    args = p.parse_args()

    demo_path = Path(args.demo_dir)
    if demo_path.is_file():
        dem_files = [demo_path]
    else:
        dem_files = sorted(demo_path.glob('*.dem'))

    if not dem_files:
        print(f'No .dem files found in {demo_path}')
        return

    print(f'Found {len(dem_files)} demo(s)')
    print(f'Settings: FOV={args.fov}, resolution={args.width}x{args.height}, '
          f'tick_step={args.tick_step}')
    print()

    all_positions = []
    all_empty_ticks = {}

    for i, dem in enumerate(dem_files):
        print(f'[{i+1}/{len(dem_files)}] Parsing {dem.name}...')
        try:
            positions, empty_ticks = parse_demo(
                dem, args.steam_id, args.fov,
                args.width, args.height, args.tick_step,
            )
            all_positions.extend(positions)
            all_empty_ticks[dem.name] = empty_ticks
            print(f'  -> {len(positions)} positions, {len(empty_ticks)} empty ticks')
        except Exception as e:
            print(f'  [ERROR] {e}')

    if not all_positions:
        print('\nNo positions extracted!')
        return

    positions_arr = np.array(all_positions, dtype=np.float32)

    # Save positions
    output_path = Path(args.output)
    np.savez_compressed(str(output_path), positions=positions_arr)
    print(f'\nSaved {len(positions_arr)} positions to {output_path}')

    # Save empty ticks
    empty_path = output_path.with_name('empty_ticks.json')
    with open(empty_path, 'w') as f:
        json.dump(all_empty_ticks, f)
    print(f'Saved empty ticks to {empty_path}')

    # Statistics
    print(f'\n{"="*50}')
    print(f'Position distribution statistics:')
    print(f'  Total positions: {len(positions_arr)}')
    labels = ['cx', 'cy', 'h', 'w']
    for i, name in enumerate(labels):
        col = positions_arr[:, i]
        print(f'  {name:>2s}: min={col.min():.3f}  max={col.max():.3f}  '
              f'mean={col.mean():.3f}  std={col.std():.3f}')

    total_empty = sum(len(v) for v in all_empty_ticks.values())
    print(f'  Empty ticks: {total_empty}')


if __name__ == '__main__':
    main()
