"""
Extract realistic player screen-space positions from CS2 demo files.

Parses .dem files, projects enemy positions to screen-space using 3D->2D
projection, and saves normalized position distributions for use in
synthetic YOLO dataset generation.

Output:
  - positions.npz  — (N, 4) array [cx_norm, cy_norm, h_norm, w_norm]
  - empty_frames.json — empty ticks matched to real frame files

Usage:
  python detect/extract_demo_positions.py \
    --demo-dir "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo" \
    --frames-dir "D:\FramesDataset" \
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
                        target_pos, fov, width, height, size_mult=1.0):
    """Project a player's bounding box to screen-space.

    Projects 4 key points (head, feet, left, right) and returns
    the screen-space bounding box.

    Args:
        size_mult: multiplier for player dimensions (>1.0 = larger detection zone,
                   useful for conservative empty-frame filtering)

    Returns:
        (cx, cy, w, h) in pixels, or None if not visible.
    """
    delta = target_pos - observer_pos

    ph = PLAYER_HEIGHT * size_mult
    pw = PLAYER_WIDTH * size_mult

    # 4 key points in world space
    pos_head  = delta + np.array([0, 0, ph])
    pos_feet  = delta.copy()
    pos_left  = delta + np.array([0, -pw / 2, ph / 2])
    pos_right = delta + np.array([0,  pw / 2, ph / 2])

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

def _get_live_tick_ranges(parser):
    """Get tick ranges where actual gameplay is happening.

    Uses round_freeze_end and round_end events to find intervals
    between freeze end (buy phase over) and round end.
    Excludes: warmup, freeze time, victory screens, game over.

    Returns:
        list of (start_tick, end_tick) tuples
    """
    events = parser.parse_events(['round_freeze_end', 'round_end'])
    freeze_ends = []
    round_ends = []
    for name, df in events:
        if name == 'round_freeze_end':
            freeze_ends = sorted(df['tick'].tolist())
        elif name == 'round_end':
            round_ends = sorted(df['tick'].tolist())

    # Build ranges: [freeze_end_tick, round_end_tick]
    ranges = []
    for fe_tick in freeze_ends:
        # Find the next round_end after this freeze_end
        matching_end = None
        for re_tick in round_ends:
            if re_tick > fe_tick:
                matching_end = re_tick
                break
        if matching_end is not None:
            ranges.append((fe_tick, matching_end))

    return ranges


def _tick_in_live_ranges(tick, live_ranges):
    """Check if tick is within any live gameplay range."""
    for start, end in live_ranges:
        if start <= tick <= end:
            return True
    return False


def parse_demo(demo_path, steam_id, fov, width, height,
               available_ticks=None, tick_step=16):
    """Parse a single demo file and extract screen-space positions.

    Args:
        demo_path: path to .dem file
        steam_id: POV player's Steam ID
        fov: horizontal FOV in degrees
        width: screen width in pixels
        height: screen height in pixels
        available_ticks: set of ticks that have frame files (if None, sample uniformly)
        tick_step: sample every N ticks for positions (default 16)

    Returns:
        positions: list of (cx_norm, cy_norm, h_norm, w_norm) — enemy positions
        empty_ticks: list of ticks with NO visible players at all (enemies + teammates)
    """
    parser = DemoParser(str(demo_path))

    # Get live gameplay ranges (between freeze_end and round_end)
    live_ranges = _get_live_tick_ranges(parser)
    print(f'  {len(live_ranges)} live round ranges')

    df = parser.parse_ticks([
        'tick', 'steamid', 'X', 'Y', 'Z',
        'yaw', 'pitch', 'is_alive', 'team_num',
    ])

    if df.empty:
        print(f'  [WARN] Empty data for {demo_path}')
        return [], []

    unique_ticks = sorted(df['tick'].unique())

    # For positions: sample every tick_step ticks (from all ticks in live ranges)
    sampled_ticks_for_pos = set(
        t for t in unique_ticks[::tick_step]
        if _tick_in_live_ranges(t, live_ranges)
    )

    # For empty frames: only check frame ticks within live ranges
    if available_ticks is not None:
        ticks_for_empty = set(
            t for t in available_ticks
            if t in set(unique_ticks) and _tick_in_live_ranges(t, live_ranges)
        )
    else:
        ticks_for_empty = sampled_ticks_for_pos

    # Union: check both position-sampled ticks and frame ticks
    all_ticks_to_check = sorted(sampled_ticks_for_pos | ticks_for_empty)

    positions = []
    empty_ticks = []

    for tick in all_ticks_to_check:
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

        # --- For POSITIONS: project enemies only ---
        enemies = tick_data[
            (tick_data['steamid'] != steam_id) &
            (tick_data['is_alive'] == True) &
            (tick_data['team_num'] != obs_team)
        ]

        for _, enemy in enemies.iterrows():
            target_pos = np.array([enemy['X'], enemy['Y'], enemy['Z']],
                                  dtype=np.float64)
            result = project_player_bbox(
                obs_pos, obs_yaw, obs_pitch,
                target_pos, fov, width, height,
            )
            if result is not None and tick in sampled_ticks_for_pos:
                cx, cy, bw, bh = result
                positions.append([
                    cx / width, cy / height,
                    bh / height, bw / width,
                ])

        # --- For EMPTY FRAMES: project ALL other players (enemies + teammates) ---
        # Use size_mult=2.0 for conservative check (accounts for weapons, arms, poses)
        if tick in ticks_for_empty:
            all_others = tick_data[
                (tick_data['steamid'] != steam_id) &
                (tick_data['is_alive'] == True)
            ]

            anyone_visible = False
            for _, other in all_others.iterrows():
                target_pos = np.array([other['X'], other['Y'], other['Z']],
                                      dtype=np.float64)
                result = project_player_bbox(
                    obs_pos, obs_yaw, obs_pitch,
                    target_pos, fov, width, height,
                    size_mult=2.0,
                )
                if result is not None:
                    anyone_visible = True
                    break

            if not anyone_visible:
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
    p.add_argument('--frames-dir', default=None,
                   help='Directory with frame folders (D:\\FramesDataset). '
                        'If set, only parses demos with matching frame folders '
                        'and matches empty ticks to existing frames.')
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
                   help='Sample every N ticks for positions')

    args = p.parse_args()

    # Discover demos
    demo_dir = Path(args.demo_dir)
    frames_dir = Path(args.frames_dir) if args.frames_dir else None

    if demo_dir.is_file():
        dem_files = [demo_dir]
    else:
        dem_files = sorted(demo_dir.glob('*.dem'))

    if not dem_files:
        print(f'No .dem files found in {demo_dir}')
        return

    # Filter: only demos with frame folders
    if frames_dir is not None:
        frame_folders = {f.name for f in frames_dir.iterdir() if f.is_dir()}
        filtered = []
        for dem in dem_files:
            folder_name = dem.stem  # e.g. "1-03f67162-...-1-1"
            if folder_name in frame_folders:
                filtered.append(dem)
        print(f'Found {len(dem_files)} demo(s), {len(filtered)} have frame folders')
        dem_files = filtered
    else:
        print(f'Found {len(dem_files)} demo(s)')

    if not dem_files:
        print('No demos with matching frames!')
        return

    print(f'Settings: FOV={args.fov}, resolution={args.width}x{args.height}, '
          f'tick_step={args.tick_step}')
    print()

    all_positions = []
    all_empty_frames = {}  # demo_folder -> list of frame paths

    for i, dem in enumerate(dem_files):
        folder_name = dem.stem
        print(f'[{i+1}/{len(dem_files)}] Parsing {dem.name}...')

        # Get available frame ticks for this demo
        available_ticks = None
        if frames_dir is not None:
            frames_folder = frames_dir / folder_name
            if frames_folder.exists():
                # Parse tick numbers from filenames: tick_XXXX.jpg
                available_ticks = set()
                for fp in frames_folder.glob('tick_*.jpg'):
                    try:
                        tick_num = int(fp.stem.split('_')[1])
                        available_ticks.add(tick_num)
                    except (ValueError, IndexError):
                        pass
                print(f'  {len(available_ticks)} frame files found')

        try:
            positions, empty_ticks = parse_demo(
                dem, args.steam_id, args.fov,
                args.width, args.height,
                available_ticks=available_ticks,
                tick_step=args.tick_step,
            )
            all_positions.extend(positions)

            # Save empty ticks as frame paths
            empty_frame_paths = []
            for tick in empty_ticks:
                if frames_dir is not None:
                    frame_path = str(frames_dir / folder_name / f'tick_{tick}.jpg')
                else:
                    frame_path = f'{folder_name}/tick_{tick}.jpg'
                empty_frame_paths.append(frame_path)
            all_empty_frames[folder_name] = empty_frame_paths

            print(f'  -> {len(positions)} positions, {len(empty_ticks)} empty frames')
        except Exception as e:
            print(f'  [ERROR] {e}')

    # Save positions
    output_path = Path(args.output)
    if all_positions:
        positions_arr = np.array(all_positions, dtype=np.float32)
        np.savez_compressed(str(output_path), positions=positions_arr)
        print(f'\nSaved {len(positions_arr)} positions to {output_path}')
    else:
        positions_arr = np.array([], dtype=np.float32)
        print('\nNo positions extracted!')

    # Save empty frames
    empty_path = output_path.with_name('empty_frames.json')
    with open(empty_path, 'w') as f:
        json.dump(all_empty_frames, f, indent=2)

    total_empty = sum(len(v) for v in all_empty_frames.values())
    print(f'Saved {total_empty} empty frame paths to {empty_path}')

    # Statistics
    if len(positions_arr) > 0 and positions_arr.ndim == 2:
        print(f'\n{"="*50}')
        print(f'Position distribution statistics:')
        print(f'  Total positions: {len(positions_arr)}')
        labels = ['cx', 'cy', 'h', 'w']
        for i, name in enumerate(labels):
            col = positions_arr[:, i]
            print(f'  {name:>2s}: min={col.min():.3f}  max={col.max():.3f}  '
                  f'mean={col.mean():.3f}  std={col.std():.3f}')
        print(f'  Empty frames: {total_empty}')


if __name__ == '__main__':
    main()
