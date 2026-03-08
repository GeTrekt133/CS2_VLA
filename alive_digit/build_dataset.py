"""
Build alive-digit dataset from CS2 demo files + frame screenshots.

Pipeline:
1. Parse each .dem file → extract ct_alive, t_alive per tick
2. Match ticks to frame files (tick_XXXXX.jpg)
3. Stability filter: alive count must be same ±1 sec (64 ticks) around frame tick
4. Crop 25x25 digit regions from each frame (CT left, T right)
5. Save crops as grayscale PNGs with labels in CSV

Usage:
    python build_dataset.py --demos D:/DemosCS --frames D:/FramesDataset --out D:/alive_digit_dataset
"""

import argparse
import csv
import os
import random
import re
from pathlib import Path

import cv2
import numpy as np
from demoparser2 import DemoParser
from tqdm import tqdm

# Crop coordinates in 640x640 frame (center_x, center_y, size)
# CT alive digit (left of center scoreboard)
CT_CROP_CENTER = (288, 15)
# T alive digit (right of center scoreboard)
T_CROP_CENTER = (350, 15)
CROP_SIZE = 25

TICKRATE = 64
STABILITY_WINDOW = TICKRATE  # 1 second = 64 ticks
X_JITTER = 5  # random shift ±5px on X axis
N_JITTER = 3  # number of extra jittered crops per frame
FRAME_STRIDE = 16  # take every Nth frame (reduces dataset ~16x)


def crop_digit(frame: np.ndarray, center: tuple, size: int = CROP_SIZE) -> np.ndarray:
    """Crop a square region centered at (cx, cy) from frame, return grayscale."""
    cx, cy = center
    half = size // 2
    h, w = frame.shape[:2]
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)
    crop = frame[y1:y2, x1:x2]
    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Ensure exact size (pad if needed at edges)
    if crop.shape[0] != size or crop.shape[1] != size:
        padded = np.zeros((size, size), dtype=np.uint8)
        padded[:crop.shape[0], :crop.shape[1]] = crop
        crop = padded
    return crop


def parse_demo_alive(dem_path: str) -> tuple:
    """
    Parse .dem file, return:
        alive_map: {tick: (ct_alive, t_alive)}
        live_ticks: set of ticks that are in live phase (not freeze, not round over)
    """
    parser = DemoParser(dem_path)
    ticks_df = parser.parse_ticks(['is_alive', 'team_num'])

    alive_per_tick = {}
    for tick, group in ticks_df.groupby('tick'):
        alive = group[group['is_alive'] == True]
        ct = len(alive[alive['team_num'] == 3])
        t = len(alive[alive['team_num'] == 2])
        alive_per_tick[int(tick)] = (ct, t)

    # Get round phase boundaries
    round_prestart = sorted(parser.parse_event('round_prestart')['tick'].tolist())
    try:
        round_freeze_end = sorted(parser.parse_event('round_freeze_end')['tick'].tolist())
    except Exception:
        round_freeze_end = []
    try:
        round_end_ticks = sorted(parser.parse_event('round_end')['tick'].tolist())
    except Exception:
        round_end_ticks = []

    # Build set of "live" ticks (after freeze ends, before round ends)
    live_ticks = set()
    for i, freeze_end in enumerate(round_freeze_end):
        # Find the next round_end after this freeze_end
        round_end = None
        for ret in round_end_ticks:
            if ret > freeze_end:
                round_end = ret
                break
        if round_end is None:
            round_end = max(alive_per_tick.keys()) if alive_per_tick else freeze_end

        for tick in range(freeze_end, round_end):
            live_ticks.add(tick)

    return alive_per_tick, live_ticks


def is_stable(alive_map: dict, tick: int, side_idx: int, window: int = STABILITY_WINDOW) -> bool:
    """Check if alive count is the same at tick-window and tick+window."""
    current = alive_map.get(tick)
    if current is None:
        return False
    val = current[side_idx]
    # Must be 1-5
    if val < 1 or val > 5:
        return False
    before = alive_map.get(tick - window)
    after = alive_map.get(tick + window)
    if before is None or after is None:
        return False
    return before[side_idx] == val and after[side_idx] == val


def get_frame_ticks(frames_dir: str) -> list:
    """Get sorted list of (tick, filepath) from tick_XXXXX.jpg files."""
    pattern = re.compile(r'tick_(\d+)\.jpg')
    results = []
    for fname in os.listdir(frames_dir):
        m = pattern.match(fname)
        if m:
            tick = int(m.group(1))
            results.append((tick, os.path.join(frames_dir, fname)))
    results.sort(key=lambda x: x[0])
    return results


def build_dataset(demos_dir: str, frames_dir: str, out_dir: str):
    """Build the full dataset."""
    os.makedirs(out_dir, exist_ok=True)
    crops_dir = os.path.join(out_dir, 'crops')
    os.makedirs(crops_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, 'labels.csv')
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['filename', 'label', 'side', 'demo', 'tick'])

    # Find matching demo/frame pairs
    frame_folders = [d for d in os.listdir(frames_dir)
                     if os.path.isdir(os.path.join(frames_dir, d))]

    total_crops = 0
    label_counts = {i: 0 for i in range(1, 6)}

    for folder_name in sorted(frame_folders):
        dem_path = os.path.join(demos_dir, folder_name + '.dem')
        folder_path = os.path.join(frames_dir, folder_name)

        if not os.path.exists(dem_path):
            print(f"[SKIP] No .dem for {folder_name}")
            continue

        print(f"\n[PARSE] {folder_name}")
        print(f"  Demo: {dem_path}")

        # Parse demo
        try:
            alive_map, live_ticks = parse_demo_alive(dem_path)
        except Exception as e:
            print(f"  [ERROR] Failed to parse demo: {e}")
            continue

        print(f"  Ticks with alive data: {len(alive_map)}, live ticks: {len(live_ticks)}")

        # Get frames
        frame_ticks = get_frame_ticks(folder_path)
        print(f"  Frames: {len(frame_ticks)}")

        demo_crops = 0
        for i, (tick, frame_path) in enumerate(tqdm(frame_ticks, desc=f"  Cropping")):
            if i % FRAME_STRIDE != 0:
                continue
            # Check if tick exists in demo data AND is in live phase
            if tick not in alive_map or tick not in live_ticks:
                continue

            ct_alive, t_alive = alive_map[tick]

            # Process both sides
            for side, center, alive_val in [
                ('ct', CT_CROP_CENTER, ct_alive),
                ('t', T_CROP_CENTER, t_alive),
            ]:
                # Skip invalid values
                if alive_val < 1 or alive_val > 5:
                    continue

                # Stability check: same value ±1 sec
                side_idx = 0 if side == 'ct' else 1
                if not is_stable(alive_map, tick, side_idx):
                    continue

                # Load frame and crop
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue

                # Original crop (center)
                crop = crop_digit(frame, center)
                fname = f"{folder_name}_{tick}_{side}.png"
                cv2.imwrite(os.path.join(crops_dir, fname), crop)
                writer.writerow([fname, alive_val, side, folder_name, tick])
                label_counts[alive_val] = label_counts.get(alive_val, 0) + 1
                demo_crops += 1

                # Jittered crops (random X shift ±5px)
                cx, cy = center
                for j in range(N_JITTER):
                    dx = random.randint(-X_JITTER, X_JITTER)
                    jittered_center = (cx + dx, cy)
                    crop_j = crop_digit(frame, jittered_center)
                    fname_j = f"{folder_name}_{tick}_{side}_j{j}_{dx:+d}.png"
                    cv2.imwrite(os.path.join(crops_dir, fname_j), crop_j)
                    writer.writerow([fname_j, alive_val, side, folder_name, tick])
                    label_counts[alive_val] = label_counts.get(alive_val, 0) + 1
                    demo_crops += 1

        total_crops += demo_crops
        print(f"  Crops saved: {demo_crops}")

    csv_file.close()

    print(f"\n{'='*50}")
    print(f"Total crops: {total_crops}")
    print(f"Label distribution: {label_counts}")
    print(f"Dataset saved to: {out_dir}")
    print(f"Labels CSV: {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build alive digit dataset')
    parser.add_argument('--demos', default='D:/DemosCS', help='Path to .dem files')
    parser.add_argument('--frames', default='D:/FramesDataset', help='Path to frame folders')
    parser.add_argument('--out', default='D:/alive_digit_dataset', help='Output directory')
    args = parser.parse_args()

    build_dataset(args.demos, args.frames, args.out)
