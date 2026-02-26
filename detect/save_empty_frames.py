"""
Save all empty frames resized to 640x640 into a separate folder.

Usage:
  python detect/save_empty_frames.py \
    --empty-frames empty_frames.json \
    --output D:\empty_frames_640
"""

import json
import argparse
import cv2
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description='Save empty frames as 640x640')
    p.add_argument('--empty-frames', required=True, help='Path to empty_frames.json')
    p.add_argument('--output', required=True, help='Output directory')
    p.add_argument('--size', type=int, default=640, help='Output image size')
    args = p.parse_args()

    with open(args.empty_frames, 'r') as f:
        empty_frames = json.load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = sum(len(v) for v in empty_frames.values())
    print(f"Total empty frames: {total}")

    saved = 0
    for demo_name, paths in empty_frames.items():
        for frame_path in paths:
            fp = Path(frame_path)
            if not fp.exists():
                continue

            img = cv2.imread(str(fp))
            if img is None:
                continue

            img = cv2.resize(img, (args.size, args.size))

            # Name: demoname_tick_XXXX.jpg
            out_name = f"{demo_name}_{fp.stem}.jpg"
            cv2.imwrite(str(output_dir / out_name), img)
            saved += 1

            if saved % 500 == 0:
                print(f"  {saved}/{total} saved...")

    print(f"\nDone: {saved} frames saved to {output_dir}")


if __name__ == '__main__':
    main()
