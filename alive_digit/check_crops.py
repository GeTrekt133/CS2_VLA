"""
Visual check of crop coordinates.

Draws crop rectangles on a frame and saves the result + individual crops.
Use this to verify/adjust CT_CROP_CENTER and T_CROP_CENTER before building dataset.

Usage:
    python check_crops.py D:/FramesDataset/1-0b0a3f84-0078-4a21-9a18-8671c37009e0-1-1/tick_10000.jpg
    python check_crops.py D:/FramesDataset/1-0b0a3f84-0078-4a21-9a18-8671c37009e0-1-1/tick_100000.jpg
"""

import sys
import cv2
import numpy as np

CT_CROP_CENTER = (288, 15)
T_CROP_CENTER = (350, 15)
CROP_SIZE = 25


def check(frame_path: str):
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Cannot read: {frame_path}")
        return

    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")

    vis = frame.copy()
    half = CROP_SIZE // 2

    for name, (cx, cy) in [('CT', CT_CROP_CENTER), ('T', T_CROP_CENTER)]:
        x1, y1 = cx - half, cy - half
        x2, y2 = x1 + CROP_SIZE, y1 + CROP_SIZE
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(vis, name, (x1, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Save individual crops (upscaled 8x for visibility)
        crop_big = cv2.resize(crop, (CROP_SIZE * 8, CROP_SIZE * 8), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f'check_crop_{name}.png', crop_big)
        print(f"  {name} crop: ({x1},{y1})-({x2},{y2}) saved to check_crop_{name}.png")

    cv2.imwrite('check_frame.png', vis)
    print(f"Annotated frame saved to check_frame.png")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_crops.py <frame_path>")
        sys.exit(1)
    check(sys.argv[1])
