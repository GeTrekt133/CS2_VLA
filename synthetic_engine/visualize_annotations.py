"""
Visualize COCO annotations on captured frames.

Usage:
    python visualize_annotations.py --dataset D:/DetectionDataset
    python visualize_annotations.py --dataset D:/DetectionDataset --max 10
    python visualize_annotations.py --dataset D:/DetectionDataset --inplace  (overwrite originals)
"""

from __future__ import annotations

import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont


CLASS_COLORS = {
    'ct':       (50, 150, 255),    # blue
    't':        (255, 140, 0),     # orange
    'ct_head':  (0, 220, 255),     # cyan
    't_head':   (255, 220, 0),     # yellow
    'bomb':     (255, 0, 255),     # magenta
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='Dataset folder containing annotations.json + frames/')
    ap.add_argument('--max',     type=int, default=None, help='Limit number of images to render')
    ap.add_argument('--inplace', action='store_true', help='Overwrite original frames instead of saving to viz/')
    args = ap.parse_args()

    ann_path = os.path.join(args.dataset, 'annotations.json')
    with open(ann_path) as f:
        coco = json.load(f)

    # Build lookups
    cats = {c['id']: c['name'] for c in coco['categories']}
    img_by_id = {img['id']: img for img in coco['images']}
    anns_by_img = {}
    for a in coco['annotations']:
        anns_by_img.setdefault(a['image_id'], []).append(a)

    print(f'Dataset: {len(coco["images"])} images, {len(coco["annotations"])} annotations')
    print(f'Classes: {cats}')

    # Stats
    by_cls = {n: 0 for n in cats.values()}
    for a in coco['annotations']:
        by_cls[cats[a['category_id']]] += 1
    print('Annotations per class:')
    for k, v in by_cls.items():
        print(f'  {k:10s}: {v}')

    out_dir = args.dataset if args.inplace else os.path.join(args.dataset, 'viz')
    os.makedirs(out_dir, exist_ok=True) if not args.inplace else None
    if not args.inplace:
        out_dir = os.path.join(args.dataset, 'viz')
        os.makedirs(out_dir, exist_ok=True)

    images = list(coco['images'])
    if args.max:
        images = images[:args.max]

    try:
        font = ImageFont.truetype('arial.ttf', 14)
    except Exception:
        font = ImageFont.load_default()

    frames_dir = os.path.join(args.dataset, 'frames')
    n_drawn = 0
    for img_meta in images:
        img_path = os.path.join(frames_dir, img_meta['file_name'])
        if not os.path.exists(img_path):
            print(f'  missing: {img_path}')
            continue

        img = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        anns = anns_by_img.get(img_meta['id'], [])
        for ann in anns:
            x, y, w, h = ann['bbox']
            cls_name = cats[ann['category_id']]
            color = CLASS_COLORS.get(cls_name, (255, 255, 255))

            draw.rectangle((x, y, x + w, y + h), outline=color, width=2)
            label = cls_name
            if 'slot' in ann: label = f"{cls_name}#{ann['slot']}"
            draw.text((x + 2, max(0, y - 14)), label, fill=color, font=font)

            # Per-bone visibility dots: green=visible, red=occluded; label = "idx:frac"
            for b in ann.get('bones_debug', []):
                bx, by = b['sx'], b['sy']
                bone_color = (0, 255, 0) if b['vis'] else (255, 32, 32)
                r = 3
                draw.ellipse((bx - r, by - r, bx + r, by + r),
                             outline=(0, 0, 0), fill=bone_color, width=1)
                frac = b.get('frac', -1.0)
                txt = f"{b['idx']}" if frac < 0 else f"{b['idx']}:{frac:.2f}"
                draw.text((bx + 4, by - 6), txt, fill=bone_color, font=font)

            # Optional: anchor crosshair for entities with pos_screen (bomb etc)
            if 'pos_screen' in ann:
                px, py = ann['pos_screen']
                r = 4
                draw.ellipse((px - r, py - r, px + r, py + r), outline=color, width=2)
                draw.line((px - r * 2, py, px + r * 2, py), fill=color, width=1)
                draw.line((px, py - r * 2, px, py + r * 2), fill=color, width=1)

        out_path = os.path.join(out_dir, img_meta['file_name'])
        img.save(out_path, quality=90)
        n_drawn += 1

    print(f'\nRendered {n_drawn} images -> {out_dir}')


if __name__ == '__main__':
    main()
