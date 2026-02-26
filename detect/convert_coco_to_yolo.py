"""
Convert COCO detection dataset (CT/T classes) to YOLO format (single body class).

Reads COCO annotations from all splits, merges all categories into class 0 (body),
pools images, and re-splits with custom ratios (default 90/5/5).

Usage:
  python detect/convert_coco_to_yolo.py \
    --coco-dir "C:/Users/misas/Downloads/cs2.v1i.coco" \
    --output "D:/yolo_dataset/cs2_real" \
    --split 0.90 0.05 0.05
"""

import json
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict


def load_coco_split(coco_dir: Path, split: str):
    """Load images + annotations from one COCO split. Returns list of (src_path, w, h, anns)."""
    ann_file = coco_dir / split / '_annotations.coco.json'
    if not ann_file.exists():
        return []

    with open(ann_file, 'r') as f:
        coco = json.load(f)

    id_to_info = {img['id']: img for img in coco['images']}

    img_anns = defaultdict(list)
    for ann in coco['annotations']:
        img_anns[ann['image_id']].append(ann)

    entries = []
    for img_id, info in id_to_info.items():
        src = coco_dir / split / info['file_name']
        if not src.exists():
            continue
        entries.append({
            'src_path': src,
            'file_name': info['file_name'],
            'width': info['width'],
            'height': info['height'],
            'annotations': img_anns.get(img_id, []),
        })

    return entries


def write_yolo(entries, output_dir: Path, split_name: str):
    """Write entries as YOLO images/ + labels/."""
    images_out = output_dir / split_name / 'images'
    labels_out = output_dir / split_name / 'labels'
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    n_ann = 0
    for entry in entries:
        dst_img = images_out / entry['file_name']
        if not dst_img.exists():
            shutil.copy2(str(entry['src_path']), str(dst_img))

        w, h = entry['width'], entry['height']
        lines = []
        for ann in entry['annotations']:
            bx, by, bw, bh = ann['bbox']
            if bw < 2 or bh < 2:
                continue
            x_center = max(0.0, min(1.0, (bx + bw / 2) / w))
            y_center = max(0.0, min(1.0, (by + bh / 2) / h))
            w_norm = max(0.0, min(1.0, bw / w))
            h_norm = max(0.0, min(1.0, bh / h))
            lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            n_ann += 1

        label_name = Path(entry['file_name']).stem + '.txt'
        with open(labels_out / label_name, 'w') as f:
            f.write('\n'.join(lines))

    return len(entries), n_ann


def main():
    p = argparse.ArgumentParser(
        description='Convert COCO to YOLO format (all classes -> body)')
    p.add_argument('--coco-dir', required=True,
                   help='Path to COCO dataset root (with train/valid/test)')
    p.add_argument('--output', required=True,
                   help='Output directory for YOLO format')
    p.add_argument('--split', nargs=3, type=float, default=[0.90, 0.05, 0.05],
                   help='Train/val/test split ratios (default: 0.90 0.05 0.05)')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    coco_dir = Path(args.coco_dir)
    output_dir = Path(args.output)
    train_r, val_r, test_r = args.split

    random.seed(args.seed)

    # Pool all splits
    print(f"Loading COCO from {coco_dir}")
    all_entries = []
    for split in ['train', 'valid', 'test']:
        entries = load_coco_split(coco_dir, split)
        print(f"  {split}: {len(entries)} images, "
              f"{sum(len(e['annotations']) for e in entries)} annotations")
        all_entries.extend(entries)

    # Deduplicate by filename (in case same image appears in multiple splits)
    seen = set()
    unique = []
    for e in all_entries:
        if e['file_name'] not in seen:
            seen.add(e['file_name'])
            unique.append(e)
    all_entries = unique

    print(f"\nTotal unique: {len(all_entries)} images, "
          f"{sum(len(e['annotations']) for e in all_entries)} annotations")

    # Shuffle and split
    random.shuffle(all_entries)
    n = len(all_entries)
    n_train = int(n * train_r)
    n_val = int(n * val_r)

    train_entries = all_entries[:n_train]
    val_entries = all_entries[n_train:n_train + n_val]
    test_entries = all_entries[n_train + n_val:]

    print(f"\nSplit ({train_r:.0%}/{val_r:.0%}/{test_r:.0%}):")

    for name, entries in [('train', train_entries), ('valid', val_entries),
                          ('test', test_entries)]:
        n_img, n_ann = write_yolo(entries, output_dir, name)
        print(f"  {name}: {n_img} images, {n_ann} annotations")

    # data.yaml
    yaml_path = output_dir / 'data.yaml'
    yaml_content = f"""# CS2 Body Detection - Real Data
# Converted from COCO format (CT+T merged into body)
# Split: {train_r:.0%}/{val_r:.0%}/{test_r:.0%}

train: {output_dir / 'train' / 'images'}
val: {output_dir / 'valid' / 'images'}
test: {output_dir / 'test' / 'images'}

nc: 1
names: ['body']
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nSaved to {output_dir}")


if __name__ == '__main__':
    main()
