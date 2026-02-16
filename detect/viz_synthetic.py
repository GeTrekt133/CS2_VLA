"""Quick visualization of SyntheticYOLODataset samples."""

import sys
sys.path.insert(0, r'c:\Users\misas\CS2_NN\detect')

import cv2
import numpy as np
import torch
import random
from pathlib import Path
from dataset_yolo import SyntheticYOLODataset

BACKGROUNDS = r'D:\backgrounds'
RGBA_CROPS = r'D:\yolo_dataset\crops_rgba'
OUT_DIR = r'c:\Users\misas\CS2_NN\detect\viz_output'

CLASS_COLORS = {
    0: (0, 255, 0),   # body = green
    1: (0, 0, 255),   # head = red
}
CLASS_NAMES = {0: 'body', 1: 'head'}


def draw_bboxes(image_tensor, bboxes, class_labels, img_size=640):
    """Draw bboxes on image tensor, return BGR numpy array."""
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8).copy()
    # RGB -> BGR for cv2
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
        class_labels = class_labels.cpu().numpy()

    h, w = img.shape[:2]

    for bbox, cls in zip(bboxes, class_labels):
        xc, yc, bw, bh = bbox
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        color = CLASS_COLORS.get(int(cls), (255, 255, 0))
        label = CLASS_NAMES.get(int(cls), str(int(cls)))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img


def make_grid(images, ncols=4, pad=4):
    """Arrange images in a grid with padding."""
    nrows = (len(images) + ncols - 1) // ncols
    h, w = images[0].shape[:2]

    grid = np.ones((nrows * (h + pad) - pad, ncols * (w + pad) - pad, 3), dtype=np.uint8) * 40

    for i, img in enumerate(images):
        r, c = divmod(i, ncols)
        y = r * (h + pad)
        x = c * (w + pad)
        grid[y:y+h, x:x+w] = img

    return grid


def main():
    out = Path(OUT_DIR)
    out.mkdir(exist_ok=True, parents=True)

    print("Loading SyntheticYOLODataset...")
    ds = SyntheticYOLODataset(
        backgrounds_dir=BACKGROUNDS,
        rgba_crops_dir=RGBA_CROPS,
        img_size=640,
        augment=True,
        min_players=1,
        max_players=5,
        negative_ratio=0.15,
        flash_prob=0.15,
        occlusion_prob=0.3,
        scale_range=(1.5, 3.0),
    )
    print(f"Dataset size: {len(ds)}")

    # ---- Grid 1: 16 random samples with all augmentations ----
    print("\nGenerating 16 random samples...")
    imgs = []
    stats = {'body': 0, 'head': 0, 'negative': 0, 'total_players': 0}

    for i in range(16):
        idx = random.randint(0, len(ds) - 1)
        sample = ds[idx]
        img = draw_bboxes(sample['image'], sample['bboxes'], sample['class_labels'])
        imgs.append(img)

        labels = sample['class_labels'].numpy() if len(sample['class_labels']) > 0 else []
        stats['body'] += (labels == 0).sum() if len(labels) > 0 else 0
        stats['head'] += (labels == 1).sum() if len(labels) > 0 else 0
        if len(labels) == 0:
            stats['negative'] += 1
        stats['total_players'] += len(labels)

    grid = make_grid(imgs, ncols=4)
    path1 = str(out / 'synthetic_grid_16.png')
    cv2.imwrite(path1, grid)
    print(f"Saved: {path1}")
    print(f"Stats: {stats['body']} body, {stats['head']} head, "
          f"{stats['negative']} negative, avg {stats['total_players']/16:.1f} bboxes/img")

    # ---- Grid 2: augmentation comparison (same bg, different augmentations) ----
    print("\nGenerating augmentation comparison (same background, 8 variants)...")
    imgs2 = []
    fixed_idx = random.randint(0, len(ds) - 1)
    for i in range(8):
        sample = ds[fixed_idx]
        img = draw_bboxes(sample['image'], sample['bboxes'], sample['class_labels'])
        imgs2.append(img)

    grid2 = make_grid(imgs2, ncols=4)
    path2 = str(out / 'augmentation_variants.png')
    cv2.imwrite(path2, grid2)
    print(f"Saved: {path2}")

    # ---- Grid 3: specific augmentations ----
    print("\nGenerating specific augmentation examples...")

    # Flash only
    ds.flash_prob = 1.0
    ds.occlusion_prob = 0.0
    sample_flash = ds[random.randint(0, len(ds) - 1)]
    img_flash = draw_bboxes(sample_flash['image'], sample_flash['bboxes'], sample_flash['class_labels'])

    # Occlusion only
    ds.flash_prob = 0.0
    ds.occlusion_prob = 1.0
    sample_occ = ds[random.randint(0, len(ds) - 1)]
    img_occ = draw_bboxes(sample_occ['image'], sample_occ['bboxes'], sample_occ['class_labels'])

    # No augmentations
    ds.flash_prob = 0.0
    ds.occlusion_prob = 0.0
    ds.augment = False
    sample_clean = ds[random.randint(0, len(ds) - 1)]
    img_clean = draw_bboxes(sample_clean['image'], sample_clean['bboxes'], sample_clean['class_labels'])

    # All augmentations
    ds.augment = True
    ds.flash_prob = 0.15
    ds.occlusion_prob = 0.3
    sample_all = ds[random.randint(0, len(ds) - 1)]
    img_all = draw_bboxes(sample_all['image'], sample_all['bboxes'], sample_all['class_labels'])

    # Add labels
    for img, label in [(img_clean, 'Clean'), (img_flash, 'Flash'),
                        (img_occ, 'Beam Occlusion'), (img_all, 'Mixed Augs')]:
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    grid3 = make_grid([img_clean, img_flash, img_occ, img_all], ncols=2)
    path3 = str(out / 'augmentation_types.png')
    cv2.imwrite(path3, grid3)
    print(f"Saved: {path3}")

    print(f"\nAll visualizations saved to {OUT_DIR}")


if __name__ == '__main__':
    main()
