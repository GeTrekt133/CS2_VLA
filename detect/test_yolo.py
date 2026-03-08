"""
Test trained YOLO11 model on real screenshots with detection visualization.

Loads checkpoint, runs inference on all images in a directory, draws bboxes,
saves annotated images + summary grid + statistics.

Usage (server):
    python test_yolo.py \
        --checkpoint runs/yolo11l_body/best.pt \
        --images ./real_screenshots \
        --yaml yolo11.yaml \
        --scale l \
        --nc 1 \
        --conf 0.25 \
        --iou 0.45 \
        --output ./test_results
"""

import argparse
import math
import os
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import yaml

# Reuse architecture from train_yolo
from train_yolo import YOLO11, xywh2xyxy
from torchvision.ops import batched_nms


CLASS_COLORS = {
    0: (0, 255, 0),   # body — green
    1: (0, 0, 255),   # head — red
}


def get_class_names(nc):
    if nc == 1:
        return {0: 'body'}
    return {0: 'body', 1: 'head'}


def load_checkpoint(yaml_path, checkpoint_path, nc, scale, device):
    """Build model and load trained weights from checkpoint."""
    model = YOLO11(cfg=yaml_path, nc=nc, scale=scale)

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Support both direct state_dict and nested checkpoint formats
    if 'ema_state_dict' in ckpt:
        state = ckpt['ema_state_dict']
        print(f"  Using EMA weights (epoch {ckpt.get('epoch', '?')})")
    elif 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
        print(f"  Using model weights (epoch {ckpt.get('epoch', '?')})")
    elif 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: YOLO11-{scale}, nc={nc}, params={n_params:,}")

    if 'best_map' in ckpt:
        print(f"  Best mAP@50: {ckpt['best_map']:.4f}")

    return model


@torch.no_grad()
def run_inference(model, img_path, device, img_size=640, conf_thresh=0.25,
                  nms_iou=0.45, nc=1):
    """Run inference on a single image. Returns (boxes_xyxy, scores, classes) in pixel coords."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None, None, None

    orig_h, orig_w = img.shape[:2]

    # Resize to model input (stretch, matching training) + BGR→RGB
    resized = cv2.resize(img, (img_size, img_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    input_batch = tensor.unsqueeze(0).to(device)

    output = model(input_batch)
    decoded = output[0]   # [1, 4+nc, N]
    pred = decoded[0]     # [4+nc, N]

    boxes_xywh = pred[:4].T   # [N, 4] in img_size pixel coords
    cls_scores = pred[4:].T   # [N, nc]
    max_sc, max_cl = cls_scores.max(1)

    mask = max_sc > conf_thresh
    if mask.sum() == 0:
        return img, torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long)

    bx = xywh2xyxy(boxes_xywh[mask])
    sc = max_sc[mask]
    cl = max_cl[mask]
    keep = batched_nms(bx, sc, cl, nms_iou)[:100]
    bx, sc, cl = bx[keep], sc[keep], cl[keep]

    # Scale boxes back to original image resolution
    sx = orig_w / img_size
    sy = orig_h / img_size
    bx[:, 0] *= sx
    bx[:, 2] *= sx
    bx[:, 1] *= sy
    bx[:, 3] *= sy

    # Clamp to image bounds
    bx[:, 0].clamp_(0, orig_w)
    bx[:, 1].clamp_(0, orig_h)
    bx[:, 2].clamp_(0, orig_w)
    bx[:, 3].clamp_(0, orig_h)

    return img, bx.cpu(), sc.cpu(), cl.cpu().long()


def draw_detections(img, boxes, scores, classes, class_names, line_width=2):
    """Draw bboxes with labels on image. Returns annotated copy."""
    canvas = img.copy()
    h, w = canvas.shape[:2]

    # Scale font/line based on image size
    font_scale = max(0.4, min(w, h) / 1200)
    thickness = max(1, line_width)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].int().tolist()
        c = int(classes[i].item())
        conf = scores[i].item()
        color = CLASS_COLORS.get(c, (255, 255, 0))
        label = f"{class_names.get(c, str(c))} {conf:.2f}"

        # Box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        # Label background
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                              font_scale, 1)
        label_y1 = max(0, y1 - th - 6)
        cv2.rectangle(canvas, (x1, label_y1), (x1 + tw + 4, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1,
                    cv2.LINE_AA)

    return canvas


def make_summary_grid(images, n_cols=4, cell_size=640):
    """Create a grid of images resized to cell_size."""
    n = len(images)
    if n == 0:
        return None
    n_rows = math.ceil(n / n_cols)
    grid = np.zeros((n_rows * cell_size, n_cols * cell_size, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        row, col = i // n_cols, i % n_cols
        resized = cv2.resize(img, (cell_size, cell_size))
        y, x = row * cell_size, col * cell_size
        grid[y:y + cell_size, x:x + cell_size] = resized

    return grid


def main():
    p = argparse.ArgumentParser(description='Test YOLO11 on real screenshots')

    p.add_argument('--checkpoint', required=True,
                   help='Path to trained checkpoint (.pt)')
    p.add_argument('--images', required=True,
                   help='Directory with test images (png/jpg)')
    p.add_argument('--yaml', default='yolo11.yaml',
                   help='Path to yolo11.yaml')
    p.add_argument('--scale', default='l', choices=['n', 's', 'm', 'l', 'x'])
    p.add_argument('--nc', type=int, default=1)
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.25,
                   help='Confidence threshold')
    p.add_argument('--iou', type=float, default=0.45,
                   help='NMS IoU threshold')
    p.add_argument('--output', default='./test_results',
                   help='Output directory for annotated images')
    p.add_argument('--grid-cols', type=int, default=4,
                   help='Columns in summary grid')
    p.add_argument('--max-grid', type=int, default=32,
                   help='Max images in summary grid')
    p.add_argument('--save-all', action='store_true',
                   help='Save all annotated images (not just grid)')
    p.add_argument('--no-cuda', action='store_true')

    args = p.parse_args()

    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available()
                          else 'cuda')
    class_names = get_class_names(args.nc)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('YOLO11 Test — CS2 Detection')
    print('=' * 60)
    print(f'  Device:     {device}')
    print(f'  Checkpoint: {args.checkpoint}')
    print(f'  Images:     {args.images}')
    print(f'  Conf:       {args.conf}')
    print(f'  NMS IoU:    {args.iou}')
    print(f'  Output:     {output_dir}')
    print()

    # Load model
    print('Loading model...')
    model = load_checkpoint(args.yaml, args.checkpoint, args.nc, args.scale, device)
    print()

    # Collect images
    img_dir = Path(args.images)
    image_paths = sorted(
        list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg')) +
        list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.bmp'))
    )
    print(f'Found {len(image_paths)} images')
    if not image_paths:
        print('No images found, exiting.')
        return

    # Run inference
    total_det = 0
    per_class_count = {c: 0 for c in range(args.nc)}
    all_confs = []
    images_with_det = 0
    images_no_det = 0
    det_per_image = []
    grid_images = []
    times = []

    for img_path in tqdm(image_paths, desc='Inference'):
        t0 = time.perf_counter()
        img, boxes, scores, classes = run_inference(
            model, img_path, device, args.img_size, args.conf, args.iou, args.nc)
        dt = time.perf_counter() - t0
        times.append(dt)

        if img is None:
            continue

        n_det = len(boxes)
        det_per_image.append(n_det)

        if n_det > 0:
            images_with_det += 1
            total_det += n_det
            for c_id in range(args.nc):
                per_class_count[c_id] += (classes == c_id).sum().item()
            all_confs.extend(scores.tolist())
        else:
            images_no_det += 1

        # Draw detections
        annotated = draw_detections(img, boxes, scores, classes, class_names)

        # Add info text at top
        info = f"{img_path.name} | {n_det} det"
        if n_det > 0:
            info += f" | conf: {scores.min():.2f}-{scores.max():.2f}"
        cv2.putText(annotated, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Save individual images
        if args.save_all:
            cv2.imwrite(str(output_dir / f'det_{img_path.name}'), annotated)

        # Collect for grid
        if len(grid_images) < args.max_grid:
            grid_images.append(annotated)

    # Statistics
    n_images = len(det_per_image)
    mean_conf = float(np.mean(all_confs)) if all_confs else 0.0
    max_conf = float(np.max(all_confs)) if all_confs else 0.0
    min_conf = float(np.min(all_confs)) if all_confs else 0.0
    mean_det = float(np.mean(det_per_image)) if det_per_image else 0.0
    median_det = float(np.median(det_per_image)) if det_per_image else 0.0
    mean_time = float(np.mean(times)) if times else 0.0
    fps = 1.0 / mean_time if mean_time > 0 else 0.0

    print()
    print('=' * 60)
    print('Results')
    print('=' * 60)
    print(f'  Images processed:  {n_images}')
    print(f'  With detections:   {images_with_det} ({images_with_det/max(n_images,1)*100:.1f}%)')
    print(f'  No detections:     {images_no_det} ({images_no_det/max(n_images,1)*100:.1f}%)')
    print(f'  Total detections:  {total_det}')
    print(f'  Det/image:         mean={mean_det:.1f}, median={median_det:.1f}')
    for c_id in range(args.nc):
        print(f'  {class_names[c_id]:>16s}:  {per_class_count[c_id]}')
    print(f'  Confidence:        mean={mean_conf:.3f}, min={min_conf:.3f}, max={max_conf:.3f}')
    print(f'  Speed:             {mean_time*1000:.1f} ms/img ({fps:.1f} FPS)')
    print('=' * 60)

    # Confidence distribution histogram
    if all_confs:
        hist_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        confs_arr = np.array(all_confs)
        bins = np.linspace(0, 1, 21)
        hist, _ = np.histogram(confs_arr, bins=bins)
        max_count = max(hist.max(), 1)

        bar_w = 25
        for i, count in enumerate(hist):
            bar_h = int(count / max_count * 300)
            x1 = 50 + i * bar_w
            y1 = 350 - bar_h
            color = (0, int(255 * (i / 20)), int(255 * (1 - i / 20)))
            cv2.rectangle(hist_img, (x1, y1), (x1 + bar_w - 2, 350), color, -1)
            if count > 0:
                cv2.putText(hist_img, str(count), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

        # Axis labels
        for i in range(0, 21, 5):
            x = 50 + i * bar_w
            cv2.putText(hist_img, f'{i*5}%', (x - 5, 370),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.putText(hist_img, 'Confidence Distribution', (150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(hist_img, f'N={len(all_confs)}, mean={mean_conf:.3f}', (180, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.imwrite(str(output_dir / 'confidence_hist.png'), hist_img)

    # Summary grid
    if grid_images:
        # Sort: images with most detections first
        sorted_imgs = sorted(grid_images,
                             key=lambda x: -len(np.where(
                                 np.all(x[:25, :, :] != 0, axis=2))[0]))
        grid = make_summary_grid(sorted_imgs[:args.max_grid],
                                 n_cols=args.grid_cols)
        if grid is not None:
            cv2.imwrite(str(output_dir / 'summary_grid.jpg'), grid,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f'\nSaved summary grid ({min(len(grid_images), args.max_grid)} images)')

    # Save stats to text file
    stats_path = output_dir / 'stats.txt'
    with open(stats_path, 'w') as f:
        f.write(f"YOLO11-{args.scale} Test Results\n")
        f.write(f"{'='*40}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Images dir: {args.images}\n")
        f.write(f"Conf thresh: {args.conf}\n")
        f.write(f"NMS IoU: {args.iou}\n")
        f.write(f"NC: {args.nc}\n\n")
        f.write(f"Images: {n_images}\n")
        f.write(f"With detections: {images_with_det} ({images_with_det/max(n_images,1)*100:.1f}%)\n")
        f.write(f"Total detections: {total_det}\n")
        f.write(f"Det/image: mean={mean_det:.2f}, median={median_det:.1f}\n")
        for c_id in range(args.nc):
            f.write(f"{class_names[c_id]}: {per_class_count[c_id]}\n")
        f.write(f"Confidence: mean={mean_conf:.4f}, min={min_conf:.4f}, max={max_conf:.4f}\n")
        f.write(f"Speed: {mean_time*1000:.1f} ms/img ({fps:.1f} FPS)\n")

    print(f'Stats saved to {stats_path}')
    print(f'Output dir: {output_dir}')


if __name__ == '__main__':
    main()
