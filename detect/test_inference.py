"""Test YOLO11n checkpoint on real screenshots."""

import sys
sys.path.insert(0, r'c:\Users\misas\CS2_NN\detect')

import cv2
import numpy as np
import torch
import random
from pathlib import Path

from train_yolo import YOLO11n

# ============================================================================
# Config
# ============================================================================

CHECKPOINT = r'C:\Users\misas\CS2_NN\runs\yolo11n_body_head\best.pt'
IMAGES_DIR = r'D:\screenshots2_clean\1-0b0a3f84-0078-4a21-9a18-8671c37009e0-1-1'
YAML_CFG = r'C:\Users\misas\CS2_NN\src\yolo11n.yaml'
OUTPUT_DIR = r'C:\Users\misas\CS2_NN\detect\test_output'
NC = 2
IMG_SIZE = 640
CONF_THRESH = 0.1
IOU_THRESH = 0.45
NUM_SAMPLES = 0  # 0 = all images

CLASS_NAMES = {0: 'body', 1: 'head'}
CLASS_COLORS = {
    0: (0, 255, 0),   # body = green
    1: (0, 0, 255),   # head = red
}


def nms(boxes, scores, iou_threshold=0.45):
    """Simple NMS. boxes: [N,4] xyxy, scores: [N]."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)

    keep = []
    while len(order) > 0:
        i = order[0].item()
        keep.append(i)
        if len(order) == 1:
            break

        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return keep


def postprocess(output, conf_thresh=0.25, iou_thresh=0.45, nc=2):
    """
    Process raw model output into detections.
    output[0]: [B, 4+nc, N] decoded predictions (xyxy + class scores)
    Returns list of [N, 6] tensors: x1, y1, x2, y2, conf, cls
    """
    decoded = output[0]  # [B, 4+nc, N]
    batch_results = []

    for b in range(decoded.shape[0]):
        pred = decoded[b]  # [4+nc, N]
        boxes_xywh = pred[:4].T  # [N, 4] - x_center, y_center, w, h (pixel coords)
        cls_scores = pred[4:].T  # [N, nc]

        # Convert xywh to xyxy
        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)  # [N, 4]

        # Best class per detection
        max_scores, max_cls = cls_scores.max(dim=1)  # [N], [N]

        # Filter by confidence
        mask = max_scores > conf_thresh
        boxes_xyxy = boxes_xyxy[mask]
        max_scores = max_scores[mask]
        max_cls = max_cls[mask]

        if len(boxes_xyxy) == 0:
            batch_results.append(torch.zeros(0, 6))
            continue

        # Per-class NMS
        keep_all = []
        for c in range(nc):
            c_mask = max_cls == c
            if c_mask.sum() == 0:
                continue
            c_boxes = boxes_xyxy[c_mask]
            c_scores = max_scores[c_mask]
            c_indices = torch.where(c_mask)[0]

            k = nms(c_boxes, c_scores, iou_thresh)
            keep_all.extend(c_indices[k].tolist())

        if not keep_all:
            batch_results.append(torch.zeros(0, 6))
            continue

        keep_all = sorted(keep_all)
        result = torch.cat([
            boxes_xyxy[keep_all],
            max_scores[keep_all].unsqueeze(1),
            max_cls[keep_all].float().unsqueeze(1),
        ], dim=1)

        batch_results.append(result)

    return batch_results


def draw_detections(img, detections, img_size=640):
    """Draw detection boxes on image. detections: [N, 6] x1,y1,x2,y2,conf,cls in model coords."""
    h, w = img.shape[:2]
    scale_x = w / img_size
    scale_y = h / img_size

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cls_id = int(cls_id.item())
        conf = conf.item()

        # Scale back to original image coords
        x1 = int(x1.item() * scale_x)
        y1 = int(y1.item() * scale_y)
        x2 = int(x2.item() * scale_x)
        y2 = int(y2.item() * scale_y)

        color = CLASS_COLORS.get(cls_id, (255, 255, 0))
        label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return img


def draw_detections_640(canvas, detections):
    """Draw detection boxes directly on 640x640 letterboxed image (no rescaling needed)."""
    img = canvas.copy()

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cls_id = int(cls_id.item())
        conf = conf.item()

        x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())

        color = CLASS_COLORS.get(cls_id, (255, 255, 0))
        label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img


def preprocess(img_path, img_size=640):
    """Load and preprocess image for inference. Direct resize to square (matches training)."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None, None

    # Direct resize to 640x640 — same as SyntheticYOLODataset training
    resized = cv2.resize(img, (img_size, img_size))

    # To tensor
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0  # [3, 640, 640]
    return tensor, img, resized


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load model
    print(f'Loading checkpoint: {CHECKPOINT}')
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    print(f'  Epoch: {ckpt.get("epoch", "?")}, mAP@50: {ckpt.get("mAP50", "?")}, best_mAP: {ckpt.get("best_map", "?")}')

    model = YOLO11n(cfg=YAML_CFG, nc=NC).to(device)

    # Try EMA weights first, fallback to model weights
    if 'ema_state_dict' in ckpt:
        model.load_state_dict(ckpt['ema_state_dict'])
        print('  Loaded EMA weights')
    else:
        model.load_state_dict(ckpt['model_state_dict'])
        print('  Loaded model weights')

    model.eval()

    # Get images
    img_dir = Path(IMAGES_DIR)
    images = sorted(img_dir.glob('*.png')) + sorted(img_dir.glob('*.jpg'))
    print(f'\nFound {len(images)} images in {IMAGES_DIR}')

    if NUM_SAMPLES > 0 and len(images) > NUM_SAMPLES:
        images = random.sample(images, NUM_SAMPLES)
        print(f'Sampling {NUM_SAMPLES} random images')

    # Output directories
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_640_dir = Path(OUTPUT_DIR) / '640x640'
    out_640_dir.mkdir(parents=True, exist_ok=True)

    # Inference
    total_detections = 0
    body_count = 0
    head_count = 0

    with torch.no_grad():
        for i, img_path in enumerate(images):
            tensor, orig_img, canvas_640 = preprocess(img_path, IMG_SIZE)
            if tensor is None:
                print(f'  [{i+1}] SKIP (cannot read): {img_path.name}')
                continue

            # Inference
            input_batch = tensor.unsqueeze(0).to(device)  # [1, 3, 640, 640]
            output = model(input_batch)

            # Postprocess
            results = postprocess(output, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH, nc=NC)
            dets = results[0]  # [N, 6]

            n_det = len(dets)
            n_body = (dets[:, 5] == 0).sum().item() if n_det > 0 else 0
            n_head = (dets[:, 5] == 1).sum().item() if n_det > 0 else 0
            total_detections += n_det
            body_count += n_body
            head_count += n_head

            # Draw on original image
            vis = draw_detections(orig_img.copy(), dets, IMG_SIZE)
            out_path = out_dir / f'{img_path.stem}_det.jpg'
            cv2.imwrite(str(out_path), vis)

            # Draw on 640x640 letterboxed image
            vis_640 = draw_detections_640(canvas_640, dets)
            out_640_path = out_640_dir / f'{img_path.stem}_det.jpg'
            cv2.imwrite(str(out_640_path), vis_640)

            if (i + 1) % 10 == 0 or n_det > 0:
                confs = dets[:, 4].tolist() if n_det > 0 else []
                conf_str = f' confs={[f"{c:.2f}" for c in confs[:5]]}' if confs else ''
                print(f'  [{i+1}/{len(images)}] {img_path.name}: {n_body} body, {n_head} head{conf_str}')

    print(f'\n{"="*50}')
    print(f'Results:')
    print(f'  Images tested:    {len(images)}')
    print(f'  Total detections: {total_detections}')
    print(f'  Body:             {body_count}')
    print(f'  Head:             {head_count}')
    print(f'  Avg det/image:    {total_detections/max(len(images),1):.1f}')
    print(f'  Output (original): {OUTPUT_DIR}')
    print(f'  Output (640x640):  {out_640_dir}')


if __name__ == '__main__':
    main()
