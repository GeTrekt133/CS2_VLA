"""
Pre-compute YOLO backbone features and detection vectors for the entire dataset.
Saves feat_cache.pt and det_cache.pt to disk — Train.py loads them on startup
so the first epoch runs at full speed (no cold-start).

Usage:
    cd final_model
    CUDA_VISIBLE_DEVICES=0 python ../tools/warmup_cache.py \
        --dataset /data/msirotkin/dataset_train.json \
        --yolo-weights /data/msirotkin/yolo11l_best.pt \
        --feat-cache /data/msirotkin/feat_cache.pt \
        --det-cache  /data/msirotkin/det_cache.pt \
        --batch-size 32 \
        --img-size 640

Then in Train.py set:
    FEAT_CACHE_PATH = '/data/msirotkin/feat_cache.pt'
    DET_CACHE_PATH  = '/data/msirotkin/det_cache.pt'
"""

import sys
import os
import json
import argparse
import threading
import time
import csv
import cv2
import numpy as np
import torch
import torchvision.ops as ops
from tqdm import tqdm

# Add final_model to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'final_model'))

from Yolo import DetectionModel, FeatureCache, load_pretrained_weights
from Yolo import Detect

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


# ──────────────────────────────────────────────────────────────────────────────
# Resource monitor (background thread)
# ──────────────────────────────────────────────────────────────────────────────

class _Monitor:
    def __init__(self, log_file: str, interval: float = 15.0):
        self.log_file = log_file
        self.interval = interval
        self._iter = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        with open(log_file, 'w', newline='') as f:
            csv.writer(f).writerow(['time', 'iter', 'ram_used_gb', 'ram_total_gb',
                                    'ram_pct', 'vram_alloc_gb', 'vram_reserv_gb'])

    def start(self):
        if _PSUTIL:
            psutil.cpu_percent()  # warm up
        self._thread.start()

    def set_iter(self, i: int):
        with self._lock:
            self._iter = i

    def _run(self):
        while not self._stop.wait(self.interval):
            with self._lock:
                it = self._iter
            t = time.strftime('%H:%M:%S')
            ram_u = ram_t = ram_p = 0.0
            if _PSUTIL:
                vm = psutil.virtual_memory()
                ram_u = round(vm.used / 1e9, 1)
                ram_t = round(vm.total / 1e9, 1)
                ram_p = vm.percent
            vram_a = vram_r = 0.0
            if torch.cuda.is_available():
                vram_a = round(torch.cuda.memory_allocated() / 1e9, 2)
                vram_r = round(torch.cuda.memory_reserved()  / 1e9, 2)
            print(f'[Monitor iter={it}]  RAM {ram_u}/{ram_t}GB ({ram_p}%)  '
                  f'VRAM alloc={vram_a}GB reserv={vram_r}GB')
            with open(self.log_file, 'a', newline='') as f:
                csv.writer(f).writerow([t, it, ram_u, ram_t, ram_p, vram_a, vram_r])


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (copied from Train.py to avoid import side effects)
# ──────────────────────────────────────────────────────────────────────────────

def load_image_tensor(path: str, img_size: int = 640) -> torch.Tensor:
    img = cv2.imread(path)
    if img is None:
        return torch.zeros(3, img_size, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


def _det_vec_from_bboxes(bboxes: torch.Tensor, max_det: int, img_size: int,
                          device) -> torch.Tensor:
    """Convert postprocessed bboxes tensor to a fixed-size detection vector."""
    boxes_xyxy = ops.box_convert(bboxes[:, :4], in_fmt='cxcywh', out_fmt='xyxy')
    conf = bboxes[:, 4]
    cls  = bboxes[:, 5]
    if boxes_xyxy.shape[0] > 0:
        keep = ops.nms(boxes_xyxy, conf, iou_threshold=0.3)
        boxes_xyxy, conf, cls = boxes_xyxy[keep], conf[keep], cls[keep]
    stack = []
    for j, cl in enumerate(cls):
        if int(cl.item()) == 0:
            stack.append(torch.cat([boxes_xyxy[j] / img_size, conf[j].unsqueeze(0)]))
    if not stack:
        return torch.zeros(max_det * 5, device=device)
    det_vec = torch.cat(stack)
    pad = max_det * 5 - det_vec.shape[0]
    if pad > 0:
        det_vec = torch.cat([det_vec, torch.zeros(pad, device=device)])
    else:
        det_vec = det_vec[:max_det * 5]
    return det_vec


@torch.no_grad()
def get_det_vectors_batch(paths: list, yolo: DetectionModel, device,
                          max_det: int = 20, img_size: int = 640,
                          nc: int = 2) -> list:
    """Run detection on a batch of images, return list of det vectors (cpu, half)."""
    imgs = torch.stack([load_image_tensor(p, img_size) for p in paths]).to(device)
    raw = yolo.forward_detections_only(imgs)
    decoded = raw[0] if isinstance(raw, tuple) else raw
    bboxes_batch = Detect.postprocess(decoded, max_det, nc)
    result = []
    for bboxes in bboxes_batch:
        vec = _det_vec_from_bboxes(bboxes, max_det, img_size, device)
        result.append(vec.cpu().half())
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Collect unique frame paths from dataset JSON
# ──────────────────────────────────────────────────────────────────────────────

def collect_frame_paths(dataset_json: str) -> list:
    with open(dataset_json, 'r') as f:
        data = json.load(f)

    paths = set()
    for demo in data['demos']:
        demo_path = demo['demo_path']
        for rnd in demo['rounds']:
            for state in rnd['states']:
                tick = state['tick']
                paths.add(os.path.join(demo_path, f"tick_{tick}.jpg"))

    paths = sorted(paths)
    print(f"Found {len(paths):,} unique frame paths")
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',      required=True, help='Path to dataset JSON')
    p.add_argument('--yolo-weights', default=None,  help='Path to YOLO .pt weights (optional)')
    p.add_argument('--feat-cache',   required=True, help='Output feat_cache.pt path')
    p.add_argument('--det-cache',    required=True, help='Output det_cache.pt path')
    p.add_argument('--batch-size',   type=int, default=32)
    p.add_argument('--img-size',     type=int, default=640)
    p.add_argument('--max-det',      type=int, default=20)
    p.add_argument('--nc',           type=int, default=2)
    p.add_argument('--skip-existing', action='store_true',
                   help='Load existing caches and only process missing frames')
    p.add_argument('--max-cache-size', type=int, default=100_000,
                   help='Max entries kept in RAM (LRU eviction). '
                        '500K frames * 3.7MB = 1.85TB — set to what fits in RAM. '
                        'Default 100K ~ 370GB.')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load YOLO
    yolo = DetectionModel(nc=args.nc)
    if args.yolo_weights and os.path.exists(args.yolo_weights):
        yolo = load_pretrained_weights(yolo, args.yolo_weights)
        print(f"Loaded YOLO weights from {args.yolo_weights}")
    yolo = yolo.to(device).eval()
    for p_ in yolo.model.parameters():
        p_.requires_grad_(False)

    # Collect frame paths
    all_paths = collect_frame_paths(args.dataset)

    max_size = args.max_cache_size
    print(f"Cache size limit: {max_size:,} entries "
          f"(~{max_size * 3.7 / 1024:.0f} GB RAM)")

    # Load existing caches if requested
    if args.skip_existing and os.path.exists(args.feat_cache):
        feat_cache = FeatureCache.load(args.feat_cache, max_size=max_size)
    else:
        feat_cache = FeatureCache(max_size=max_size)

    if args.skip_existing and os.path.exists(args.det_cache):
        det_cache = FeatureCache.load(args.det_cache, max_size=max_size)
    else:
        det_cache = FeatureCache(max_size=max_size)

    # Filter to only missing paths
    missing_paths = [p for p in all_paths if feat_cache.get(p) is None]
    print(f"Frames to process: {len(missing_paths):,} "
          f"(already cached: {len(all_paths) - len(missing_paths):,})")

    monitor = _Monitor(
        log_file=os.path.join(os.path.dirname(args.feat_cache), 'warmup_resource_log.csv'),
    )
    monitor.start()

    if not missing_paths:
        print("All frames already cached.")
    else:
        # ---- Feature cache (P3 + P5) ----
        print("\n=== Building feat_cache (P3 + P5) ===")
        batch_size = args.batch_size
        for start in tqdm(range(0, len(missing_paths), batch_size)):
            monitor.set_iter(start // batch_size)
            batch_paths = missing_paths[start:start + batch_size]
            imgs = torch.stack([load_image_tensor(p, args.img_size) for p in batch_paths]).to(device)
            with torch.no_grad():
                p3, p5 = yolo.extract_features(imgs)
            for i, path in enumerate(batch_paths):
                feat_cache.put(path, (p3[i], p5[i]))

        feat_cache.save(args.feat_cache)

        # ---- Detection cache ----
        print("\n=== Building det_cache ===")
        missing_det = [p for p in all_paths if det_cache.get(p) is None]
        for start in tqdm(range(0, len(missing_det), batch_size)):
            monitor.set_iter(start // batch_size)
            batch_paths = missing_det[start:start + batch_size]
            # Separate existing vs missing files
            existing = [p for p in batch_paths if os.path.exists(p)]
            missing  = [p for p in batch_paths if not os.path.exists(p)]
            for p in missing:
                det_cache.put(p, torch.zeros(args.max_det * 5).half())
            if existing:
                det_vecs = get_det_vectors_batch(
                    existing, yolo, device, args.max_det, args.img_size, args.nc)
                for p, v in zip(existing, det_vecs):
                    det_cache.put(p, v)

        det_cache.save(args.det_cache)

    print("\nDone.")
    print(f"  feat_cache: {len(feat_cache):,} entries -> {args.feat_cache}")
    print(f"  det_cache:  {len(det_cache):,} entries -> {args.det_cache}")


if __name__ == '__main__':
    main()
