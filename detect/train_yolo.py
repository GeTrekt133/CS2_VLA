"""
YOLO11n Training — CS2 Body/Head Detection.

Architecture identical to DetectionModel backbone+head in audio_adaptation/src/Yolo.py
(without embed branch). Weights are cross-loadable via load_pretrained_weights.

On-the-fly synthetic data from SyntheticYOLODataset (alpha-composited RGBA crops).
Loss: ultralytics v8DetectionLoss (TaskAlignedAssigner + CIoU + DFL + BCE).

Usage (remote server):
    python train_yolo.py \
        --backgrounds /data/backgrounds \
        --rgba-crops /data/crops_rgba \
        --yaml yolo11n.yaml \
        --pretrained yolo11n.pt \
        --output ./runs \
        --epochs 300 --batch-size 16 --lr 1e-3

Files needed on server:
    - train_yolo.py, dataset_yolo.py, yolo11n.yaml
    - yolo11n.pt  (optional, for pretrained backbone)
    - backgrounds/  (empty map screenshots)
    - crops_rgba/   (player_*.png + metadata.json)

pip install:
    torch torchvision ultralytics opencv-python-headless albumentations tqdm tensorboard
"""

import argparse
import copy
import math
import os
import random
import time

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import yaml

from dataset_yolo import SyntheticYOLODataset, collate_fn

from ultralytics.utils.loss import v8DetectionLoss
from torchvision.ops import box_iou, batched_nms


# ============================================================================
# Architecture (YOLO11n — identical backbone/head to DetectionModel in Yolo.py)
# ============================================================================

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)
    return torch.cat((x1y1, x2y2), dim)


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Detect(nn.Module):
    dynamic = False
    export = False
    format = None
    end2end = False
    max_det = 300
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy = False
    xyxy = False

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.tensor([8., 16., 32.])
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def _inference(self, x):
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        return dist2bbox(bboxes, anchors, xywh=xywh and not self.end2end and not self.xyxy, dim=1)


class YOLO11n(nn.Module):
    """YOLO11n detection-only (no embed branch). Backbone+head identical to DetectionModel."""

    def __init__(self, cfg='yolo11n.yaml', ch=3, nc=2):
        super().__init__()
        with open(cfg, 'r', encoding='utf-8') as f:
            self.yaml = yaml.safe_load(f)
        self.model, self.save = self.parse_model(self.yaml, ch=[ch], nc=nc)

    def parse_model(self, d, ch, nc):
        layers, save = [], []
        all_layers = []
        if 'backbone' in d:
            all_layers += d['backbone']
        if 'head' in d:
            all_layers += d['head']

        gd = d.get('depth_mult', 0.5)
        gw = d.get('width_mult', 0.25)

        for i, (f, n, m, args) in enumerate(all_layers):
            m = eval(m) if isinstance(m, str) else m
            args = [x for x in args]
            n = max(round(n * gd), 1) if n > 1 else n

            if isinstance(f, int):
                c1 = ch[f] if f != -1 else ch[-1]
            elif isinstance(f, list):
                c1 = sum(ch[j] for j in f)
            else:
                c1 = ch[-1]

            if m.__name__ in ("Conv", "C2f", "C3", "C3k2", "SPPF", "C2PSA"):
                c2 = int(args[0] * gw)
                if m.__name__ == "C3k2":
                    if len(args) == 3:
                        args = [c1, c2, 1, args[1], args[2]]
                    else:
                        args = [c1, c2, 1, args[1]]
                else:
                    args = [c1, c2, *args[1:]]
            elif m.__name__ == "Upsample":
                scale = args[1] if len(args) > 1 and args[1] is not None else 2
                mode = args[2] if len(args) > 2 else "nearest"
                args = {"scale_factor": scale, "mode": mode}
            elif m.__name__ in ("Concat",):
                c2 = sum(ch[j] if isinstance(ch[j], int) else ch[j][0] for j in f)
                args = [1]
            elif m.__name__ in ("Detect",):
                args = [nc, [ch[x] for x in f]]
                c2 = None
            else:
                c2 = c1

            if n > 1 and m.__name__ not in ("Detect", "Concat"):
                block = nn.Sequential(*[m(*args) for _ in range(n)])
            elif m.__name__ == "Upsample":
                block = m(**args)
            else:
                block = m(*args)

            block.f = f
            layers.append(block)
            ch.append(c2 if c2 is not None else ch[-1])
            if isinstance(f, list) and len(f) > 1:
                save.extend(x for x in f if x != -1)

        return nn.ModuleList(layers), sorted(save)

    def forward(self, x):
        # Backbone (layers 0-10)
        x = self.model[0](x)
        x = self.model[1](x)
        x = self.model[2](x)
        x = self.model[3](x)
        x4 = self.model[4](x)
        x = self.model[5](x4)
        x6 = self.model[6](x)
        x = self.model[7](x6)
        x = self.model[8](x)
        x = self.model[9](x)
        x10 = self.model[10](x)

        # Head (layers 11-23)
        x = self.model[11](x10)
        x = torch.cat((x, x6), dim=1)
        x = self.model[13](x)
        x13 = x

        x = self.model[14](x13)
        x = torch.cat((x, x4), dim=1)
        x = self.model[16](x)
        x16 = x

        x = self.model[17](x)
        x = torch.cat((x, x13), dim=1)
        x = self.model[19](x)
        x19 = x

        x = self.model[20](x)
        x = torch.cat((x, x10), dim=1)
        x = self.model[22](x)
        x22 = x

        # Detect (layer 23)
        return self.model[23]([x16, x19, x22])


# ============================================================================
# Weight Loading
# ============================================================================

def load_pretrained_weights(model, path):
    """Load pretrained weights with shape matching (handles nc mismatch gracefully)."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)

    if 'model' in ckpt and hasattr(ckpt['model'], 'state_dict'):
        state = ckpt['model'].state_dict()
    elif 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt

    model_state = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(model_state, strict=False)
    print(f"  Pretrained: loaded {loaded}/{loaded + skipped} layers (skipped {skipped})")
    return model


# ============================================================================
# Loss Adapter (wraps model interface for ultralytics v8DetectionLoss)
# ============================================================================

class _LossModelAdapter:
    """Makes YOLO11n compatible with ultralytics v8DetectionLoss."""
    def __init__(self, model, hyp):
        self._model = model
        self.model = model.model   # nn.ModuleList — loss accesses model.model[-1] (Detect)
        self.args = hyp

    def parameters(self):
        return self._model.parameters()


# ============================================================================
# EMA (Exponential Moving Average)
# ============================================================================

class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        # Warmup: decay starts low (~0.1) and ramps up to target (0.9999)
        # At step 1: d=0.18, step 100: d=0.92, step 1000: d=0.99, step 10000: d=0.9999
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        with torch.no_grad():
            for ep, mp in zip(self.ema.parameters(), model.parameters()):
                ep.data.mul_(d).add_(mp.data, alpha=1 - d)
            for eb, mb in zip(self.ema.buffers(), model.buffers()):
                eb.data.copy_(mb.data)


# ============================================================================
# mAP Evaluation
# ============================================================================

def xywh2xyxy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def _compute_ap(recall, precision):
    """All-point interpolation AP."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device, nc=2,
             conf_thresh=0.001, nms_iou=0.65, match_iou=0.5):
    """Validation: loss + mAP@50 in single pass."""
    model.eval()

    total_loss, loss_box, loss_cls, loss_dfl = 0., 0., 0., 0.
    n_batches = 0
    all_preds, all_targets = [], []

    for batch in tqdm(dataloader, desc='  Eval', leave=False):
        images = batch['images'].to(device)
        targets = batch['targets'].to(device)
        bs = images.shape[0]

        output = model(images)          # eval: (decoded, raw_feats)
        decoded = output[0]             # [B, 4+nc, N]

        # --- Loss (v8DetectionLoss extracts raw feats from output[1]) ---
        batch_dict = {
            'batch_idx': targets[:, 0],
            'cls': targets[:, 1],
            'bboxes': targets[:, 2:6],
        }
        loss_vec, items = loss_fn(output, batch_dict)
        total_loss += items.sum().item()
        loss_box += items[0].item()
        loss_cls += items[1].item()
        loss_dfl += items[2].item()
        n_batches += 1

        # --- Collect predictions & GT for mAP ---
        for b in range(bs):
            pred = decoded[b]                   # [4+nc, N]
            boxes = pred[:4].T                  # [N, 4] xywh pixels
            cls_scores = pred[4:].T             # [N, nc]
            max_sc, max_cl = cls_scores.max(1)

            mask = max_sc > conf_thresh
            if mask.sum() > 0:
                bx = xywh2xyxy(boxes[mask])
                sc = max_sc[mask]
                cl = max_cl[mask]
                keep = batched_nms(bx, sc, cl, nms_iou)[:300]
                all_preds.append((bx[keep].cpu(), sc[keep].cpu(), cl[keep].cpu()))
            else:
                z4 = torch.zeros(0, 4)
                z0 = torch.zeros(0)
                all_preds.append((z4, z0, z0.long()))

            gt_mask = targets[:, 0] == b
            gt_bboxes = targets[gt_mask, 2:6].cpu()
            gt_cls = targets[gt_mask, 1].long().cpu()
            if len(gt_bboxes) > 0:
                gt_px = gt_bboxes.clone()
                gt_px[:, [0, 2]] *= images.shape[3]
                gt_px[:, [1, 3]] *= images.shape[2]
                all_targets.append((xywh2xyxy(gt_px), gt_cls))
            else:
                all_targets.append((torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)))

    # --- mAP per class ---
    aps = []
    for c in range(nc):
        scores_list, matched_list = [], []
        n_gt = 0
        for (pb, ps, pc), (gb, gc) in zip(all_preds, all_targets):
            gt_c = gb[gc == c]
            n_gt += len(gt_c)
            pmask = pc == c
            pb_c, ps_c = pb[pmask], ps[pmask]
            if len(pb_c) == 0:
                continue
            gt_matched = torch.zeros(len(gt_c), dtype=torch.bool)
            for idx in ps_c.argsort(descending=True):
                scores_list.append(ps_c[idx].item())
                if len(gt_c) == 0:
                    matched_list.append(False)
                    continue
                ious = box_iou(pb_c[idx:idx + 1], gt_c)[0]
                best_val, best_idx = ious.max(0)
                if best_val >= match_iou and not gt_matched[best_idx]:
                    matched_list.append(True)
                    gt_matched[best_idx] = True
                else:
                    matched_list.append(False)

        if n_gt == 0 or len(scores_list) == 0:
            aps.append(0.0)
            continue
        order = np.argsort(-np.array(scores_list))
        matched = np.array(matched_list)[order]
        tp = np.cumsum(matched)
        fp = np.cumsum(~matched)
        aps.append(_compute_ap(tp / n_gt, tp / (tp + fp)))

    map50 = float(np.mean(aps)) if aps else 0.0
    n = max(n_batches, 1)
    return {
        'val_loss': total_loss / n,
        'val_box': loss_box / n,
        'val_cls': loss_cls / n,
        'val_dfl': loss_dfl / n,
        'mAP50': map50,
        'AP50': aps,
    }


# ============================================================================
# Real Image Evaluation (no GT labels — detection stats + TensorBoard images)
# ============================================================================

@torch.no_grad()
def evaluate_real_images(model, real_images_dir, device, writer, epoch,
                         img_size=640, nc=2, conf_thresh=0.25, nms_iou=0.45,
                         max_vis=16):
    """Run inference on real screenshots and log stats + sample images to TensorBoard.

    Since there are no GT labels, we track:
    - total detections, per-class counts
    - mean/max confidence
    - sample images with drawn boxes
    """
    model.eval()
    img_dir = Path(real_images_dir)
    images = sorted(img_dir.glob('*.png')) + sorted(img_dir.glob('*.jpg'))
    if not images:
        return {}

    CLASS_NAMES = {0: 'body', 1: 'head'}
    CLASS_COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}

    total_det = 0
    body_count = 0
    head_count = 0
    all_confs = []
    images_with_det = 0
    vis_images = []

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        orig = img.copy()

        # Preprocess: direct resize to 640x640 (matches training)
        resized = cv2.resize(img, (img_size, img_size))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        input_batch = tensor.unsqueeze(0).to(device)

        output = model(input_batch)
        decoded = output[0]  # [1, 4+nc, N]
        pred = decoded[0]    # [4+nc, N]

        boxes_xywh = pred[:4].T   # [N, 4]
        cls_scores = pred[4:].T   # [N, nc]
        max_sc, max_cl = cls_scores.max(1)

        mask = max_sc > conf_thresh
        if mask.sum() > 0:
            bx = xywh2xyxy(boxes_xywh[mask])
            sc = max_sc[mask]
            cl = max_cl[mask]
            keep = batched_nms(bx, sc, cl, nms_iou)[:100]
            bx, sc, cl = bx[keep], sc[keep], cl[keep]

            n_det = len(bx)
            total_det += n_det
            body_count += (cl == 0).sum().item()
            head_count += (cl == 1).sum().item()
            all_confs.extend(sc.cpu().tolist())
            if n_det > 0:
                images_with_det += 1

            # Draw on 640x640 canvas for TensorBoard
            if len(vis_images) < max_vis:
                canvas = resized.copy()
                for i in range(len(bx)):
                    x1, y1, x2, y2 = bx[i].int().cpu().tolist()
                    c = int(cl[i].item())
                    conf = sc[i].item()
                    color = CLASS_COLORS.get(c, (255, 255, 0))
                    label = f"{CLASS_NAMES.get(c, str(c))} {conf:.2f}"
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(canvas, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
                    cv2.putText(canvas, label, (x1 + 1, y1 - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                # BGR -> RGB for TensorBoard
                canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                vis_images.append(canvas_rgb)
        else:
            # No detections — still add to vis if needed
            if len(vis_images) < max_vis:
                canvas_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                vis_images.append(canvas_rgb)

    n_images = len(images)
    mean_conf = float(np.mean(all_confs)) if all_confs else 0.0
    max_conf = float(np.max(all_confs)) if all_confs else 0.0

    # Log scalars to TensorBoard
    writer.add_scalar('real/total_detections', total_det, epoch)
    writer.add_scalar('real/body_count', body_count, epoch)
    writer.add_scalar('real/head_count', head_count, epoch)
    writer.add_scalar('real/det_per_image', total_det / max(n_images, 1), epoch)
    writer.add_scalar('real/images_with_det_pct', images_with_det / max(n_images, 1) * 100, epoch)
    writer.add_scalar('real/mean_conf', mean_conf, epoch)
    writer.add_scalar('real/max_conf', max_conf, epoch)

    # Log sample images as grid to TensorBoard
    if vis_images:
        # Stack into [N, 3, H, W] tensor
        vis_tensor = torch.stack([
            torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            for img in vis_images
        ])
        writer.add_images('real/detections', vis_tensor, epoch)

    result = {
        'real_total': total_det,
        'real_body': body_count,
        'real_head': head_count,
        'real_det_per_img': total_det / max(n_images, 1),
        'real_det_pct': images_with_det / max(n_images, 1) * 100,
        'real_mean_conf': mean_conf,
    }

    print(f'  Real   {n_images} imgs: {total_det} det ({body_count} body, {head_count} head), '
          f'{images_with_det}/{n_images} with det, '
          f'mean_conf={mean_conf:.3f}, max_conf={max_conf:.3f}')

    return result


# ============================================================================
# Training Loop
# ============================================================================

def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device,
                    ema, epoch, warmup_steps, base_lr, steps_done):
    model.train()
    total_loss, lb, lc, ld = 0., 0., 0., 0.
    n = 0

    pbar = tqdm(loader, desc=f'  Train ep {epoch}')
    for step, batch in enumerate(pbar):
        global_step = steps_done + step

        # Linear warmup
        if global_step < warmup_steps:
            lr = base_lr * (global_step + 1) / max(warmup_steps, 1)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        images = batch['images'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)
        bs = images.shape[0]

        batch_dict = {
            'batch_idx': targets[:, 0],
            'cls': targets[:, 1],
            'bboxes': targets[:, 2:6],
        }

        with autocast('cuda', enabled=scaler.is_enabled()):
            preds = model(images)
            loss_vec, items = loss_fn(preds, batch_dict)  # loss_vec = [box, cls, dfl] * batch_size
            loss = loss_vec.sum()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if ema is not None:
            ema.update(model)

        total_loss += items.sum().item()
        lb += items[0].item()
        lc += items[1].item()
        ld += items[2].item()
        n += 1

        pbar.set_postfix(
            loss=f'{items.sum().item():.3f}',
            box=f'{items[0]:.3f}', cls=f'{items[1]:.3f}', dfl=f'{items[2]:.3f}',
            lr=f'{optimizer.param_groups[0]["lr"]:.1e}',
        )

    return {
        'train_loss': total_loss / max(n, 1),
        'train_box': lb / max(n, 1),
        'train_cls': lc / max(n, 1),
        'train_dfl': ld / max(n, 1),
    }, steps_done + len(loader)


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(description='YOLO11n Training — CS2 Body/Head Detection')

    # Data
    p.add_argument('--backgrounds', required=True, help='Directory with empty map screenshots')
    p.add_argument('--rgba-crops', required=True, help='Directory with RGBA crops + metadata.json')

    # Model
    p.add_argument('--yaml', required=True, help='Path to yolo11n.yaml')
    p.add_argument('--pretrained', default=None, help='Pretrained weights (yolo11n.pt)')
    p.add_argument('--resume', default=None, help='Resume training from checkpoint')
    p.add_argument('--nc', type=int, default=2, help='Number of classes')

    # Training hyperparameters
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3, help='Base learning rate (AdamW)')
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--epoch-size', type=int, default=0,
                   help='Virtual epoch size, 0 = number of backgrounds')

    # Synthetic dataset params
    p.add_argument('--min-players', type=int, default=1)
    p.add_argument('--max-players', type=int, default=5)
    p.add_argument('--negative-ratio', type=float, default=0.15)
    p.add_argument('--flash-prob', type=float, default=0.15)
    p.add_argument('--occlusion-prob', type=float, default=0.0)
    p.add_argument('--scale-min', type=float, default=0.5)
    p.add_argument('--scale-max', type=float, default=2.5)
    p.add_argument('--demo-positions', default=None,
                   help='Path to positions.npz from extract_demo_positions.py')
    p.add_argument('--demo-position-prob', type=float, default=0.7,
                   help='Probability of using demo position vs random (0.0-1.0)')

    # Loss weights
    p.add_argument('--box-weight', type=float, default=7.5)
    p.add_argument('--cls-weight', type=float, default=0.5)
    p.add_argument('--dfl-weight', type=float, default=1.5)

    # Evaluation
    p.add_argument('--val-interval', type=int, default=5,
                   help='Compute mAP every N epochs')
    p.add_argument('--val-ratio', type=float, default=0.2,
                   help='Fraction of backgrounds held out for validation')
    p.add_argument('--real-images', default=None,
                   help='Directory with real screenshots for validation (no GT labels)')

    # Output
    p.add_argument('--output', default='./runs', help='Root output directory')
    p.add_argument('--name', default='yolo11n_body_head', help='Run name')

    # Misc
    p.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    p.add_argument('--seed', type=int, default=42)

    args = p.parse_args()
    use_amp = not args.no_amp

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = Path(args.output) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(run_dir / 'tb_logs'))

    CLASS_NAMES = {0: 'body', 1: 'head'}

    print('=' * 60)
    print('YOLO11n Training — CS2 Body/Head Detection')
    print('=' * 60)
    print(f'  Device:       {device}')
    print(f'  Classes:      {args.nc} ({", ".join(CLASS_NAMES.values())})')
    print(f'  Epochs:       {args.epochs}')
    print(f'  Batch size:   {args.batch_size}')
    print(f'  LR:           {args.lr}')
    print(f'  Image size:   {args.img_size}')
    print(f'  AMP:          {use_amp}')
    print(f'  Warmup:       {args.warmup_epochs} epochs')
    print(f'  Loss weights: box={args.box_weight} cls={args.cls_weight} dfl={args.dfl_weight}')
    print(f'  Output:       {run_dir}')
    print('=' * 60)

    # ==== Datasets (separate instances for proper train/val split) ====
    print('\nLoading datasets...')
    train_ds = SyntheticYOLODataset(
        backgrounds_dir=args.backgrounds,
        rgba_crops_dir=args.rgba_crops,
        img_size=args.img_size,
        augment=True,
        min_players=args.min_players,
        max_players=args.max_players,
        negative_ratio=args.negative_ratio,
        flash_prob=args.flash_prob,
        occlusion_prob=args.occlusion_prob,
        scale_range=(args.scale_min, args.scale_max),
        demo_positions_path=args.demo_positions,
        demo_position_prob=args.demo_position_prob,
    )
    val_ds = SyntheticYOLODataset(
        backgrounds_dir=args.backgrounds,
        rgba_crops_dir=args.rgba_crops,
        img_size=args.img_size,
        augment=False,
        min_players=args.min_players,
        max_players=args.max_players,
        negative_ratio=0.0,
        scale_range=(args.scale_min, args.scale_max),
        demo_positions_path=args.demo_positions,
        demo_position_prob=args.demo_position_prob,
    )

    # Split backgrounds 80/20
    all_bgs = train_ds.background_paths[:]
    random.shuffle(all_bgs)
    split = int((1 - args.val_ratio) * len(all_bgs))
    train_ds.background_paths = all_bgs[:split]
    val_ds.background_paths = all_bgs[split:]

    print(f'  Train backgrounds: {len(train_ds)}')
    print(f'  Val backgrounds:   {len(val_ds)}')

    # DataLoaders
    epoch_size = args.epoch_size if args.epoch_size > 0 else len(train_ds)
    if args.epoch_size > 0:
        sampler = RandomSampler(train_ds, replacement=True, num_samples=epoch_size)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  collate_fn=collate_fn, num_workers=args.workers,
                                  pin_memory=True, persistent_workers=args.workers > 0)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_fn, num_workers=args.workers,
                                  pin_memory=True, persistent_workers=args.workers > 0)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0)

    steps_per_epoch = len(train_loader)
    print(f'  Epoch size:        {epoch_size} ({steps_per_epoch} steps)')

    # ==== Model ====
    print('\nBuilding YOLO11n...')
    model = YOLO11n(cfg=args.yaml, nc=args.nc).to(device)

    if args.pretrained and Path(args.pretrained).exists():
        load_pretrained_weights(model, args.pretrained)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {n_params:,}')

    # ==== Loss ====
    from types import SimpleNamespace
    hyp = SimpleNamespace(
        tal_topk=10,
        box=args.box_weight,
        cls=args.cls_weight,
        dfl=args.dfl_weight,
    )
    loss_fn = v8DetectionLoss(_LossModelAdapter(model, hyp))

    # ==== Optimizer + Scheduler ====
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.lr * 0.01
    )

    scaler = GradScaler('cuda', enabled=use_amp)
    ema = ModelEMA(model, decay=0.9999)

    # ==== Resume ====
    start_epoch = 1
    best_map = 0.0
    steps_done = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'ema_state_dict' in ckpt:
            ema.ema.load_state_dict(ckpt['ema_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_map = ckpt.get('best_map', 0.0)
        steps_done = (start_epoch - 1) * steps_per_epoch
        print(f'  Resumed from epoch {start_epoch - 1}, best mAP={best_map:.4f}')

    warmup_steps = args.warmup_epochs * steps_per_epoch

    # ==== Training ====
    print(f'\nStarting training from epoch {start_epoch}...\n')

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Train
        train_m, steps_done = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device,
            ema, epoch, warmup_steps, args.lr, steps_done,
        )

        # LR scheduler (after warmup)
        if epoch >= args.warmup_epochs:
            scheduler.step()

        # Evaluate (loss + mAP) — fixed seed for stable validation
        do_map = (epoch % args.val_interval == 0) or (epoch == args.epochs)
        rng_state = random.getstate()
        np_rng_state = np.random.get_state()
        random.seed(args.seed)
        np.random.seed(args.seed)
        val_m = evaluate(ema.ema, val_loader, loss_fn, device, nc=args.nc)
        random.setstate(rng_state)
        np.random.set_state(np_rng_state)

        # Evaluate on real screenshots (no GT — detection stats only)
        if args.real_images and do_map:
            evaluate_real_images(ema.ema, args.real_images, device, writer, epoch,
                                img_size=args.img_size, nc=args.nc)

        dt = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        # TensorBoard
        writer.add_scalar('train/loss', train_m['train_loss'], epoch)
        writer.add_scalar('train/box_loss', train_m['train_box'], epoch)
        writer.add_scalar('train/cls_loss', train_m['train_cls'], epoch)
        writer.add_scalar('train/dfl_loss', train_m['train_dfl'], epoch)
        writer.add_scalar('val/loss', val_m['val_loss'], epoch)
        writer.add_scalar('val/box_loss', val_m['val_box'], epoch)
        writer.add_scalar('val/cls_loss', val_m['val_cls'], epoch)
        writer.add_scalar('val/dfl_loss', val_m['val_dfl'], epoch)
        writer.add_scalar('val/mAP50', val_m['mAP50'], epoch)
        for ci, ap in enumerate(val_m['AP50']):
            writer.add_scalar(f'val/AP50_{CLASS_NAMES.get(ci, ci)}', ap, epoch)
        writer.add_scalar('lr', lr, epoch)

        # Print
        print(f'\nEpoch {epoch}/{args.epochs} ({dt:.0f}s)  LR={lr:.1e}')
        print(f'  Train  loss={train_m["train_loss"]:.4f}  '
              f'box={train_m["train_box"]:.4f}  cls={train_m["train_cls"]:.4f}  '
              f'dfl={train_m["train_dfl"]:.4f}')
        print(f'  Val    loss={val_m["val_loss"]:.4f}  '
              f'box={val_m["val_box"]:.4f}  cls={val_m["val_cls"]:.4f}  '
              f'dfl={val_m["val_dfl"]:.4f}')
        ap_str = '  '.join(f'{CLASS_NAMES.get(i, i)}={ap:.4f}' for i, ap in enumerate(val_m['AP50']))
        print(f'  mAP@50={val_m["mAP50"]:.4f}  ({ap_str})')

        # Save
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema.ema.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_map': best_map,
            'mAP50': val_m['mAP50'],
            'args': vars(args),
        }

        if val_m['mAP50'] > best_map:
            best_map = val_m['mAP50']
            save_dict['best_map'] = best_map
            torch.save(save_dict, run_dir / 'best.pt')
            print(f'  >> New best mAP@50: {best_map:.4f}')

        torch.save(save_dict, run_dir / 'last.pt')

        if epoch % 10 == 0:
            torch.save(save_dict, run_dir / f'epoch{epoch}.pt')

    writer.close()
    print(f'\nTraining complete. Best mAP@50={best_map:.4f}')
    print(f'Checkpoints: {run_dir}')


if __name__ == '__main__':
    main()
