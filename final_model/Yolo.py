"""
YOLOv11l DetectionModel with Scene Embedding Branch.

Architecture ported from detect/train_yolo.py (YOLO11 class) with:
- Dynamic forward pass using y[] cache (identical to official ultralytics)
- Correct channel calculation: make_divisible(min(c, max_ch) * gw, 8)
- l-scale: gd=1.0, gw=1.0, max_ch=512
- Embed branch: P3 (layer 16, 256ch, 80×80) + P5 skip (layer 10, 512ch, 20×20)
- UNet-like embed head: P3 processed down to 20×20, skip-add P5 → InvRes → pool → (B,512)
- extract_features(): stops at layer 16 — returns (p3, p5) for caching
- embed_from_features(p3, p5): runs trainable embed head → (B, 512)
- forward(): full pass → (detections, scene_embed)

Backbone caching pattern (Train.py):
    with torch.no_grad():
        p3, p5 = yolo.extract_features(img_tensor)       # frozen backbone
    scene_embed = yolo.embed_from_features(p3, p5)        # trainable, with grad
"""

import math
import yaml
import copy
import torch
import torch.nn as nn
import torchvision.ops as ops
from collections import OrderedDict
from typing import Optional


# ============================================================================
# Helpers
# ============================================================================

def make_divisible(x, divisor=8):
    return math.ceil(x / divisor) * divisor


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


# ============================================================================
# YOLO11 Building Blocks (identical to detect/train_yolo.py)
# ============================================================================

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
        self.scale = self.key_dim ** -0.5
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
        return self.proj(x)


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

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80) -> torch.Tensor:
        """Post-process YOLO predictions → (B, max_det, 6) [cx,cy,w,h,conf,cls]."""
        preds = preds.permute(0, 2, 1)
        batch_size, anchors, _ = preds.shape
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


# ============================================================================
# Embed branch helpers (MobileNet-style blocks for P3 processing)
# ============================================================================

class InvertedResidual(nn.Module):
    """MobileNetV2-style: expand → depthwise → project."""
    def __init__(self, inp, oup, stride=1, expand_ratio=2):
        super().__init__()
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim), nn.SiLU(inplace=True)
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False), nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


class FusedMBConv(nn.Module):
    """EfficientNetV2-style fused conv."""
    def __init__(self, inp, oup, stride=1, expand_ratio=2):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False), nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


# ============================================================================
# Main DetectionModel
# ============================================================================

class DetectionModel(nn.Module):
    """
    YOLOv11 + Scene Embedding Branch.

    Compatible with weights from detect/train_yolo.py (YOLO11 class).
    Uses dynamic forward pass (y=[] cache) identical to ultralytics.

    Args:
        cfg: path to yolo11.yaml (official format with 'scales' dict)
        ch: input channels (3 for RGB)
        nc: number of detection classes
        scale: 'n'/'s'/'m'/'l'/'x' (default 'l' for 25M backbone)

    Embed branch:
        Input: P3 head features (layer 16), 256ch @ 80x80 for l-scale
        Output: (B, 512) scene embedding via pool → MBConv → flatten
    """

    EMBED_LAYER_IDX  = 16   # P3 head: C3k2([256, False]) in yolo11.yaml
    P5_BACKBONE_IDX  = 10   # Backbone P5: C2PSA([1024]) — computed on the way to P3

    def __init__(self, cfg='yolo11.yaml', ch=3, nc=2, scale='l'):
        super().__init__()
        with open(cfg, 'r', encoding='utf-8') as f:
            self.yaml = yaml.safe_load(f)

        # Apply scale constants
        if 'scales' in self.yaml and scale in self.yaml['scales']:
            gd, gw, max_ch = self.yaml['scales'][scale]
            self.yaml['depth_mult'] = gd
            self.yaml['width_mult'] = gw
            self.yaml['max_channels'] = max_ch
        self.scale = scale

        self.model, self.save = self._parse_model(self.yaml, ch=[ch], nc=nc)

        gw = self.yaml.get('width_mult', 1.0)
        max_ch = self.yaml.get('max_channels', 512)
        # P3 head (layer 16): C3k2([256]) → 256ch for l-scale
        p3_ch = make_divisible(min(256, max_ch) * gw, 8)
        # Backbone P5 (layer 10): C2PSA([1024]) → min(1024,512)*1.0 = 512ch for l-scale
        p5_ch = make_divisible(min(1024, max_ch) * gw, 8)

        # Multi-scale embed head: P3 (80×80) + P5 (20×20) → 512
        #
        # UNet-like pipeline:
        #   P3 (p3_ch, 80×80)
        #     → embeds_stem  (p3_ch→128, 1×1)           # channel reduction
        #     → embeds_p3_down1  InvRes(128→128, s=2)   # 80→40  (DWS-like, cheap)
        #     → embeds_p3_down2  FusedMBConv(128→256, s=2)  # 40→20
        #     = (256, 20×20)
        #                ↕  add  ← UNet-like skip
        #   P5 (p5_ch, 20×20)
        #     → embeds_p5_proj  (p5_ch→256, 1×1)        # match channels
        #     = (256, 20×20)
        #
        #   concat → (512, 20×20)  ← UNet-like skip concat
        #
        #   → embeds_post  InvRes(512→512, s=1)          # main capacity
        #   → AvgPool(1×1) → Flatten → LayerNorm(512)
        #
        # Parameters: ~33K + ~69K + ~181K + ~132K + ~1063K = ~1.48M

        # P3 branch
        self.embeds_stem = nn.Sequential(
            nn.Conv2d(p3_ch, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
        )
        self.embeds_p3_down1 = InvertedResidual(128, 128, stride=2, expand_ratio=2)   # 80→40
        self.embeds_p3_down2 = FusedMBConv(128, 256, stride=2, expand_ratio=1)         # 40→20

        # P5 skip projection
        self.embeds_p5_proj = nn.Sequential(
            nn.Conv2d(p5_ch, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )

        # Post-skip processing + pool (input: 256+256=512 after concat)
        self.embeds_post = nn.Sequential(
            InvertedResidual(512, 512, stride=1, expand_ratio=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(512),
        )

    def _parse_model(self, d, ch, nc):
        """Parse YAML model definition. Identical to detect/train_yolo.py."""
        layers, save = [], []
        all_layers = []
        if 'backbone' in d:
            all_layers += d['backbone']
        if 'head' in d:
            all_layers += d['head']

        gd = d.get('depth_mult', 0.5)
        gw = d.get('width_mult', 0.25)
        max_ch = d.get('max_channels', 1024)

        for i, (f, n, m, args) in enumerate(all_layers):
            m = eval(m) if isinstance(m, str) else m
            args = [x for x in args]
            n_ = n = max(round(n * gd), 1) if n > 1 else n

            if isinstance(f, int):
                c1 = ch[f] if f != -1 else ch[-1]
            elif isinstance(f, list):
                c1 = sum(ch[j] for j in f)
            else:
                c1 = ch[-1]

            if m.__name__ in ("Conv", "C2f", "C3", "C3k2", "SPPF", "C2PSA"):
                c2 = make_divisible(min(args[0], max_ch) * gw, 8)
                args = [c1, c2, *args[1:]]
                if m.__name__ in ("C3k2", "C2PSA", "C2f", "C3"):
                    args.insert(2, n)
                    n = n_ = 1
                if m.__name__ == "C3k2" and self.scale in "mlx":
                    args[3] = True
            elif m.__name__ == "Upsample":
                c2 = ch[f]
                scale = args[1] if len(args) > 1 and args[1] is not None else 2
                mode = args[2] if len(args) > 2 else "nearest"
                args = {"scale_factor": scale, "mode": mode}
            elif m.__name__ in ("Concat",):
                c2 = sum(ch[j] for j in f)
                args = [1]
            elif m.__name__ in ("Detect",):
                args = [nc, [ch[x] for x in f]]
                c2 = None
            else:
                c2 = c1

            if n_ > 1 and m.__name__ not in ("Detect", "Concat"):
                block = nn.Sequential(*[m(*args) for _ in range(n_)])
            elif m.__name__ == "Upsample":
                block = m(**args)
            else:
                block = m(*args)

            block.f = f
            block.i = i
            layers.append(block)

            if i == 0:
                ch = []
            ch.append(c2 if c2 is not None else ch[-1])

            if isinstance(f, list) and len(f) > 1:
                save.extend(x % i for x in f if x != -1)

        return nn.ModuleList(layers), sorted(save)

    def forward(self, x):
        """
        Full forward pass.

        Returns:
            detections: raw YOLO output (training) or (decoded, raw_feats) (eval)
            embed: (B, 512) scene embedding from P3 head
        """
        y = []
        p5_backbone = None
        embed = None
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            if m.i == self.P5_BACKBONE_IDX:
                p5_backbone = x
            if m.i == self.EMBED_LAYER_IDX:
                embed = self.embed_from_features(x, p5_backbone)  # (B, 512)
            y.append(x if m.i in self.save else None)
        return x, embed

    @torch.no_grad()
    def extract_features(self, x):
        """
        Extract P3 (layer 16) and backbone P5 (layer 10) features.

        Stops forward at P3 — skips P4/P5 detection head.
        P5 backbone is captured for free on the way to P3.
        Call under torch.no_grad() when backbone is frozen.

        Returns:
            p3: (B, p3_ch, 80, 80)  — fine detail, small players
            p5: (B, p5_ch, 20, 20)  — global context, scene layout
        """
        y = []
        p5_backbone = None
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            if m.i == self.P5_BACKBONE_IDX:
                p5_backbone = x   # captured; x gets reassigned at next iter
            if m.i == self.EMBED_LAYER_IDX:
                return x, p5_backbone   # (p3, p5)
            y.append(x if m.i in self.save else None)
        return x, p5_backbone  # fallback

    def embed_from_features(self, p3, p5):
        """
        Apply trainable multi-scale embed head to cached P3 + P5 features.

        Args:
            p3: (B, p3_ch, 80, 80) from extract_features()
            p5: (B, p5_ch, 20, 20) from extract_features()

        Returns:
            embed: (B, 512)
        """
        x = self.embeds_stem(p3)            # (B, 128, 80, 80)
        x = self.embeds_p3_down1(x)         # (B, 128, 40, 40)
        x = self.embeds_p3_down2(x)         # (B, 256, 20, 20)
        p5_feat = self.embeds_p5_proj(p5)   # (B, 256, 20, 20)
        x = torch.cat([x, p5_feat], dim=1)  # (B, 512, 20, 20) — UNet skip concat
        return self.embeds_post(x)          # (B, 512)

    @torch.no_grad()
    def forward_detections_only(self, x):
        """Full forward for detections only (no embed head). No grad."""
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x


# ============================================================================
# LRU Feature Cache for backbone caching
# ============================================================================

class FeatureCache:
    """
    LRU cache for frozen YOLO P3 features and detection vectors.
    Keys are frame file paths (strings).
    Stores tensors as float16 on CPU to reduce memory.
    """

    def __init__(self, max_size: int = 2000):
        self._cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, key: str, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            if isinstance(value, tuple):
                self._cache[key] = tuple(v.detach().half().cpu() for v in value)
            else:
                self._cache[key] = value.detach().half().cpu()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __len__(self):
        return len(self._cache)


# ============================================================================
# Weight Loading
# ============================================================================

def load_pretrained_weights(model: nn.Module, path: str) -> nn.Module:
    """
    Load pretrained weights with shape matching (handles nc/scale mismatch).
    Skips embeds head layers (they train from scratch).
    """
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
# Quick sanity test
# ============================================================================

if __name__ == "__main__":
    import os
    yaml_path = os.path.join(os.path.dirname(__file__), "yolo11.yaml")
    print(f"Testing DetectionModel with scale='l', yaml={yaml_path}")

    model = DetectionModel(cfg=yaml_path, nc=2, scale='l')
    total = sum(p.numel() for p in model.parameters())
    embed_params = sum(
        p.numel() for name, p in model.named_parameters()
        if name.startswith('embeds')
    )
    backbone_params = total - embed_params
    print(f"  Total params:    {total:,}")
    print(f"  Backbone params: {backbone_params:,}")
    print(f"  Embed params:    {embed_params:,}")

    # Test extract_features + embed_from_features
    x = torch.randn(2, 3, 640, 640)
    p3, p5 = model.extract_features(x)
    print(f"  P3 features: {p3.shape}")   # (2, 256, 80, 80) for l-scale
    print(f"  P5 features: {p5.shape}")   # (2, 512, 20, 20) for l-scale

    embed = model.embed_from_features(p3, p5)
    print(f"  Embed:       {embed.shape}")  # (2, 512)

    # Test full forward (eval mode)
    model.eval()
    with torch.no_grad():
        dets, embed2 = model(x)
    if isinstance(dets, tuple):
        print(f"  Detections (decoded): {dets[0].shape}")  # (2, 4+nc, N)
    print(f"  Embed (full fwd): {embed2.shape}")  # (2, 512)

    print("All checks passed!")
