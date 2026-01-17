import torch
import torchvision.ops as ops
import math
import torch.nn as nn
import time
import copy
import yaml
import cv2
import numpy as np
from ultralytics import YOLO


# def autopad(k, p=None, d=1):
#     if d > 1:
#         # actual kernel-size
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
#     if p is None:
#         # auto-pad
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
#     return p

# def make_anchors(x, strides, offset=0.5):
#     # assert x is not None
#     if x[0].numel() >= 0: #TODO заглушка для валидации
#         dummy = torch.zeros((1, 2)).to(x[0].device)
#         stride_dummy = torch.zeros((1, 1)).to(x[0].device)
#         return dummy, stride_dummy
#     anchor_tensor, stride_tensor = [], []
#     dtype, device = x[0].dtype, x[0].device
#     for i, stride in enumerate(strides):
#         _, _, h, w = x[i].shape
#         sx = torch.arange(end=w, device=device, dtype=dtype) + offset  # shift x
#         sy = torch.arange(end=h, device=device, dtype=dtype) + offset  # shift y
#         sy, sx = torch.meshgrid(sy, sx)
#         anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
#         stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
#     return torch.cat(anchor_tensor), torch.cat(stride_tensor)

# def fuse_conv(conv, norm):
#     fused_conv = nn.Conv2d(
#         conv.in_channels,
#         conv.out_channels,
#         kernel_size=conv.kernel_size,
#         stride=conv.stride,
#         padding=conv.padding,
#         groups=conv.groups,
#         bias=True).requires_grad_(False).to(conv.weight.device
#     )

#     w_conv = conv.weight.clone().view(conv.out_channels, -1)
#     w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
#     fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

#     b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
#     b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
#     fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

#     return fused_conv


# # This is the activation function used in YOLOv11
# class SiLU(nn.Module):
#     @staticmethod
#     def forward(x):
#         return x * torch.sigmoid(x)

# class Conv(nn.Module):

#     def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0, g=1):
#         # in_ch = input channels
#         # out_ch = output channels
#         # activation = the torch function of the activation function (SiLU or Identity)
#         # k = kernel size
#         # s = stride
#         # p = padding
#         # g = groups
#         super().__init__()
#         self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
#         self.norm = nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
#         self.relu = activation

#     def forward(self, x):
#         # Passing the input by convolution layer and using the activation function
#         # on the normalized output
#         x = self.conv(x)
#         x = self.norm(x)
#         return self.relu(x)
        
#     def fuse_forward(self, x):
#         x = self.conv(x)
#         return self.relu(x)

# class Residual(nn.Module):
#     def __init__(self, ch, e=0.5):
#         super().__init__()
#         self.conv1 = Conv(ch, int(ch * e), nn.SiLU(), k=3, p=1)
#         self.conv2 = Conv(int(ch * e), ch, nn.SiLU(), k=3, p=1)

#     def forward(self, x):
#         # The input is passed through 2 Conv blocks and if the shortcut is true and
#         # if input and output channels are same, then it will the input as residual
#         return x + self.conv2(self.conv1(x))

# class C3K(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv1 = Conv(in_ch, out_ch // 2, nn.SiLU())
#         self.conv2 = Conv(in_ch, out_ch // 2, nn.SiLU())
#         self.conv3 = Conv(2 * (out_ch // 2), out_ch, nn.SiLU())
#         self.res_m = nn.Sequential(Residual(out_ch // 2, e=1.0),
#                                          Residual(out_ch // 2, e=1.0))

#     def forward(self, x):
#         y = self.res_m(self.conv1(x)) # Process half of the input channels
#         # Process the other half directly, Concatenate along the channel dimension
#         return self.conv3(torch.cat((y, self.conv2(x)), dim=1))

# class C3K2(nn.Module):
#     def __init__(self, in_ch, out_ch, n, csp, r):
#         super().__init__()
#         self.conv1 = Conv(in_ch, 2 * (out_ch // r), nn.SiLU())
#         self.conv2 = Conv((2 + n) * (out_ch // r), out_ch, nn.SiLU())

#         if not csp:
#             # Using the CSP Module when mentioned True at shortcut
#             self.res_m = nn.ModuleList(Residual(out_ch // r) for _ in range(n))
#         else:
#             # Using the Bottlenecks when mentioned False at shortcut
#             self.res_m = nn.ModuleList(C3K(out_ch // r, out_ch // r) for _ in range(n))

#     def forward(self, x):
#         y = list(self.conv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.res_m)
#         return self.conv2(torch.cat(y, dim=1))

# class SPPF(nn.Module):
#     def __init__(self, c1, c2, k=5):
#         super().__init__()
#         c_ = c1 // 2
#         self.conv1 = Conv(c1, c_, nn.SiLU(), 1, 1)
#         self.conv2 = Conv(c_ * 4, c2, nn.SiLU(), 1, 1)
#         self.maxpool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

#     def forward(self, x):
#         x = self.conv1(x)
#         y1 = self.maxpool(x)
#         y2 = self.maxpool(y1) 
#         return self.conv2(torch.cat((x, y1, y2, self.maxpool(y2)), 1)) # Ending with Conv Block

# class Attention(nn.Module):
#     def __init__(self, ch, num_head):
#         super().__init__()
#         self.num_head = num_head
#         self.dim_head = ch // num_head
#         self.dim_key = self.dim_head // 2
#         self.scale = self.dim_key ** -0.5

#         self.qkv = Conv(ch, ch + self.dim_key * num_head * 2, nn.Identity())

#         self.conv1 = Conv(ch, ch, nn.Identity(), k=3, p=1, g=ch)
#         self.conv2 = Conv(ch, ch, nn.Identity())

#     def forward(self, x):
#         b, c, h, w = x.shape

#         qkv = self.qkv(x)
#         qkv = qkv.view(b, self.num_head, self.dim_key * 2 + self.dim_head, h * w)

#         q, k, v = qkv.split([self.dim_key, self.dim_key, self.dim_head], dim=2)

#         attn = (q.transpose(-2, -1) @ k) * self.scale
#         attn = attn.softmax(dim=-1)

#         x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.conv1(v.reshape(b, c, h, w))
#         return self.conv2(x)

# class PSABlock(nn.Module):
#     # This Module has a sequential of one Attention module and 2 Conv Blocks
#     def __init__(self, ch, num_head):
#         super().__init__()
#         self.conv1 = Attention(ch, num_head)
#         self.conv2 = nn.Sequential(Conv(ch, ch * 2, nn.SiLU()),
#                                          Conv(ch * 2, ch, nn.Identity()))

#     def forward(self, x):
#         x = x + self.conv1(x)
#         return x + self.conv2(x)

# class PSA(nn.Module):
#     def __init__(self, ch, n):
#         super().__init__()
#         self.conv1 = Conv(ch, 2 * (ch // 2), nn.SiLU())
#         self.conv2 = Conv(2 * (ch // 2), ch, nn.SiLU())
#         self.res_m = nn.Sequential(*(PSABlock(ch // 2, ch // 128) for _ in range(n)))

#     def forward(self, x):
#         # Passing the input to the Conv Block and splitting into two feature maps
#         x, y = self.conv1(x).chunk(2, 1)
#         # 'n' number of PSABlocks are made sequential, and then passes one them (y) of
#         # the feature maps and concatenate with the remaining feature map (x)
#         return self.conv2(torch.cat(tensors=(x, self.res_m(y)), dim=1))

# class DFL(nn.Module):
#     # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss 
#     def __init__(self, c1=16):
#         super().__init__()
#         self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
#         x = torch.arange(c1, dtype=torch.float)
#         self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
#         self.c1 = c1

#     def forward(self, x):
#         b, c, a = x.shape
#         return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

# class DarkNet(nn.Module):
#     def __init__(self, width, depth, csp):
#         super().__init__()
#         self.p1 = []
#         self.p2 = []
#         self.p3 = []
#         self.p4 = []
#         self.p5 = []

#         # p1/2
#         self.p1.append(Conv(width[0], width[1], nn.SiLU(), k=3, s=2, p=1))
#         # p2/4
#         self.p2.append(Conv(width[1], width[2], nn.SiLU(), k=3, s=2, p=1))
#         self.p2.append(C3K2(width[2], width[3], depth[0], csp[0], r=4))
#         # p3/8
#         self.p3.append(Conv(width[3], width[3], nn.SiLU(), k=3, s=2, p=1))
#         self.p3.append(C3K2(width[3], width[4], depth[1], csp[0], r=4))
#         # p4/16
#         self.p4.append(Conv(width[4], width[4], nn.SiLU(), k=3, s=2, p=1))
#         self.p4.append(C3K2(width[4], width[4], depth[2], csp[1], r=2))
#         # p5/32
#         self.p5.append(Conv(width[4], width[5], nn.SiLU(), k=3, s=2, p=1))
#         self.p5.append(C3K2(width[5], width[5], depth[3], csp[1], r=2))
#         self.p5.append(SPPF(width[5], width[5]))
#         self.p5.append(PSA(width[5], depth[4]))

#         self.p1 = nn.Sequential(*self.p1)
#         self.p2 = nn.Sequential(*self.p2)
#         self.p3 = nn.Sequential(*self.p3)
#         self.p4 = nn.Sequential(*self.p4)
#         self.p5 = nn.Sequential(*self.p5)

#         self.embeds = nn.Sequential(
#             nn.Conv2d(width[3], 32, 1, bias=False),
#             nn.Conv2d(32, 64, 3, padding=1, groups=32, bias=False),
#             nn.Conv2d(64, 128, 1, bias=False),
#             nn.AdaptiveAvgPool2d((4, 4)),
#             nn.Flatten(),
#             nn.LayerNorm(128 * 4 * 4)
#         )
#         # self.pos_embedding = nn.Parameter(torch.zeros(1, 512, 40, 40))
#         # self.proj = nn.Conv2d(width[4], 512, 1)
#         # self.transformer = nn.TransformerEncoder(
#         #     nn.TransformerEncoderLayer(512, 4, 512*4, batch_first=True),
#         #     num_layers=8
#         # )
#         # self.post_proj = nn.Sequential(
#         #     nn.Linear(512, 512),
#         #     nn.LayerNorm(512),
#         #     nn.SiLU()
#         # )

#     def forward(self, x, stride_head=False):
#         p1 = self.p1(x)
#         p2 = self.p2(p1)
#         # B, C, H, W = p2.shape
#         if stride_head == False:
#             p3 = self.p3(p2)
#             p4 = self.p4(p3)
#             # x = self.proj(p4).flatten(2).permute(0, 2, 1)
#             # pos_embed = self.pos_embedding.flatten(2).permute(0, 2, 1)
#             # x = x + pos_embed
#             # embed = self.transformer(x)
#             # embed = embed.mean(dim=1)
#             # embed = self.post_proj(embed)
#             embed = self.embeds(p2)
#             p5 = self.p5(p4)
#             return p3, p4, p5, embed
#         p3 = self.p3(p2)
#         p4 = self.p4(p3)
#         p5 = self.p5(p4)
#         return p3, p4, p5


# class DarkFPN(nn.Module):
#     def __init__(self, width, depth, csp):
#         super().__init__()
#         self.up = nn.Upsample(scale_factor=2)
#         self.h1 = C3K2(width[4] + width[5], width[4], depth[5], csp[0], r=2)
#         self.h2 = C3K2(width[4] + width[4], width[3], depth[5], csp[0], r=2)
#         self.h3 = Conv(width[3], width[3], nn.SiLU(), k=3, s=2, p=1)
#         self.h4 = C3K2(width[3] + width[4], width[4], depth[5], csp[0], r=2)
#         self.h5 = Conv(width[4], width[4], nn.SiLU(), k=3, s=2, p=1)
#         self.h6 = C3K2(width[4] + width[5], width[5], depth[5], csp[1], r=2)

#     def forward(self, x):
#         p3, p4, p5 = x
#         p4 = self.h1(torch.cat(tensors=[self.up(p5), p4], dim=1))
#         p3 = self.h2(torch.cat(tensors=[self.up(p4), p3], dim=1))
#         p4 = self.h4(torch.cat(tensors=[self.h3(p3), p4], dim=1))
#         p5 = self.h6(torch.cat(tensors=[self.h5(p4), p5], dim=1))
#         return p3, p4, p5

# class Head(nn.Module):
#     anchors = torch.empty(0)
#     strides = torch.empty(0)

#     def __init__(self, nc=80, filters=()):
#         super().__init__()
#         self.ch = 16  # DFL channels
#         self.nc = nc  # number of classes
#         self.nl = len(filters)  # number of detection layers
#         self.no = nc + self.ch * 4  # number of outputs per anchor
#         self.stride = torch.zeros(self.nl)  # strides computed during build

#         box = max(64, filters[0] // 4)
#         cls = max(80, filters[0], self.nc)

#         self.dfl = DFL(self.ch)
        
#         self.box = nn.ModuleList(
#            nn.Sequential(Conv(x, box, nn.SiLU(), k=3, p=1),
#            Conv(box, box, nn.SiLU(), k=3, p=1),
#            nn.Conv2d(box, out_channels=4 * self.ch,kernel_size=1)) for x in filters)
        
#         self.cls = nn.ModuleList(
#             nn.Sequential(Conv(x, x, nn.SiLU(), k=3, p=1, g=x),
#             Conv(x, cls, nn.SiLU()),
#             Conv(cls, cls, nn.SiLU(), k=3, p=1, g=cls),
#             Conv(cls, cls, nn.SiLU()),
#             nn.Conv2d(cls, out_channels=self.nc,kernel_size=1)) for x in filters)

#     def initialize_biases(self):
#         for module in self.cls:
#             conv = module[-1]  # последний Conv2d (классы)
#             b = conv.bias.view(self.nc)
#             # установить bias примерно так, чтобы вероятности были около 0.01
#             b.data.fill_(-math.log((1 - 0.01) / 0.01))
#             conv.bias = nn.Parameter(b)

#         for module in self.box:
#             conv = module[-1]  # последний Conv2d (bbox)
#             b = conv.bias.view(4 * self.ch)
#             b.data.zero_()  # начальные смещения bbox = 0
#             conv.bias = nn.Parameter(b)
    
#     def forward(self, x):
#         for i, (box, cls) in enumerate(zip(self.box, self.cls)):
#             x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1)
#         if self.training:
#             return x

#         self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
#         x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
#         box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

#         a, b = self.dfl(box).chunk(2, 1)
#         a = self.anchors.unsqueeze(0) - a
#         b = self.anchors.unsqueeze(0) + b
#         box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

#         return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)

# class YOLO(nn.Module):
#     def __init__(self, width, depth, csp, num_classes):
#         super().__init__()
#         self.net = DarkNet(width, depth, csp)
#         self.fpn = DarkFPN(width, depth, csp)

#         img_dummy = torch.zeros(1, width[0], 256, 256)
#         self.head = Head(num_classes, (width[3], width[4], width[5]))
#         # self.embedding_stack = []
#         self.head.stride = torch.tensor([256 / x[0].shape[-2] for x in self.forward(img_dummy, True) if len(x) == 3])
#         self.stride = self.head.stride
#         self.head.initialize_biases()

#     def forward(self, x, stride_flag=False):
#         if stride_flag == False:
#             x = self.net(x)
#             embed = x[3].view(x[3].shape[-1])
#             x = x[:3]
#             x = self.fpn(x)
#             x = self.head(list(x))
#             # self.embedding_stack.append(embed)
#             # if len(self.embedding_stack) == 17:
#             #     self.embedding_stack.pop(0)
#             # # print(len(self.embedding_stack))
#             # embeds = torch.stack(self.embedding_stack, 0)
#             return x, embed
#         x = self.net(x, True)
#         x = self.fpn(x)
#         x = self.head(list(x))
#         return x

#     def fuse(self):
#         for m in self.modules():
#             if type(m) is Conv and hasattr(m, 'norm'):
#                 m.conv = fuse_conv(m.conv, m.norm)
#                 m.forward = m.fuse_forward
#                 delattr(m, 'norm')
#         return self

# class YOLOv11:
#     def __init__(self):
#         self.dynamic_weighting = {
#         'n':{'csp': [False, True], 'depth' : [1, 1, 1, 1, 1, 1], 'width' : [3, 16, 32, 64, 128, 256]},
#         's':{'csp': [False, True], 'depth' : [1, 1, 1, 1, 1, 1], 'width' : [3, 32, 64, 128, 256, 512]},
#         'm':{'csp': [True, True], 'depth' : [1, 1, 1, 1, 1, 1], 'width' : [3, 64, 128, 256, 512, 512]},
#         'l':{'csp': [True, True], 'depth' : [2, 2, 2, 2, 2, 2], 'width' : [3, 64, 128, 256, 512, 512]},
#         'x':{'csp': [True, True], 'depth' : [2, 2, 2, 2, 2, 2], 'width' : [3, 96, 192, 384, 768, 768]},
#         }
#     def build_model(self, version, num_classes):
#         csp = self.dynamic_weighting[version]['csp']
#         depth = self.dynamic_weighting[version]['depth']
#         width = self.dynamic_weighting[version]['width']
#         return YOLO(width, depth, csp, num_classes)

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def make_divisible(x, divisor=8):
    """Округляем число каналов до ближайшего кратного divisor"""
    return math.ceil(x / divisor) * divisor

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


class Conv(nn.Module):

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k(C3):

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C2f(nn.Module):

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3k2(C2f):

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

class SPPF(nn.Module):

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))    

class Attention(nn.Module):

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class C2PSA(nn.Module):

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class Concat(nn.Module):

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        return torch.cat(x, self.d)

class DWConv(Conv):

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

class DFL(nn.Module):

    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        

class Detect(nn.Module):

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output

    def __init__(self, nc: int = 80, ch: tuple = ()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.tensor([8., 16., 32.])  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def _inference(self, x: list[torch.Tensor]) -> torch.Tensor:
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)
    
    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """Decode bounding boxes from predictions."""
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not self.end2end and not self.xyxy,
            dim=1,
        )
    
    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        preds = preds.permute(0, 2, 1)
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)
    

class DetectionModel(nn.Module):

    def __init__(self, cfg='yolo11n.yaml', ch=3, nc=80):
        super().__init__()
        # Загружаем YAML
        with open(cfg, 'r', encoding='utf-8') as f:
            self.yaml = yaml.safe_load(f)

        self.embeds = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False),
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 128, 1, bias=False),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.LayerNorm(128 * 4 * 4)
        )
        self.model, self.save = self.parse_model(self.yaml, ch=[ch])

    def parse_model(self, d, ch):  # ch = input channels
        layers, save = [], []
        all_layers = []

        # Собираем backbone + head
        if 'backbone' in d:
            all_layers += d['backbone']
        if 'head' in d:
            all_layers += d['head']

        # Читаем множители глубины и ширины
        gd = d.get('depth_mult', 0.5)
        gw = d.get('width_mult', 0.25)
        max_channels = d.get('max_channels', 1024)
        nc = d.get('nc', 80)

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

            # --- Подгоняем аргументы под модуль ---
            if m.__name__ in ("Conv", "C2f", "C3", "C3k2", "SPPF", "C2PSA"):
                c2 = int(args[0] * gw)  # применяем width multiplier
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
                args = [1]  # default dim=1 (concat along channels)
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
        # --- Backbone ---
        x = self.model[0](x)    # Conv(3->64)
        x = self.model[1](x)    # Conv(64->128)
        x = self.model[2](x)    # C3k2(128->256)
        x = self.model[3](x)    # Conv(256->256)
        x4 = self.model[4](x)   # C3k2(256->512)
        x = self.model[5](x4)   # Conv(512->512)
        x6 = self.model[6](x)   # C3k2(512->512)
        x = self.model[7](x6)   # Conv(512->1024)
        x = self.model[8](x)    # C3k2(1024->1024)
        x = self.model[9](x)    # SPPF(1024)
        x10 = self.model[10](x) # C2PSA(1024)

        # --- Head ---
        x = self.model[11](x10)          # Upsample(×2)
        x = torch.cat((x, x6), dim=1)    # Concat backbone P4
        x = self.model[13](x)            # C3k2(768->512)

        x13 = x                          # сохраняем только нужный feature map
        embed = self.embeds(x13)
        x = self.model[14](x13)          # Upsample(×2)
        x = torch.cat((x, x4), dim=1)    # Concat backbone P3
        x = self.model[16](x)            # C3k2(384->256)
        x16 = x

        x = self.model[17](x)            # Conv(256->256, /2)
        x = torch.cat((x, x13), dim=1)   # Concat head P4
        x = self.model[19](x)            # C3k2(768->512)
        x19 = x

        x = self.model[20](x)            # Conv(512->512, /2)
        x = torch.cat((x, x10), dim=1)   # Concat head P5
        x = self.model[22](x)            # C3k2(1536->1024)
        x22 = x

        # --- Detect ---
        return self.model[23]([x16, x19, x22]), embed



def load_pretrained_weights(model, pretrained_path, strict=False):
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

    if 'model' in checkpoint:  # формат Ultralytics
        checkpoint = checkpoint['model'].state_dict()
    elif 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    model_state = model.state_dict()
    loaded = {}

    num_not_found, num_not_match = 0, 0
    for k, v in checkpoint.items():
        if k in model_state and model_state[k].shape == v.shape:
            loaded[k] = v
            # print(f"✅ Слой загружен: {k}")
        elif k not in model_state:
            # print(f"⚠️ Пропускаю слой: {k} - не найден")
            num_not_found += 1
        else:
            # print(f"⚠️ Пропускаю слой: {k} — несовместимая форма {v.shape}, {model_state[k].shape}")
            num_not_match += 1

    # print(f"Не найдено - {num_not_found}")
    # print(f"Формы не совпадают - {num_not_match}")
    # print(f"✅ Загружено {len(loaded)}/{len(model_state)} слоёв")
    model_state.update(loaded)
    model.load_state_dict(model_state, strict=False)
    return model


if __name__ == "__main__":
    yolo = DetectionModel(cfg=r"C:\Users\misas\CS2_NN\src\yolo11n.yaml")
    ultralytics_yolo = YOLO('yolo11n').eval()
    image_path = r"D:\FramesDataset\1-03f67162-abf3-437e-b575-86538acdb399-1-1\tick_4568.jpg"
    # for i, k in enumerate(yolo.named_parameters()):
    #     print(i, k[0])
    # model = yolo.build_model('n', 2).to('cuda')
    model = load_pretrained_weights(yolo, "yolo11n.pt").to('cuda').eval()
    print(model)
    # checkpoint = torch.load("yolo11n.pt", map_location='cpu', weights_only=False)['model'].state_dict()
    # model = torch.load("yolo11n.pt", map_location='cpu', weights_only=False)['model']
    # print(model)
    # for i, (k, v) in enumerate(checkpoint.items()):
    #     print(i, k)
    # print(model)
    # for _ in range(64):
    img = cv2.imread(image_path)        # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    img = cv2.resize(img, (640, 640))  # YOLO input size
    img = img.astype(np.float32) / 255.0  # нормализация 0-1
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.from_numpy(img).unsqueeze(0).to('cuda')
    t = time.time()
    with torch.no_grad():
        output = model(img_tensor)
        out_ult = ultralytics_yolo(img_tensor, save=True)
    # print(time.time() - t)

    print(f"Type of the output :{type(output)}, Length of the output: {len(output)}")
    print(output[0][0].shape)
    print(output[0][1][0].shape)
    print(output[0][1][1].shape)
    print(output[0][1][2].shape)
    print(output[1].shape)
    bboxes = Detect.postprocess(output[0][0], 100, 80)
    print(bboxes)
    img_show = cv2.imread(image_path)    
    # img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
    img_show = cv2.resize(img_show, (640, 640))
    boxes_xyxy, conf, cls = bboxes[0][:, :4], bboxes[0][:, 4], bboxes[0][:, 5]
    boxes_xyxy = ops.box_convert(boxes_xyxy, in_fmt="cxcywh", out_fmt="xyxy")
    keep = ops.nms(boxes_xyxy, conf, iou_threshold=0.3)
    boxes_xyxy = boxes_xyxy[keep]
    conf = conf[keep]
    cls = cls[keep]
    for (x1, y1, x2, y2), c, cl in zip(boxes_xyxy.cpu(), conf.cpu(), cls.cpu()):
        label = f"{int(cl)} {c:.2f}"
        if cl == 0:
            cv2.rectangle(img_show, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img_show, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite('boxes.jpg', img_show)
    print(out_ult)
        # print(bboxes)
        # print(f"The shapes each output feature map : {output[0].shape}, {output[1].shape}, {output[2].shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Обучаемых параметров: {total_params:,}")
    