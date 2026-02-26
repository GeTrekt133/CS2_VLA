"""
RadarEncoder — EfficientNet-B0 based minimap encoder.

Uses EfficientNet-B0 blocks 1-5 (expanded from original 1-4):
  Block 1: 16 ch
  Block 2: 24 ch
  Block 3: 40 ch
  Block 4: 80 ch
  Block 5: 112 ch  ← NEW (was stopping at block 4)

Parameters: ~1.0M (was ~0.35M with blocks 1-4)
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class RadarEncoderEffB0(nn.Module):
    def __init__(self, pretrained=True, in_ch=3, out_ch=64, embed_dim=512):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)

        # Stem (features[0] = [Conv, BN, Act])
        if in_ch != 3:
            conv0 = backbone.features[0][0]
            self.conv_stem = nn.Conv2d(in_ch, conv0.out_channels,
                                       kernel_size=conv0.kernel_size,
                                       stride=conv0.stride,
                                       padding=conv0.padding,
                                       bias=False)
            self.bn0 = backbone.features[0][1]
            self.act0 = backbone.features[0][2]
        else:
            self.conv_stem = backbone.features[0][0]
            self.bn0 = backbone.features[0][1]
            self.act0 = backbone.features[0][2]

        # EfficientNet-B0 MBConv blocks 1–5 (features[1:6])
        # Block 1: 16 ch, Block 2: 24 ch, Block 3: 40 ch, Block 4: 80 ch, Block 5: 112 ch
        self.blocks = nn.Sequential(*backbone.features[1:6])

        # Project to out_ch then embed_dim
        last_ch = 112  # output channels of EfficientNet-B0 block 5
        self.proj = nn.Sequential(
            nn.Conv2d(last_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_ch, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, in_ch, H, W) — radar crop, e.g. (B, 3, 224, 224)
        x = self.act0(self.bn0(self.conv_stem(x)))
        x = self.blocks(x)
        x = self.proj(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.norm(x)
        return x  # (B, embed_dim)


if __name__ == "__main__":
    model = RadarEncoderEffB0(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")   # (2, 512)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total:,}")  # ~1.0M
