import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class RadarEncoderEffB0(nn.Module):
    def __init__(self, pretrained=True, in_ch=3, out_ch=64, embed_dim=512):
        super().__init__()
        # Загрузим стандартную B0 из torchvision
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)

        # Если вход у радара не RGB — заменим первый conv
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

        # Возьмём первые 3 блока EfficientNet (до MBConv3 включительно)
        # Полный список блоков EfficientNet-B0 (7 групп):
        # 0-stem, [1-7] — MBConv blocks
        self.blocks = nn.Sequential(*backbone.features[1:5])  # до третьего блока включительно

        # Добавим небольшой conv, чтобы привести размер каналов к out_ch
        last_ch = 80  # выход 4-го блока EfficientNet-B0
        self.proj = nn.Sequential(
            nn.Conv2d(last_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_ch, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, 1, 145, 190)
        x = self.act0(self.bn0(self.conv_stem(x)))
        x = self.blocks(x)
        x = self.proj(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.norm(x)
        return x


if __name__ == "__main__":
    model = RadarEncoderEffB0()
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print(output.shape)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Обучаемых параметров: {total_params:,}")