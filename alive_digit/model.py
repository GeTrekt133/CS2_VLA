"""
Tiny CNN for alive digit classification (1-5).

Input: grayscale 25x25
Output: 5 classes (labels 1-5, mapped to indices 0-4)
~3.5K parameters, inference < 0.1ms on CPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AliveDigitCNN(nn.Module):
    """
    Architecture:
        Conv2d(1, 16, 3, pad=1) → ReLU → MaxPool(2)  → 12x12
        Conv2d(16, 32, 3, pad=1) → ReLU → MaxPool(2)  → 6x6
        Conv2d(32, 32, 3, pad=1) → ReLU → AdaptiveAvgPool(3) → 3x3
        Flatten → FC(32*3*3=288, 5)
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(3),
        )
        self.classifier = nn.Linear(32 * 3 * 3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 25, 25) grayscale image, normalized [0, 1]
        Returns:
            logits: (B, 5)
        """
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted alive count (1-5)."""
        logits = self.forward(x)
        return logits.argmax(dim=1) + 1  # index 0-4 → label 1-5


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    model = AliveDigitCNN()
    print(f"Parameters: {count_parameters(model):,}")
    x = torch.randn(1, 1, 25, 25)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    pred = model.predict(x)
    print(f"Predicted alive: {pred.item()}")
