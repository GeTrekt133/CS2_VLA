"""
Train alive digit classifier.

Usage:
    python train.py --data D:/alive_digit_dataset --epochs 30
"""

import argparse
import csv
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import AliveDigitCNN


class AliveDigitDataset(Dataset):
    """Dataset of 25x25 grayscale digit crops with labels 1-5."""

    def __init__(self, data_dir: str, split: str = 'train', val_ratio: float = 0.15,
                 seed: int = 42, augment: bool = False):
        self.crops_dir = os.path.join(data_dir, 'crops')
        self.augment = augment

        # Load labels
        csv_path = os.path.join(data_dir, 'labels.csv')
        samples = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = int(row['label'])
                if 1 <= label <= 5:
                    samples.append((row['filename'], label))

        # Split train/val deterministically
        random.seed(seed)
        random.shuffle(samples)
        split_idx = int(len(samples) * (1 - val_ratio))

        if split == 'train':
            self.samples = samples[:split_idx]
        else:
            self.samples = samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = cv2.imread(os.path.join(self.crops_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((25, 25), dtype=np.uint8)

        # Augmentation (train only)
        if self.augment:
            # Random shift ±3px
            dx, dy = random.randint(-3, 3), random.randint(-3, 3)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            img = cv2.warpAffine(img, M, (25, 25), borderMode=cv2.BORDER_REFLECT)
            # Random brightness ±20
            delta = random.randint(-20, 20)
            img = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)

        # Normalize to [0, 1], add channel dim
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.unsqueeze(0)  # (1, 25, 25)
        target = label - 1  # label 1-5 → index 0-4
        return tensor, target


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Datasets
    train_ds = AliveDigitDataset(args.data, split='train', augment=True)
    val_ds = AliveDigitDataset(args.data, split='val', augment=False)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Label distribution
    from collections import Counter
    train_labels = Counter(s[1] for s in train_ds.samples)
    print(f"Train label distribution: {dict(sorted(train_labels.items()))}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # Model
    model = AliveDigitCNN().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class weights for imbalanced data
    total = sum(train_labels.values())
    weights = torch.tensor([total / (5 * train_labels.get(i, 1)) for i in range(1, 6)],
                           dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    ckpt_dir = os.path.join(args.data, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_correct += (logits.argmax(1) == targets).sum().item()
            train_total += imgs.size(0)
        scheduler.step()

        # Val
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                logits = model(imgs)
                val_correct += (logits.argmax(1) == targets).sum().item()
                val_total += imgs.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"loss={train_loss/max(train_total,1):.4f} | "
              f"train_acc={train_acc:.3f} | val_acc={val_acc:.3f} | lr={lr:.6f}")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(ckpt_dir, 'best.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Best model saved ({val_acc:.3f})")

    # Save final
    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'final.pt'))
    print(f"\nTraining complete. Best val acc: {best_acc:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='D:/alive_digit_dataset')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
