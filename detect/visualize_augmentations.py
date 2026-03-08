"""
Скрипт для визуализации аугментаций датасета.

Показывает оригинальные изображения и их аугментированные версии.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from dataset_yolo import YOLODetectionDataset


def draw_bbox_on_image(image, bboxes, class_labels, img_size=640):
    """
    Рисует bbox на изображении.

    Args:
        image: Tensor[3, H, W] или np.ndarray[H, W, 3]
        bboxes: Tensor[N, 4] - YOLO format (normalized)
        class_labels: Tensor[N]
        img_size: размер изображения

    Returns:
        np.ndarray[H, W, 3] BGR
    """
    # Конвертируем в numpy если нужно
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
    else:
        image = image.copy()

    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
        class_labels = class_labels.cpu().numpy()

    h, w = image.shape[:2]

    for bbox, cls in zip(bboxes, class_labels):
        x_center, y_center, width, height = bbox

        # Конвертируем в абсолютные координаты
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        # Цвет: CT=зелёный, T=красный
        color = (0, 255, 0) if cls == 0 else (0, 0, 255)
        label = 'CT' if cls == 0 else 'T'

        # Рисуем bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Рисуем label
        cv2.putText(image, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


def visualize_sample(dataset, idx, save_path=None):
    """
    Визуализирует один сэмпл из датасета.

    Args:
        dataset: YOLODetectionDataset
        idx: индекс сэмпла
        save_path: путь для сохранения (опционально)
    """
    sample = dataset[idx]

    image = sample['image']
    bboxes = sample['bboxes']
    class_labels = sample['class_labels']

    # Рисуем bbox
    vis_image = draw_bbox_on_image(image, bboxes, class_labels)

    # Конвертируем BGR → RGB для matplotlib
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

    # Показываем
    plt.figure(figsize=(10, 10))
    plt.imshow(vis_image)
    plt.title(f"Sample {idx} - {len(bboxes)} bboxes")
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_augmentations_grid(dataset, idx, num_augmentations=8, save_path=None):
    """
    Создаёт сетку с оригиналом и несколькими аугментированными версиями.

    Args:
        dataset: YOLODetectionDataset
        idx: индекс сэмпла
        num_augmentations: количество аугментированных версий
        save_path: путь для сохранения
    """
    # Создаём сетку
    rows = int(np.ceil((num_augmentations + 1) / 3))
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    # Оригинал (отключаем аугментации)
    original_augment = dataset.augment
    dataset.augment = False
    original_sample = dataset[idx]
    dataset.augment = original_augment

    original_image = draw_bbox_on_image(
        original_sample['image'],
        original_sample['bboxes'],
        original_sample['class_labels']
    )
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    axes[0].imshow(original_image)
    axes[0].set_title(f"Original - {len(original_sample['bboxes'])} bboxes")
    axes[0].axis('off')

    # Аугментированные версии
    for i in range(num_augmentations):
        sample = dataset[idx]

        aug_image = draw_bbox_on_image(
            sample['image'],
            sample['bboxes'],
            sample['class_labels']
        )
        aug_image = cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB)

        axes[i + 1].imshow(aug_image)
        axes[i + 1].set_title(f"Augmented {i+1} - {len(sample['bboxes'])} bboxes")
        axes[i + 1].axis('off')

    # Скрываем неиспользуемые subplot'ы
    for i in range(num_augmentations + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved grid to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_batch(dataset, num_samples=16, save_dir=None):
    """
    Визуализирует батч из нескольких сэмплов.

    Args:
        dataset: YOLODetectionDataset
        num_samples: количество сэмплов
        save_dir: директория для сохранения
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i, idx in enumerate(indices):
        save_path = save_dir / f"sample_{i:03d}_idx{idx}.jpg" if save_dir else None
        visualize_sample(dataset, idx, save_path=save_path)


def visualize_specific_augmentations(dataset, idx, save_dir=None):
    """
    Визуализирует конкретные типы аугментаций отдельно.

    Args:
        dataset: YOLODetectionDataset
        idx: индекс сэмпла
        save_dir: директория для сохранения
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

    # Сохраняем оригинальные параметры
    orig_cutout_prob = dataset.cutout_prob
    orig_flash_prob = dataset.flash_prob
    orig_occlusion_prob = dataset.occlusion_prob

    # 1. Только Cutout
    dataset.cutout_prob = 1.0
    dataset.flash_prob = 0.0
    dataset.occlusion_prob = 0.0

    sample_cutout = dataset[idx]
    img_cutout = draw_bbox_on_image(sample_cutout['image'], sample_cutout['bboxes'], sample_cutout['class_labels'])
    img_cutout = cv2.cvtColor(img_cutout, cv2.COLOR_BGR2RGB)

    # 2. Только Flash
    dataset.cutout_prob = 0.0
    dataset.flash_prob = 1.0
    dataset.occlusion_prob = 0.0

    sample_flash = dataset[idx]
    img_flash = draw_bbox_on_image(sample_flash['image'], sample_flash['bboxes'], sample_flash['class_labels'])
    img_flash = cv2.cvtColor(img_flash, cv2.COLOR_BGR2RGB)

    # 3. Только Occlusion
    dataset.cutout_prob = 0.0
    dataset.flash_prob = 0.0
    dataset.occlusion_prob = 1.0

    sample_occlusion = dataset[idx]
    img_occlusion = draw_bbox_on_image(sample_occlusion['image'], sample_occlusion['bboxes'], sample_occlusion['class_labels'])
    img_occlusion = cv2.cvtColor(img_occlusion, cv2.COLOR_BGR2RGB)

    # 4. Все вместе
    dataset.cutout_prob = 1.0
    dataset.flash_prob = 1.0
    dataset.occlusion_prob = 1.0

    sample_all = dataset[idx]
    img_all = draw_bbox_on_image(sample_all['image'], sample_all['bboxes'], sample_all['class_labels'])
    img_all = cv2.cvtColor(img_all, cv2.COLOR_BGR2RGB)

    # Восстанавливаем оригинальные параметры
    dataset.cutout_prob = orig_cutout_prob
    dataset.flash_prob = orig_flash_prob
    dataset.occlusion_prob = orig_occlusion_prob

    # Визуализируем
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    axes[0, 0].imshow(img_cutout)
    axes[0, 0].set_title(f"Cutout Only - {len(sample_cutout['bboxes'])} bboxes")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_flash)
    axes[0, 1].set_title(f"Flash Only - {len(sample_flash['bboxes'])} bboxes")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(img_occlusion)
    axes[1, 0].set_title(f"Occlusion Only - {len(sample_occlusion['bboxes'])} bboxes")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(img_all)
    axes[1, 1].set_title(f"All Augmentations - {len(sample_all['bboxes'])} bboxes")
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_dir:
        save_path = save_dir / f"augmentations_comparison_idx{idx}.jpg"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    # ============================================================================
    # Configuration
    # ============================================================================

    DATA_DIR = r"C:\Users\misas\CS2_NN\detect\screenshots_output"
    OUTPUT_DIR = r"C:\Users\misas\CS2_NN\detect\visualizations"

    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

    # ============================================================================
    # Dataset
    # ============================================================================

    print("Loading dataset...")
    dataset = YOLODetectionDataset(
        data_dir=DATA_DIR,
        img_size=640,
        augment=True,
        cutout_prob=0.5,
        max_cutout_boxes=3,
        flash_prob=0.2,
        occlusion_prob=0.3
    )

    print(f"Dataset size: {len(dataset)}")

    # ============================================================================
    # Visualizations
    # ============================================================================

    # 1. Сэмпл с оригиналом и аугментациями
    print("\n1. Visualizing augmentation grid...")
    visualize_augmentations_grid(
        dataset,
        idx=0,
        num_augmentations=8,
        save_path=Path(OUTPUT_DIR) / "augmentations_grid.jpg"
    )

    # 2. Сравнение конкретных аугментаций
    print("\n2. Visualizing specific augmentations...")
    visualize_specific_augmentations(
        dataset,
        idx=0,
        save_dir=OUTPUT_DIR
    )

    # 3. Батч сэмплов
    print("\n3. Visualizing batch of samples...")
    visualize_batch(
        dataset,
        num_samples=16,
        save_dir=Path(OUTPUT_DIR) / "batch_samples"
    )

    print(f"\n✅ Visualizations saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
