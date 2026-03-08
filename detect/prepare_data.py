"""
Скрипт для подготовки данных в формате YOLO.

Перемещает изображения и аннотации в правильную структуру:
- screenshots_output/*.jpg → screenshots_output/images/
- screenshots_output/*.txt → screenshots_output/labels/
"""

import shutil
from pathlib import Path
from tqdm import tqdm


def prepare_yolo_structure(data_dir: str):
    """
    Организует файлы в структуру YOLO dataset.

    Args:
        data_dir: путь к директории с файлами
    """
    data_path = Path(data_dir)

    # Создаём поддиректории
    images_dir = data_path / 'images'
    labels_dir = data_path / 'labels'

    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    print(f"Preparing YOLO structure in {data_path}")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")

    # Находим все изображения и аннотации в корне
    image_files = list(data_path.glob('*.jpg')) + list(data_path.glob('*.png'))
    txt_files = list(data_path.glob('*.txt'))

    # Исключаем файлы, которые уже в поддиректориях
    image_files = [f for f in image_files if f.parent == data_path]
    txt_files = [f for f in txt_files if f.parent == data_path]

    print(f"\nFound {len(image_files)} images and {len(txt_files)} txt files in root")

    # Перемещаем изображения
    moved_images = 0
    for img_path in tqdm(image_files, desc="Moving images"):
        dest_path = images_dir / img_path.name
        if not dest_path.exists():
            shutil.move(str(img_path), str(dest_path))
            moved_images += 1

    print(f"Moved {moved_images} images to {images_dir}")

    # Перемещаем аннотации
    moved_labels = 0
    for txt_path in tqdm(txt_files, desc="Moving labels"):
        # Проверяем, что это не служебный файл
        if txt_path.stem.endswith('_bbox'):
            continue

        dest_path = labels_dir / txt_path.name
        if not dest_path.exists():
            shutil.move(str(txt_path), str(dest_path))
            moved_labels += 1

    print(f"Moved {moved_labels} labels to {labels_dir}")

    # Статистика
    print("\n" + "="*60)
    print("Dataset Structure:")
    print("="*60)

    final_images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    final_labels = list(labels_dir.glob('*.txt'))

    print(f"Images: {len(final_images)}")
    print(f"Labels: {len(final_labels)}")

    # Проверяем соответствие
    image_stems = {img.stem for img in final_images}
    label_stems = {lbl.stem for lbl in final_labels}

    matched = len(image_stems & label_stems)
    images_without_labels = len(image_stems - label_stems)
    labels_without_images = len(label_stems - image_stems)

    print(f"\nMatched pairs: {matched}")
    print(f"Images without labels: {images_without_labels}")
    print(f"Labels without images: {labels_without_images}")

    if images_without_labels > 0:
        print("\n⚠️  Warning: Some images don't have corresponding labels")
        print("These images will be skipped during training")

    if labels_without_images > 0:
        print("\n⚠️  Warning: Some labels don't have corresponding images")
        print("Consider cleaning up orphaned label files")

    # Валидация аннотаций
    print("\n" + "="*60)
    print("Validating Annotations:")
    print("="*60)

    invalid_labels = []
    class_counts = {0: 0, 1: 0}  # CT: 0, T: 1

    for label_path in tqdm(final_labels, desc="Validating"):
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()

                    if len(parts) != 5:
                        invalid_labels.append((label_path.name, f"Wrong format: {len(parts)} fields"))
                        continue

                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])

                    # Проверяем class_id
                    if class_id not in [0, 1]:
                        invalid_labels.append((label_path.name, f"Invalid class_id: {class_id}"))
                        continue

                    # Проверяем координаты
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        invalid_labels.append((label_path.name, f"Coordinates out of range [0,1]"))
                        continue

                    class_counts[class_id] += 1

        except Exception as e:
            invalid_labels.append((label_path.name, str(e)))

    print(f"\nTotal annotations: {sum(class_counts.values())}")
    print(f"  CT (class 0): {class_counts[0]}")
    print(f"  T (class 1): {class_counts[1]}")

    if sum(class_counts.values()) > 0:
        ct_ratio = class_counts[0] / sum(class_counts.values()) * 100
        t_ratio = class_counts[1] / sum(class_counts.values()) * 100
        print(f"  CT ratio: {ct_ratio:.1f}%")
        print(f"  T ratio: {t_ratio:.1f}%")

    if len(invalid_labels) > 0:
        print(f"\n⚠️  Found {len(invalid_labels)} invalid labels:")
        for name, error in invalid_labels[:10]:  # Показываем первые 10
            print(f"  - {name}: {error}")
        if len(invalid_labels) > 10:
            print(f"  ... and {len(invalid_labels) - 10} more")
    else:
        print("\n✅ All labels are valid!")

    print("\n" + "="*60)
    print("✅ Data preparation complete!")
    print("="*60)

    print(f"\nYou can now run training:")
    print(f"  python detect/train_yolo.py")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = r"C:\Users\misas\CS2_NN\detect\screenshots_output"

    prepare_yolo_structure(data_dir)
