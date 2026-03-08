"""
Dataset для обучения YOLO11n детектора игроков (1 класс: player).

Два варианта:
1. YOLODetectionDataset — стандартный, загружает images/ + labels/
2. SyntheticYOLODataset — синтетический: пустые фоны + cutout кропов игроков

Аугментации:
- Cutout/копирование боксов с других изображений (имитация нескольких игроков)
- Флэш-эффект (ослепление)
- Частичное перекрытие боксов
- Стандартные геометрические и цветовые трансформации
"""

import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from typing import List, Tuple, Dict, Set, Optional


class YOLODetectionDataset(Dataset):
    """
    Dataset для YOLO11n с аугментациями.

    Структура данных:
        data_dir/
            images/
                tick_1234_5.jpg
                ...
            labels/
                tick_1234_5.txt  # YOLO format: class_id x_center y_center width height
                ...
    """

    def __init__(
        self,
        data_dir: str,
        img_size: int = 640,
        augment: bool = True,
        cutout_prob: float = 0.5,
        max_cutout_boxes: int = 3,
        flash_prob: float = 0.15,
        occlusion_prob: float = 0.3
    ):
        """
        Args:
            data_dir: путь к директории с images/ и labels/
            img_size: размер входного изображения для YOLO
            augment: применять ли аугментации
            cutout_prob: вероятность cutout (вставка боксов с других изображений)
            max_cutout_boxes: максимум дополнительных боксов для cutout
            flash_prob: вероятность флэш-эффекта
            occlusion_prob: вероятность частичного перекрытия бокса
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment = augment
        self.cutout_prob = cutout_prob
        self.max_cutout_boxes = max_cutout_boxes
        self.flash_prob = flash_prob
        self.occlusion_prob = occlusion_prob

        # Находим все изображения
        self.images_dir = self.data_dir / 'images'
        self.labels_dir = self.data_dir / 'labels'

        self.image_paths = sorted(list(self.images_dir.glob('*.jpg')) + list(self.images_dir.glob('*.png')))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.images_dir}")

        print(f"Loaded {len(self.image_paths)} images from {self.images_dir}")

        # Кэш для cutout: загружаем все боксы заранее
        self._bbox_cache = self._build_bbox_cache()

        # Аугментации
        self.transform = self._get_transforms()

    def _build_bbox_cache(self) -> List[Dict]:
        """
        Создаёт кэш всех боксов для cutout операций.

        Returns:
            List[Dict]: [{
                'image_path': Path,
                'bbox': [x_min, y_min, x_max, y_max],
                'class_id': int
            }, ...]
        """
        cache = []

        for img_path in self.image_paths:
            label_path = self.labels_dir / f"{img_path.stem}.txt"

            if not label_path.exists():
                continue

            # Читаем размер изображения
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Парсим YOLO аннотации
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])

                    # Конвертируем в абсолютные координаты
                    x_min = int((x_center - width / 2) * w)
                    y_min = int((y_center - height / 2) * h)
                    x_max = int((x_center + width / 2) * w)
                    y_max = int((y_center + height / 2) * h)

                    cache.append({
                        'image_path': img_path,
                        'bbox': [x_min, y_min, x_max, y_max],
                        'class_id': class_id
                    })

        print(f"Built bbox cache: {len(cache)} boxes from {len(self.image_paths)} images")
        return cache

    def _get_transforms(self):
        """Создаёт пайплайн аугментаций."""
        if not self.augment:
            return A.Compose([
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size,
                             border_mode=cv2.BORDER_CONSTANT, value=0),
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        return A.Compose([
            # Геометрические трансформации
            A.LongestMaxSize(max_size=self.img_size),
            A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size,
                         border_mode=cv2.BORDER_CONSTANT, value=0),

            A.HorizontalFlip(p=0.5),

            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.5),

            # Блюр и шум
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            ], p=0.3),

            # Нормализация и конвертация в тензор
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

    def _apply_flash_effect(self, image: np.ndarray) -> np.ndarray:
        """
        Применяет флэш-эффект (имитация ослепления от flashbang).

        Args:
            image: изображение в формате RGB

        Returns:
            изображение с флэш-эффектом
        """
        # Белая вспышка с градиентом от центра
        h, w = image.shape[:2]

        # Создаём радиальный градиент
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)

        # Нормализуем расстояние
        mask = 1 - (distance / max_distance)
        mask = np.clip(mask, 0, 1)

        # Интенсивность вспышки (случайная)
        intensity = random.uniform(0.3, 0.7)
        mask = mask ** 2  # квадратичный градиент для более резкого перехода
        mask = mask * intensity

        # Применяем белую вспышку
        white = np.ones_like(image) * 255
        flashed = image * (1 - mask[..., None]) + white * mask[..., None]

        return np.clip(flashed, 0, 255).astype(np.uint8)

    def _apply_bbox_occlusion(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Частично перекрывает бокс тёмным прямоугольником (имитация укрытия).

        Args:
            image: изображение в формате RGB
            bbox: [x_center, y_center, width, height] в нормализованных координатах

        Returns:
            изображение с частичным перекрытием
        """
        h, w = image.shape[:2]

        # Конвертируем bbox в абсолютные координаты
        x_center, y_center, width, height = bbox
        x_min = int((x_center - width / 2) * w)
        y_min = int((y_center - height / 2) * h)
        x_max = int((x_center + width / 2) * w)
        y_max = int((y_center + height / 2) * h)

        # Случайное перекрытие (10-40% площади бокса)
        occlusion_ratio = random.uniform(0.1, 0.4)

        # Случайная позиция перекрытия (слева, справа, сверху, снизу)
        side = random.choice(['left', 'right', 'top', 'bottom'])

        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        if side == 'left':
            occ_width = int(bbox_w * occlusion_ratio)
            occ_x1, occ_y1 = x_min, y_min
            occ_x2, occ_y2 = x_min + occ_width, y_max
        elif side == 'right':
            occ_width = int(bbox_w * occlusion_ratio)
            occ_x1, occ_y1 = x_max - occ_width, y_min
            occ_x2, occ_y2 = x_max, y_max
        elif side == 'top':
            occ_height = int(bbox_h * occlusion_ratio)
            occ_x1, occ_y1 = x_min, y_min
            occ_x2, occ_y2 = x_max, y_min + occ_height
        else:  # bottom
            occ_height = int(bbox_h * occlusion_ratio)
            occ_x1, occ_y1 = x_min, y_max - occ_height
            occ_x2, occ_y2 = x_max, y_max

        # Клипаем координаты
        occ_x1 = max(0, occ_x1)
        occ_y1 = max(0, occ_y1)
        occ_x2 = min(w, occ_x2)
        occ_y2 = min(h, occ_y2)

        # Тёмный прямоугольник (случайный цвет стены/объекта)
        color = np.random.randint(20, 80, size=3)
        cv2.rectangle(image, (occ_x1, occ_y1), (occ_x2, occ_y2), color.tolist(), -1)

        return image

    def _apply_cutout(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        class_labels: List[int]
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Копирует боксы с других изображений (cutout).

        Args:
            image: исходное изображение RGB
            bboxes: список боксов в YOLO формате (нормализованные)
            class_labels: список class_id

        Returns:
            (augmented_image, new_bboxes, new_class_labels)
        """
        if len(self._bbox_cache) == 0:
            return image, bboxes, class_labels

        h, w = image.shape[:2]

        # Сколько боксов добавить
        num_cutout = random.randint(1, self.max_cutout_boxes)

        new_bboxes = bboxes.copy()
        new_class_labels = class_labels.copy()

        for _ in range(num_cutout):
            # Выбираем случайный бокс из кэша
            cached_bbox = random.choice(self._bbox_cache)

            # Загружаем исходное изображение
            src_img = cv2.imread(str(cached_bbox['image_path']))
            if src_img is None:
                continue
            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

            # Вырезаем бокс
            x_min, y_min, x_max, y_max = cached_bbox['bbox']
            cropped = src_img[y_min:y_max, x_min:x_max]

            if cropped.size == 0:
                continue

            crop_h, crop_w = cropped.shape[:2]

            # Случайная позиция для вставки на целевом изображении
            paste_x = random.randint(0, max(0, w - crop_w))
            paste_y = random.randint(0, max(0, h - crop_h))

            # Вставляем бокс
            paste_x2 = min(paste_x + crop_w, w)
            paste_y2 = min(paste_y + crop_h, h)

            crop_w_actual = paste_x2 - paste_x
            crop_h_actual = paste_y2 - paste_y

            image[paste_y:paste_y2, paste_x:paste_x2] = cropped[:crop_h_actual, :crop_w_actual]

            # Добавляем новый бокс в YOLO формате
            new_x_center = (paste_x + paste_x2) / 2 / w
            new_y_center = (paste_y + paste_y2) / 2 / h
            new_width = (paste_x2 - paste_x) / w
            new_height = (paste_y2 - paste_y) / h

            new_bboxes.append([new_x_center, new_y_center, new_width, new_height])
            new_class_labels.append(cached_bbox['class_id'])

        return image, new_bboxes, new_class_labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                'image': Tensor[3, H, W],
                'bboxes': Tensor[N, 4],  # YOLO format: x_center, y_center, width, height
                'class_labels': Tensor[N]
            }
        """
        img_path = self.image_paths[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        # Загружаем изображение
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Загружаем аннотации
        bboxes = []
        class_labels = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])

                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)

        # Если нет боксов, создаём пустой (иначе albumentations упадёт)
        if len(bboxes) == 0:
            bboxes = [[0.5, 0.5, 0.01, 0.01]]  # dummy bbox
            class_labels = [0]

        # Аугментации
        if self.augment:
            # Cutout (добавление боксов с других изображений)
            if random.random() < self.cutout_prob:
                image, bboxes, class_labels = self._apply_cutout(image, bboxes, class_labels)

            # Флэш-эффект
            if random.random() < self.flash_prob:
                image = self._apply_flash_effect(image)

            # Частичное перекрытие первого бокса
            if len(bboxes) > 0 and random.random() < self.occlusion_prob:
                image = self._apply_bbox_occlusion(image, bboxes[0])

        # Применяем albumentations трансформации
        transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)

        image = transformed['image']
        bboxes = transformed['bboxes']
        class_labels = transformed['class_labels']

        # Конвертируем в тензоры
        if len(bboxes) == 0:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros((0,), dtype=torch.long)
        else:
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            class_labels = torch.tensor(class_labels, dtype=torch.long)

        return {
            'image': image,
            'bboxes': bboxes,
            'class_labels': class_labels
        }


def extract_player_crops(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    padding: float = 0.05
):
    """
    Вырезает bbox игроков из изображений и сохраняет как отдельные файлы.

    Args:
        images_dir: директория с изображениями
        labels_dir: директория с YOLO label файлами
        output_dir: директория для сохранения кропов
        padding: отступ вокруг bbox (доля от размера bbox)
    """
    from tqdm import tqdm

    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    txt_files = list(labels_path.glob("*.txt"))
    print(f"Found {len(txt_files)} label files")

    crop_count = 0

    for txt_file in tqdm(txt_files, desc="Extracting player crops"):
        img_stem = txt_file.stem

        # Ищем изображение
        img_file = None
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = images_path / f"{img_stem}{ext}"
            if candidate.exists():
                img_file = candidate
                break

        if img_file is None:
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

                # Конвертируем в пиксели
                bbox_w = int(w * img_w)
                bbox_h = int(h * img_h)
                bbox_cx = int(x_center * img_w)
                bbox_cy = int(y_center * img_h)

                pad_w = int(bbox_w * padding)
                pad_h = int(bbox_h * padding)

                x1 = max(0, bbox_cx - bbox_w // 2 - pad_w)
                y1 = max(0, bbox_cy - bbox_h // 2 - pad_h)
                x2 = min(img_w, bbox_cx + bbox_w // 2 + pad_w)
                y2 = min(img_h, bbox_cy + bbox_h // 2 + pad_h)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                out_name = f"crop_{crop_count:06d}.jpg"
                cv2.imwrite(str(output_path / out_name), crop)
                crop_count += 1

    print(f"Extracted {crop_count} player crops to {output_path}")
    return crop_count


def extract_coco_rgba_crops(
    coco_dir: str,
    output_dir: str,
    splits: List[str] = ('train', 'valid', 'test'),
    padding: float = 0.05,
    blur_edge_sigma: float = 1.0,
):
    """
    Извлекает RGBA кропы игроков из COCO segmentation датасета.

    Группирует аннотации body (cat_id=1) и head (cat_id=2) в пары по
    пространственному пересечению: центр head внутри bbox body.
    Каждый кроп — один игрок с объединённой маской (body + head).

    Сохраняет:
    - player_XXXXXX.png — RGBA кроп (alpha = объединённая маска)
    - metadata.json — для каждого кропа: относительные bbox body и head

    Args:
        coco_dir: корень COCO датасета (содержит train/, valid/, test/)
        output_dir: куда сохранять кропы
        padding: отступ вокруг объединённого bbox (доля от размера)
        blur_edge_sigma: sigma для GaussianBlur на краях маски
    """
    coco_path = Path(coco_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metadata = {}  # filename -> list of {class_id, bbox_rel}
    player_count = 0
    stats = {'paired': 0, 'body_only': 0, 'skipped': 0}

    for split in splits:
        split_dir = coco_path / split
        ann_file = split_dir / '_annotations.coco.json'

        if not ann_file.exists():
            print(f"  Skip {split}: no annotations file")
            continue

        with open(ann_file, 'r') as f:
            coco_data = json.load(f)

        id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}

        # Группируем аннотации по image_id и category_id
        from collections import defaultdict
        img_anns = defaultdict(lambda: {'bodies': [], 'heads': []})

        for ann in coco_data['annotations']:
            cat_id = ann['category_id']
            if cat_id == 1:  # body
                img_anns[ann['image_id']]['bodies'].append(ann)
            elif cat_id == 2:  # head
                img_anns[ann['image_id']]['heads'].append(ann)

        print(f"  Processing {split}: {len(img_anns)} images with annotations...")

        img_cache = {}

        for image_id, anns in img_anns.items():
            file_name = id_to_file.get(image_id)
            if file_name is None:
                continue

            bodies = anns['bodies']
            heads = anns['heads']

            if len(bodies) == 0:
                continue

            # Загружаем изображение
            if image_id not in img_cache:
                img_path = split_dir / file_name
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_cache[image_id] = img
                if len(img_cache) > 50:
                    oldest_key = next(iter(img_cache))
                    del img_cache[oldest_key]
            else:
                img = img_cache[image_id]

            img_h, img_w = img.shape[:2]

            # Парим head к body: центр head внутри bbox body
            used_heads = set()

            for body_ann in bodies:
                bx, by, bw, bh = body_ann['bbox']
                if bw < 4 or bh < 4:
                    stats['skipped'] += 1
                    continue

                # Ищем matching head
                matched_head = None
                for hi, head_ann in enumerate(heads):
                    if hi in used_heads:
                        continue
                    hx, hy, hw, hh = head_ann['bbox']
                    # Центр head
                    hcx = hx + hw / 2
                    hcy = hy + hh / 2
                    # Центр head внутри body bbox?
                    if bx <= hcx <= bx + bw and by <= hcy <= by + bh:
                        matched_head = head_ann
                        used_heads.add(hi)
                        break

                # Объединённый bbox (body + head если есть)
                union_x1 = int(bx)
                union_y1 = int(by)
                union_x2 = int(bx + bw)
                union_y2 = int(by + bh)

                if matched_head is not None:
                    hx, hy, hw, hh = matched_head['bbox']
                    union_x1 = min(union_x1, int(hx))
                    union_y1 = min(union_y1, int(hy))
                    union_x2 = max(union_x2, int(hx + hw))
                    union_y2 = max(union_y2, int(hy + hh))

                # Padding
                uw = union_x2 - union_x1
                uh = union_y2 - union_y1
                pad_x = int(uw * padding)
                pad_y = int(uh * padding)
                crop_x1 = max(0, union_x1 - pad_x)
                crop_y1 = max(0, union_y1 - pad_y)
                crop_x2 = min(img_w, union_x2 + pad_x)
                crop_y2 = min(img_h, union_y2 + pad_y)

                crop_w = crop_x2 - crop_x1
                crop_h = crop_y2 - crop_y1
                if crop_w < 8 or crop_h < 8:
                    stats['skipped'] += 1
                    continue

                # Создаём объединённую маску
                mask_full = np.zeros((img_h, img_w), dtype=np.uint8)

                # Body маска
                if body_ann.get('segmentation') and len(body_ann['segmentation']) > 0:
                    body_pts = np.array(body_ann['segmentation'][0], dtype=np.float32).reshape(-1, 2)
                    cv2.fillPoly(mask_full, [body_pts.astype(np.int32)], 255)

                # Head маска (добавляем к body)
                if matched_head is not None and matched_head.get('segmentation') and len(matched_head['segmentation']) > 0:
                    head_pts = np.array(matched_head['segmentation'][0], dtype=np.float32).reshape(-1, 2)
                    cv2.fillPoly(mask_full, [head_pts.astype(np.int32)], 255)

                # Сглаживание краёв маски
                if blur_edge_sigma > 0:
                    ksize = int(blur_edge_sigma * 6) | 1
                    ksize = max(ksize, 3)
                    mask_full = cv2.GaussianBlur(mask_full, (ksize, ksize), blur_edge_sigma)

                # Кропим
                rgb_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
                mask_crop = mask_full[crop_y1:crop_y2, crop_x1:crop_x2]
                rgba = np.dstack([rgb_crop, mask_crop])

                # Сохраняем PNG
                out_name = f"player_{player_count:06d}.png"
                bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
                cv2.imwrite(str(out_path / out_name), bgra)

                # Относительные bbox (в пикселях кропа)
                crop_bboxes = []

                # Body bbox relative to crop
                body_rel_x1 = int(bx) - crop_x1
                body_rel_y1 = int(by) - crop_y1
                body_rel_x2 = int(bx + bw) - crop_x1
                body_rel_y2 = int(by + bh) - crop_y1
                crop_bboxes.append({
                    'class_id': 0,
                    'bbox': [body_rel_x1, body_rel_y1, body_rel_x2, body_rel_y2]
                })

                # Head bbox relative to crop (если есть)
                if matched_head is not None:
                    hx, hy, hw, hh = matched_head['bbox']
                    head_rel_x1 = int(hx) - crop_x1
                    head_rel_y1 = int(hy) - crop_y1
                    head_rel_x2 = int(hx + hw) - crop_x1
                    head_rel_y2 = int(hy + hh) - crop_y1
                    crop_bboxes.append({
                        'class_id': 1,
                        'bbox': [head_rel_x1, head_rel_y1, head_rel_x2, head_rel_y2]
                    })
                    stats['paired'] += 1
                else:
                    stats['body_only'] += 1

                metadata[out_name] = crop_bboxes
                player_count += 1

    # Сохраняем metadata
    with open(out_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nExtraction complete:")
    print(f"  Players: {player_count}")
    print(f"    body+head pairs: {stats['paired']}")
    print(f"    body only: {stats['body_only']}")
    print(f"    skipped: {stats['skipped']}")
    print(f"  Output: {out_path}")


class RealYOLODataset(Dataset):
    """
    Dataset for real labeled images in YOLO format (images/ + labels/).

    Transforms match SyntheticYOLODataset: resize to img_size x img_size (stretch),
    basic color/blur augmentations, horizontal flip.

    Returns same dict format as SyntheticYOLODataset for ConcatDataset compatibility.
    """

    def __init__(
        self,
        data_dir: str,
        img_size: int = 640,
        augment: bool = True,
        flash_prob: float = 0.15,
    ):
        self.img_size = img_size
        self.augment = augment
        self.flash_prob = flash_prob

        data_path = Path(data_dir)
        self.images_dir = data_path / 'images'
        self.labels_dir = data_path / 'labels'

        self.image_paths = sorted(
            list(self.images_dir.glob('*.jpg')) +
            list(self.images_dir.glob('*.png'))
        )

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.images_dir}")

        print(f"RealYOLODataset: {len(self.image_paths)} images from {data_path}")
        self.transform = self._get_transforms()

    def _get_transforms(self):
        if not self.augment:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='yolo', label_fields=['class_labels'], min_visibility=0.3))

        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='yolo', label_fields=['class_labels'], min_visibility=0.3))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        image = cv2.imread(str(img_path))
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Parse YOLO labels
        bboxes = []
        class_labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])
                    if bw > 0.001 and bh > 0.001:
                        # Clamp to [0, 1] (fixes annotation precision errors)
                        x1 = max(0.0, xc - bw / 2)
                        y1 = max(0.0, yc - bh / 2)
                        x2 = min(1.0, xc + bw / 2)
                        y2 = min(1.0, yc + bh / 2)
                        bw_c = x2 - x1
                        bh_c = y2 - y1
                        if bw_c > 0.001 and bh_c > 0.001:
                            bboxes.append([(x1 + x2) / 2, (y1 + y2) / 2, bw_c, bh_c])
                            class_labels.append(cls_id)

        # Flash augmentation (before albumentations)
        if self.augment and self.flash_prob > 0 and random.random() < self.flash_prob:
            h, w = image.shape[:2]
            y, x = np.ogrid[:h, :w]
            cx = random.randint(int(w * 0.3), int(w * 0.7))
            cy = random.randint(int(h * 0.3), int(h * 0.7))
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            max_d = np.sqrt(cx ** 2 + cy ** 2)
            mask = np.clip(1 - dist / max_d, 0, 1)
            intensity = random.uniform(0.75, 1.0)
            mask = mask ** 0.8 * intensity
            white = np.ones_like(image) * 255
            image = np.clip(
                image * (1 - mask[..., None]) + white * mask[..., None],
                0, 255
            ).astype(np.uint8)

        # Dummy bbox for albumentations
        is_dummy = len(bboxes) == 0
        if is_dummy:
            bboxes_in = [[0.5, 0.5, 0.01, 0.01]]
            labels_in = [0]
        else:
            bboxes_in = bboxes
            labels_in = class_labels

        transformed = self.transform(
            image=image, bboxes=bboxes_in, class_labels=labels_in)

        image = transformed['image']
        out_bboxes = transformed['bboxes']
        out_labels = transformed['class_labels']

        if is_dummy:
            out_bboxes = []
            out_labels = []

        if len(out_bboxes) == 0:
            bboxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
        else:
            bboxes_tensor = torch.tensor(out_bboxes, dtype=torch.float32)
            labels_tensor = torch.tensor(out_labels, dtype=torch.long)

        return {
            'image': image,
            'bboxes': bboxes_tensor,
            'class_labels': labels_tensor,
        }


class SyntheticYOLODataset(Dataset):
    """
    Синтетический датасет: пустые фоны карт + cutout кропов игроков.

    Решает проблемы оригинального датасета:
    - Bbox только в центре -> теперь по всему изображению
    - 1 игрок на кадр -> 0-5 игроков (включая negative samples)
    - Фон содержит игрока -> чистые фоны без игроков

    Структура данных:
        backgrounds_dir/   - пустые POV скриншоты карт
        player_crops_dir/  - вырезанные bbox игроков (из extract_player_crops)
    """

    def __init__(
        self,
        backgrounds_dir: str,
        player_crops_dir: Optional[str] = None,
        img_size: int = 640,
        augment: bool = True,
        min_players: int = 1,
        max_players: int = 5,
        negative_ratio: float = 0.15,
        flash_prob: float = 0.15,
        occlusion_prob: float = 0.0,
        blood_prob: float = 0.12,
        muzzle_flash_prob: float = 0.10,
        cover_prob: float = 0.15,
        head_peek_prob: float = 0.08,
        scale_range: Tuple[float, float] = (0.5, 2.5),
        aspect_ratio_range: Tuple[float, float] = (0.75, 1.3),
        blending_mode: str = 'mixed',
        rgba_crops_dir: Optional[str] = None,
        demo_positions_path: Optional[str] = None,
        demo_position_prob: float = 0.7,
    ):
        """
        Args:
            backgrounds_dir: папка с пустыми скриншотами карт
            player_crops_dir: папка с RGB кропами игроков (старый формат, fallback)
            img_size: размер выходного изображения
            augment: применять аугментации
            min_players: минимум игроков (если не negative sample)
            max_players: максимум игроков на изображении
            negative_ratio: доля пустых изображений (без игроков)
            flash_prob: вероятность флэш-эффекта (слеповая граната)
            occlusion_prob: вероятность частичного перекрытия балками
            blood_prob: вероятность эффекта получения урона (красный экран)
            muzzle_flash_prob: вероятность вспышки от выстрела рядом с игроком
            cover_prob: вероятность перекрытия ног укрытием (стена/ящик)
            head_peek_prob: вероятность вставки только головы (пикающий из-за укрытия)
            scale_range: диапазон масштабирования высоты кропа (min, max)
            aspect_ratio_range: рандом множитель ширины относительно высоты (min, max)
            blending_mode: метод вставки кропов ('poisson', 'feather', 'mixed')
            rgba_crops_dir: папка с RGBA кропами + metadata.json (из extract_coco_rgba_crops)
            demo_positions_path: путь к positions.npz (из extract_demo_positions.py)
            demo_position_prob: вероятность использования реальной позиции из демки (0.0-1.0)
        """
        self.img_size = img_size
        self.augment = augment
        self.min_players = min_players
        self.max_players = max_players
        self.negative_ratio = negative_ratio
        self.flash_prob = flash_prob
        self.occlusion_prob = occlusion_prob
        self.blood_prob = blood_prob
        self.muzzle_flash_prob = muzzle_flash_prob
        self.cover_prob = cover_prob
        self.head_peek_prob = head_peek_prob
        self.scale_range = scale_range
        self.aspect_ratio_range = aspect_ratio_range
        self.blending_mode = blending_mode
        self.demo_position_prob = demo_position_prob

        # Demo positions for realistic placement
        if demo_positions_path is not None and Path(demo_positions_path).exists():
            data = np.load(demo_positions_path)
            self.demo_positions = data['positions']  # (N, 4): cx, cy, h, w normalized
            print(f"Loaded {len(self.demo_positions)} demo positions "
                  f"(prob={demo_position_prob:.0%})")
        else:
            self.demo_positions = None

        # Загружаем пути к фонам (только каждый 3-й = реальный FOV)
        bg_path = Path(backgrounds_dir)
        all_bgs = sorted(
            list(bg_path.glob('*.jpg')) + list(bg_path.glob('*.png')) + list(bg_path.glob('*.jpeg'))
        )
        self.background_paths = [p for i, p in enumerate(all_bgs) if i % 3 == 0]
        if len(self.background_paths) == 0:
            raise ValueError(f"No backgrounds found in {backgrounds_dir}")

        # RGBA кропы с сегментацией + метаданные bbox (приоритет)
        if rgba_crops_dir is not None:
            rgba_path = Path(rgba_crops_dir)
            self.use_rgba = True

            # Загружаем metadata.json с относительными bbox
            meta_file = rgba_path / 'metadata.json'
            if not meta_file.exists():
                raise ValueError(f"metadata.json not found in {rgba_crops_dir}")

            with open(meta_file, 'r') as f:
                self.crop_metadata = json.load(f)

            # Загружаем RGBA кропы в память
            self.rgba_crops = []  # list of (rgba_array, bboxes_list)
            for fname, bboxes_info in self.crop_metadata.items():
                fpath = rgba_path / fname
                crop = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
                if crop is not None and crop.ndim == 3 and crop.shape[2] == 4:
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGBA)
                    self.rgba_crops.append((crop, bboxes_info))

            if len(self.rgba_crops) == 0:
                raise ValueError(f"No RGBA crops loaded from {rgba_crops_dir}")

            paired = sum(1 for _, bbs in self.rgba_crops if len(bbs) > 1)
            # Индексы кропов с головой — для head_peek аугментации
            self.head_crop_indices = [
                i for i, (_, bbs) in enumerate(self.rgba_crops)
                if any(bb['class_id'] == 1 for bb in bbs)
            ]
            print(f"Loaded {len(self.background_paths)} backgrounds, "
                  f"{len(self.rgba_crops)} player crops ({paired} with head, "
                  f"{len(self.head_crop_indices)} usable for head_peek)")
        else:
            # Fallback: старый формат RGB кропов (single class)
            self.use_rgba = False

            if player_crops_dir is None:
                raise ValueError("Either rgba_crops_dir or player_crops_dir must be provided")

            crops_path = Path(player_crops_dir)
            crop_files = sorted(
                list(crops_path.glob('*.jpg')) + list(crops_path.glob('*.png'))
            )
            if len(crop_files) == 0:
                raise ValueError(f"No player crops found in {player_crops_dir}")

            self.player_crops = []
            for crop_file in crop_files:
                crop = cv2.imread(str(crop_file))
                if crop is not None:
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    self.player_crops.append(crop)

            print(f"Loaded {len(self.background_paths)} backgrounds, {len(self.player_crops)} player crops")

        # Аугментации
        self.transform = self._get_transforms()

    def _get_transforms(self):
        if not self.augment:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

    def _compute_iou(self, box1, box2):
        """Вычисляет IoU между двумя bbox в формате [x_center, y_center, w, h]."""
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2

        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2

        inter_x1 = max(x1_min, x2_min)
        inter_y1 = max(y1_min, y2_min)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union_area = area1 + area2 - inter_area

        return inter_area / (union_area + 1e-6)

    def _apply_blood_screen(self, image: np.ndarray) -> np.ndarray:
        """Эффект получения урона — экран краснеет.

        Два варианта (50/50):
        1. Общий красный тинт по всему экрану (лёгкий урон)
        2. Радиальное красное пятно с одной стороны (направленный урон)
        """
        h, w = image.shape[:2]
        result = image.astype(np.float32)

        if random.random() < 0.5:
            # Вариант 1: общий тинт — красный оверлей
            alpha = random.uniform(0.15, 0.40)
            red_layer = np.zeros_like(image, dtype=np.float32)
            red_layer[:, :, 0] = 220  # R
            red_layer[:, :, 1] = random.randint(10, 40)   # G (немного)
            red_layer[:, :, 2] = random.randint(10, 30)    # B

            # Виньетка — сильнее по краям (как в CS2)
            y, x = np.ogrid[:h, :w]
            cx, cy = w / 2, h / 2
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            max_d = np.sqrt(cx**2 + cy**2)
            vignette = np.clip(dist / max_d, 0, 1) ** 0.6  # сильнее к краям
            vignette = vignette * 0.7 + 0.3  # min 0.3, max 1.0

            alpha_map = (alpha * vignette)[..., None]
            result = result * (1 - alpha_map) + red_layer * alpha_map
        else:
            # Вариант 2: направленное красное пятно (удар с одной стороны)
            side = random.choice(['left', 'right', 'top', 'bottom'])
            if side == 'left':
                cx = random.randint(0, int(w * 0.2))
                cy = random.randint(int(h * 0.2), int(h * 0.8))
            elif side == 'right':
                cx = random.randint(int(w * 0.8), w)
                cy = random.randint(int(h * 0.2), int(h * 0.8))
            elif side == 'top':
                cx = random.randint(int(w * 0.2), int(w * 0.8))
                cy = random.randint(0, int(h * 0.2))
            else:
                cx = random.randint(int(w * 0.2), int(w * 0.8))
                cy = random.randint(int(h * 0.8), h)

            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            radius = random.uniform(0.3, 0.6) * max(h, w)
            mask = np.clip(1 - dist / radius, 0, 1) ** 1.5

            intensity = random.uniform(0.25, 0.55)
            mask = mask * intensity

            red_spot = np.zeros_like(image, dtype=np.float32)
            red_spot[:, :, 0] = random.randint(180, 240)
            red_spot[:, :, 1] = random.randint(0, 30)
            red_spot[:, :, 2] = random.randint(0, 20)

            result = result * (1 - mask[..., None]) + red_spot * mask[..., None]

        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_muzzle_flash(self, image: np.ndarray,
                            bboxes: List[List[float]]) -> np.ndarray:
        """Вспышка от выстрела вблизи одного из игроков.

        Маленькая яркая вспышка (жёлто-белая) рядом с bbox — имитирует
        момент когда в игрока стреляют или он стреляет в камеру.
        """
        if not bboxes:
            return image

        h, w = image.shape[:2]

        # Выбираем случайный bbox
        bbox = random.choice(bboxes)
        bx, by, bw, bh = bbox

        # Центр вспышки — рядом с bbox (чуть сбоку или сверху)
        offset_x = random.uniform(-bw * 0.5, bw * 0.5)
        offset_y = random.uniform(-bh * 0.6, bh * 0.3)
        flash_cx = int((bx + offset_x) * w)
        flash_cy = int((by + offset_y) * h)
        flash_cx = max(0, min(flash_cx, w - 1))
        flash_cy = max(0, min(flash_cy, h - 1))

        # Радиус вспышки (5-20% от высоты изображения)
        radius = random.uniform(0.05, 0.20) * h

        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - flash_cx)**2 + (y - flash_cy)**2)
        mask = np.clip(1 - dist / radius, 0, 1) ** 1.5

        intensity = random.uniform(0.6, 1.0)
        mask = mask * intensity

        # Жёлто-белая вспышка
        flash_color = np.zeros_like(image, dtype=np.float32)
        flash_color[:, :, 0] = 255  # R
        flash_color[:, :, 1] = random.randint(200, 255)  # G (жёлтый тон)
        flash_color[:, :, 2] = random.randint(120, 200)  # B

        result = image.astype(np.float32)
        result = result * (1 - mask[..., None]) + flash_color * mask[..., None]
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_cover_occlusion(self, image: np.ndarray,
                                bbox: List[float]) -> np.ndarray:
        """Перекрытие нижней части bbox укрытием (стена, ящик, перила).

        Имитирует ситуацию когда игрок виден из-за укрытия —
        ноги скрыты, видна только верхняя часть тела.
        Bbox НЕ обрезается — модель должна предсказать full body.
        """
        h, w = image.shape[:2]
        bx, by, bw, bh = bbox

        # Абсолютные координаты bbox
        x1 = int((bx - bw / 2) * w)
        y1 = int((by - bh / 2) * h)
        x2 = int((bx + bw / 2) * w)
        y2 = int((by + bh / 2) * h)
        bbox_h = y2 - y1
        bbox_w = x2 - x1

        if bbox_h < 20 or bbox_w < 10:
            return image

        # Перекрываем 20-60% нижней части
        cover_ratio = random.uniform(0.2, 0.6)
        cover_top = y2 - int(bbox_h * cover_ratio)

        # Небольшой padding за пределы bbox
        pad_x = random.randint(3, max(4, int(bbox_w * 0.15)))
        pad_y = random.randint(2, max(3, int(bbox_h * 0.05)))
        cover_x1 = max(0, x1 - pad_x)
        cover_x2 = min(w, x2 + pad_x)

        # Перекрытие только внутри bbox (+ маленький padding вниз)
        cover_y2 = min(h, y2 + pad_y)

        # Палитра CS2 укрытий
        palettes = [
            (80, 85, 90),    # серый бетон
            (100, 95, 85),   # светлый бетон
            (60, 55, 50),    # тёмная стена
            (90, 70, 50),    # коричневый ящик
            (70, 80, 60),    # зеленоватый металл
            (50, 50, 55),    # тёмный металл
            (110, 100, 85),  # песчаник
        ]
        base_r, base_g, base_b = random.choice(palettes)

        # Рисуем укрытие с текстурой (не идеально плоское)
        cover_region = image[cover_top:cover_y2, cover_x1:cover_x2].copy()
        noise = np.random.randint(-15, 15, cover_region.shape, dtype=np.int16)
        flat_color = np.full_like(cover_region, [base_r, base_g, base_b],
                                   dtype=np.int16)
        textured = np.clip(flat_color + noise, 0, 255).astype(np.uint8)

        # Верхний край — немного скошенный (не идеальная линия)
        edge_h = max(2, int(bbox_h * 0.03))
        if edge_h > 0 and cover_top > 0:
            for row in range(edge_h):
                alpha = row / edge_h
                y_pos = row
                if y_pos < textured.shape[0]:
                    textured[y_pos] = (
                        cover_region[y_pos].astype(np.float32) * (1 - alpha) +
                        textured[y_pos].astype(np.float32) * alpha
                    ).astype(np.uint8)

        image[cover_top:cover_y2, cover_x1:cover_x2] = textured
        return image

    def _apply_flash_effect(self, image: np.ndarray) -> np.ndarray:
        """Флэш-эффект (ослепление от слеповой гранаты) - агрессивная версия."""
        h, w = image.shape[:2]
        y, x = np.ogrid[:h, :w]

        # Случайный центр вспышки (может быть не в центре)
        center_x = random.randint(int(w * 0.3), int(w * 0.7))
        center_y = random.randint(int(h * 0.3), int(h * 0.7))

        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)

        # Градиент от центра
        mask = 1 - (distance / max_distance)
        mask = np.clip(mask, 0, 1)

        # Очень сильная вспышка (0.75-1.0 интенсивность, более резкий градиент)
        intensity = random.uniform(0.75, 1.0)
        mask = mask ** 0.8 * intensity  # Уменьшили степень для более резкого перехода

        # Альфа-блендинг с белым
        white = np.ones_like(image) * 255
        flashed = image * (1 - mask[..., None]) + white * mask[..., None]
        return np.clip(flashed, 0, 255).astype(np.uint8)

    def _apply_beam_occlusion(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Перекрытие балками внутри bbox (имитация столбов, стен, и т.д.).

        Args:
            image: изображение в формате RGB
            bbox: [x_center, y_center, width, height] в нормализованных координатах

        Returns:
            изображение с балками внутри bbox
        """
        h, w = image.shape[:2]

        # Конвертируем bbox в абсолютные координаты
        x_center, y_center, width, height = bbox
        x_min = int((x_center - width / 2) * w)
        y_min = int((y_center - height / 2) * h)
        x_max = int((x_center + width / 2) * w)
        y_max = int((y_center + height / 2) * h)

        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        # Количество балок (1-3)
        num_beams = random.randint(1, 3)

        for _ in range(num_beams):
            # Ориентация балки
            orientation = random.choice(['vertical', 'horizontal'])

            # Толщина балки (3-8% от размера bbox)
            if orientation == 'vertical':
                thickness = random.randint(int(bbox_w * 0.03), int(bbox_w * 0.08))
                thickness = max(2, thickness)

                # Случайная позиция вдоль ширины bbox (не у края)
                pos = random.randint(x_min + int(bbox_w * 0.2), x_max - int(bbox_w * 0.2) - thickness)

                # Рисуем вертикальную балку
                bx1, by1 = pos, y_min
                bx2, by2 = pos + thickness, y_max
            else:
                thickness = random.randint(int(bbox_h * 0.03), int(bbox_h * 0.08))
                thickness = max(2, thickness)

                # Случайная позиция вдоль высоты bbox
                pos = random.randint(y_min + int(bbox_h * 0.2), y_max - int(bbox_h * 0.2) - thickness)

                # Рисуем горизонтальную балку
                bx1, by1 = x_min, pos
                bx2, by2 = x_max, pos + thickness

            # Клипаем координаты
            bx1 = max(0, min(bx1, w))
            by1 = max(0, min(by1, h))
            bx2 = max(0, min(bx2, w))
            by2 = max(0, min(by2, h))

            # Текстурированная балка (как в cover occlusion)
            beam_h = by2 - by1
            beam_w = bx2 - bx1
            if beam_h <= 0 or beam_w <= 0:
                continue
            palettes = [
                (80, 85, 90), (100, 95, 85), (60, 55, 50),
                (90, 70, 50), (70, 80, 60), (50, 50, 55), (110, 100, 85),
            ]
            base_r, base_g, base_b = random.choice(palettes)
            noise = np.random.randint(-15, 15,
                                      (beam_h, beam_w, 3), dtype=np.int16)
            flat = np.full((beam_h, beam_w, 3),
                           [base_r, base_g, base_b], dtype=np.int16)
            textured = np.clip(flat + noise, 0, 255).astype(np.uint8)

            # Мягкие края (2px blend)
            edge = min(2, beam_h // 2, beam_w // 2)
            if edge > 0:
                orig_region = image[by1:by2, bx1:bx2].copy()
                for e in range(edge):
                    a = (e + 1) / (edge + 1)
                    if orientation == 'vertical':
                        textured[:, e] = (
                            orig_region[:, e].astype(np.float32) * (1 - a) +
                            textured[:, e].astype(np.float32) * a
                        ).astype(np.uint8)
                        textured[:, -(e + 1)] = (
                            orig_region[:, -(e + 1)].astype(np.float32) * (1 - a) +
                            textured[:, -(e + 1)].astype(np.float32) * a
                        ).astype(np.uint8)
                    else:
                        textured[e] = (
                            orig_region[e].astype(np.float32) * (1 - a) +
                            textured[e].astype(np.float32) * a
                        ).astype(np.uint8)
                        textured[-(e + 1)] = (
                            orig_region[-(e + 1)].astype(np.float32) * (1 - a) +
                            textured[-(e + 1)].astype(np.float32) * a
                        ).astype(np.uint8)

            image[by1:by2, bx1:bx2] = textured

        return image

    def _paste_poisson(self, image: np.ndarray, crop: np.ndarray,
                       paste_x: int, paste_y: int) -> np.ndarray:
        """Poisson blending через cv2.seamlessClone.
        Сохраняет градиенты (текстуру) кропа, адаптирует цвета к фону."""
        crop_h, crop_w = crop.shape[:2]
        h, w = image.shape[:2]

        # seamlessClone требует: кроп полностью внутри изображения
        if (paste_x < 0 or paste_y < 0 or
                paste_x + crop_w > w or paste_y + crop_h > h):
            return self._paste_feather(image, crop, paste_x, paste_y)

        center_x = paste_x + crop_w // 2
        center_y = paste_y + crop_h // 2

        mask = np.ones((crop_h, crop_w), dtype=np.uint8) * 255
        # Сужаем маску от краёв (2px) чтобы избежать edge artifacts
        mask[:2, :] = 0
        mask[-2:, :] = 0
        mask[:, :2] = 0
        mask[:, -2:] = 0

        try:
            return cv2.seamlessClone(crop, image, mask,
                                     (center_x, center_y), cv2.NORMAL_CLONE)
        except cv2.error:
            return self._paste_feather(image, crop, paste_x, paste_y)

    def _paste_feather(self, image: np.ndarray, crop: np.ndarray,
                       paste_x: int, paste_y: int,
                       feather_px: int = 8) -> np.ndarray:
        """Вставка с размытыми краями (alpha feathering).
        Центр кропа непрозрачный, края плавно затухают."""
        crop_h, crop_w = crop.shape[:2]
        h, w = image.shape[:2]

        paste_x2 = min(paste_x + crop_w, w)
        paste_y2 = min(paste_y + crop_h, h)
        actual_w = paste_x2 - paste_x
        actual_h = paste_y2 - paste_y

        if actual_w <= 0 or actual_h <= 0:
            return image

        crop_region = crop[:actual_h, :actual_w]

        # Маска с мягкими краями
        mask = np.ones((actual_h, actual_w), dtype=np.float32)
        feather = min(feather_px, actual_w // 4, actual_h // 4)
        feather = max(feather, 1)
        for i in range(feather):
            alpha = (i + 1) / (feather + 1)
            mask[i, :] *= alpha
            mask[-(i + 1), :] *= alpha
            mask[:, i] *= alpha
            mask[:, -(i + 1)] *= alpha

        mask_3ch = mask[..., np.newaxis]
        bg = image[paste_y:paste_y2, paste_x:paste_x2].astype(np.float32)
        blended = crop_region.astype(np.float32) * mask_3ch + bg * (1 - mask_3ch)
        image[paste_y:paste_y2, paste_x:paste_x2] = blended.astype(np.uint8)
        return image

    def _paste_crop(self, image: np.ndarray, crop: np.ndarray,
                    paste_x: int, paste_y: int) -> np.ndarray:
        """Вставляет кроп выбранным методом блендинга."""
        if self.blending_mode == 'poisson':
            return self._paste_poisson(image, crop, paste_x, paste_y)
        elif self.blending_mode == 'feather':
            return self._paste_feather(image, crop, paste_x, paste_y)
        elif self.blending_mode == 'mixed':
            if random.random() < 0.6:
                return self._paste_poisson(image, crop, paste_x, paste_y)
            else:
                return self._paste_feather(image, crop, paste_x, paste_y)
        else:
            # Simple paste (без блендинга)
            h, w = image.shape[:2]
            px2 = min(paste_x + crop.shape[1], w)
            py2 = min(paste_y + crop.shape[0], h)
            image[paste_y:py2, paste_x:px2] = crop[:py2 - paste_y, :px2 - paste_x]
            return image

    def _paste_alpha(self, image: np.ndarray, rgba_crop: np.ndarray,
                     paste_x: int, paste_y: int) -> np.ndarray:
        """Alpha compositing: вставляет RGBA кроп используя alpha-канал как маску.

        Маска следует силуэту игрока (из полигона сегментации) —
        никаких прямоугольных артефактов.
        """
        crop_h, crop_w = rgba_crop.shape[:2]
        h, w = image.shape[:2]

        paste_x2 = min(paste_x + crop_w, w)
        paste_y2 = min(paste_y + crop_h, h)
        actual_w = paste_x2 - paste_x
        actual_h = paste_y2 - paste_y

        if actual_w <= 0 or actual_h <= 0:
            return image

        rgb = rgba_crop[:actual_h, :actual_w, :3]
        alpha = rgba_crop[:actual_h, :actual_w, 3:4].astype(np.float32) / 255.0

        bg_region = image[paste_y:paste_y2, paste_x:paste_x2].astype(np.float32)
        blended = rgb.astype(np.float32) * alpha + bg_region * (1.0 - alpha)
        image[paste_y:paste_y2, paste_x:paste_x2] = blended.astype(np.uint8)
        return image

    def __len__(self) -> int:
        return len(self.background_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Загружаем фон
        bg_path = self.background_paths[idx]
        image = cv2.imread(str(bg_path))
        if image is None:
            # Fallback
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        bboxes = []
        class_labels = []

        # Negative sample?
        is_negative = random.random() < self.negative_ratio

        if not is_negative:
            has_crops = (self.use_rgba and len(self.rgba_crops) > 0) or \
                        (not self.use_rgba and hasattr(self, 'player_crops') and len(self.player_crops) > 0)

            if has_crops:
                num_players = random.randint(self.min_players, self.max_players)

                for _ in range(num_players):
                    # --- Head peek: иногда вставляем только голову (пик из-за укрытия) ---
                    is_head_peek = (
                        self.use_rgba
                        and hasattr(self, 'head_crop_indices')
                        and len(self.head_crop_indices) > 0
                        and random.random() < self.head_peek_prob
                    )

                    if is_head_peek:
                        idx = random.choice(self.head_crop_indices)
                        crop_rgba, crop_bboxes_info_full = self.rgba_crops[idx]

                        # Находим head bbox в метаданных
                        head_bb = None
                        for bb in crop_bboxes_info_full:
                            if bb['class_id'] == 1:
                                head_bb = bb
                                break

                        if head_bb is not None:
                            hx1, hy1, hx2, hy2 = head_bb['bbox']
                            # Добавляем небольшой padding вокруг головы (шея, часть плеч)
                            head_h = hy2 - hy1
                            head_w = hx2 - hx1
                            pad_x = int(head_w * 0.15)
                            pad_y = int(head_h * 0.2)
                            ch, cw = crop_rgba.shape[:2]
                            hx1 = max(0, hx1 - pad_x)
                            hy1 = max(0, hy1 - pad_y)
                            hx2 = min(cw, hx2 + pad_x)
                            hy2 = min(ch, hy2 + pad_y)

                            crop = crop_rgba[hy1:hy2, hx1:hx2].copy()
                            if crop.shape[0] < 8 or crop.shape[1] < 8:
                                is_head_peek = False
                            else:
                                # Один bbox: вся вырезка = class 0 (body)
                                crop_bboxes_info = [{
                                    'class_id': 0,
                                    'bbox': [0, 0, crop.shape[1], crop.shape[0]]
                                }]
                        else:
                            is_head_peek = False

                    if not is_head_peek:
                        if self.use_rgba:
                            # Выбираем случайного игрока (RGBA + bbox metadata)
                            crop_rgba, crop_bboxes_info = random.choice(self.rgba_crops)
                            crop = crop_rgba.copy()  # RGBA (H, W, 4)
                        else:
                            crop_bboxes_info = None
                            crop = random.choice(self.player_crops).copy()  # RGB (H, W, 3)

                    orig_h, orig_w = crop.shape[:2]

                    # --- Decide: demo position or random ---
                    use_demo = (self.demo_positions is not None
                                and random.random() < self.demo_position_prob)
                    demo_pos = None

                    if use_demo:
                        # Sample a real screen-space position from demo data
                        dp = self.demo_positions[
                            random.randint(0, len(self.demo_positions) - 1)]
                        cx_norm, cy_norm, h_norm, w_norm = dp

                        # Scale from projected bbox height (+30% padding)
                        target_h = int(h_norm * h * 1.3)
                        if target_h < 12:
                            use_demo = False
                        else:
                            scale = target_h / orig_h
                            ar_factor = random.uniform(
                                self.aspect_ratio_range[0],
                                self.aspect_ratio_range[1])
                            new_h = max(12, target_h)
                            new_w = max(12, int(orig_w * scale * ar_factor))
                            # Position from projected center
                            demo_pos = (
                                int(cx_norm * w - new_w / 2),
                                int(cy_norm * h - new_h / 2),
                            )

                    if not use_demo:
                        # Original: log-uniform scale
                        log_lo = np.log(self.scale_range[0])
                        log_hi = np.log(self.scale_range[1])
                        scale = np.exp(random.uniform(log_lo, log_hi))

                        ar_factor = random.uniform(self.aspect_ratio_range[0],
                                                   self.aspect_ratio_range[1])

                        new_h = max(12, int(orig_h * scale))
                        new_w = max(12, int(orig_w * scale * ar_factor))

                    # Ограничиваем размер кропа размером изображения
                    new_w = min(new_w, w - 4)
                    new_h = min(new_h, h - 4)
                    if new_w < 12 or new_h < 12:
                        continue

                    # Фактический масштаб (для пересчёта относительных bbox)
                    # sx != sy когда aspect_ratio_range != 1.0
                    sx = new_w / orig_w
                    sy = new_h / orig_h

                    crop = cv2.resize(crop, (new_w, new_h))

                    # Random horizontal flip
                    flipped = random.random() > 0.5
                    if flipped:
                        crop = cv2.flip(crop, 1)

                    # Random brightness/contrast (только на RGB каналах)
                    if self.augment and random.random() > 0.5:
                        a = random.uniform(0.7, 1.3)  # contrast
                        b = random.randint(-30, 30)     # brightness
                        if self.use_rgba:
                            alpha_ch = crop[:, :, 3:4]
                            rgb_ch = np.clip(crop[:, :, :3].astype(np.float32) * a + b, 0, 255).astype(np.uint8)
                            crop = np.dstack([rgb_ch, alpha_ch])
                        else:
                            crop = np.clip(crop.astype(np.float32) * a + b, 0, 255).astype(np.uint8)

                    # Генерируем позицию с retry — гарантируем что ни один bbox
                    # не залезает на зону радара (левый верхний угол экрана)
                    RADAR_W, RADAR_H = 200, 150
                    placed = False

                    for _pos_attempt in range(50):
                        if demo_pos is not None and _pos_attempt == 0:
                            # First attempt: use demo-derived position
                            paste_x = max(0, min(demo_pos[0], max(0, w - new_w)))
                            paste_y = max(0, min(demo_pos[1], max(0, h - new_h)))
                        else:
                            # Fallback / random attempts
                            paste_x = random.randint(0, max(0, w - new_w))
                            paste_y = random.randint(0, max(0, h - new_h))

                        # Проверяем что весь crop не пересекает зону радара
                        if paste_x < RADAR_W and paste_y < RADAR_H:
                            continue

                        # Для RGBA: pre-check всех sub-bbox ДО вставки
                        if self.use_rgba and crop_bboxes_info:
                            any_on_radar = False
                            for bb_info in crop_bboxes_info:
                                rx1, ry1, rx2, ry2 = bb_info['bbox']
                                # Масштабируем
                                srx1, sry1 = rx1 * sx, ry1 * sy
                                srx2, sry2 = rx2 * sx, ry2 * sy
                                # Флип
                                if flipped:
                                    srx1, srx2 = new_w - srx2, new_w - srx1
                                # Абсолютные координаты
                                ax1 = paste_x + srx1
                                ay1 = paste_y + sry1
                                ax2 = paste_x + srx2
                                ay2 = paste_y + sry2
                                # Пересечение с радаром: оба прямоугольника пересекаются
                                # если ax1 < RADAR_W и ay1 < RADAR_H (и ax2 > 0, ay2 > 0 — всегда true)
                                if ax1 < RADAR_W and ay1 < RADAR_H:
                                    any_on_radar = True
                                    break
                            if any_on_radar:
                                continue

                        # IoU с существующими bbox
                        bbox_x_center = (paste_x + new_w / 2) / w
                        bbox_y_center = (paste_y + new_h / 2) / h
                        bbox_w_norm = new_w / w
                        bbox_h_norm = new_h / h
                        overall_bbox = [bbox_x_center, bbox_y_center, bbox_w_norm, bbox_h_norm]

                        too_much_overlap = False
                        for existing_bbox in bboxes:
                            if self._compute_iou(overall_bbox, existing_bbox) > 0.3:
                                too_much_overlap = True
                                break
                        if too_much_overlap:
                            continue

                        placed = True
                        break

                    if not placed:
                        continue  # не удалось найти валидную позицию за 50 попыток

                    # Вставляем кроп (гарантированно вне радара)
                    paste_x2 = min(paste_x + new_w, w)
                    paste_y2 = min(paste_y + new_h, h)
                    actual_w = paste_x2 - paste_x
                    actual_h = paste_y2 - paste_y

                    if self.use_rgba:
                        image = self._paste_alpha(image, crop[:actual_h, :actual_w],
                                                  paste_x, paste_y)

                        # Генерируем bbox для каждого объекта (body + head)
                        for bb_info in crop_bboxes_info:
                            cls_id = bb_info['class_id']
                            rx1, ry1, rx2, ry2 = bb_info['bbox']

                            # Масштабируем под новый размер
                            rx1 = rx1 * sx
                            ry1 = ry1 * sy
                            rx2 = rx2 * sx
                            ry2 = ry2 * sy

                            # Горизонтальный флип
                            if flipped:
                                rx1_new = new_w - rx2
                                rx2_new = new_w - rx1
                                rx1, rx2 = rx1_new, rx2_new

                            # Абсолютные координаты на изображении
                            abs_x1 = paste_x + rx1
                            abs_y1 = paste_y + ry1
                            abs_x2 = paste_x + rx2
                            abs_y2 = paste_y + ry2

                            # Клампим к границам изображения
                            abs_x1 = max(0, min(abs_x1, w))
                            abs_y1 = max(0, min(abs_y1, h))
                            abs_x2 = max(0, min(abs_x2, w))
                            abs_y2 = max(0, min(abs_y2, h))

                            bb_w = abs_x2 - abs_x1
                            bb_h = abs_y2 - abs_y1
                            if bb_w < 8 or bb_h < 8:
                                continue

                            # YOLO формат
                            yolo_xc = (abs_x1 + abs_x2) / 2 / w
                            yolo_yc = (abs_y1 + abs_y2) / 2 / h
                            yolo_w = bb_w / w
                            yolo_h = bb_h / h

                            bboxes.append([yolo_xc, yolo_yc, yolo_w, yolo_h])
                            class_labels.append(cls_id)
                    else:
                        image = self._paste_crop(image, crop[:actual_h, :actual_w],
                                                 paste_x, paste_y)

                        bbox_x_center = (paste_x + actual_w / 2) / w
                        bbox_y_center = (paste_y + actual_h / 2) / h
                        bbox_w_norm = actual_w / w
                        bbox_h_norm = actual_h / h

                        bboxes.append([bbox_x_center, bbox_y_center, bbox_w_norm, bbox_h_norm])
                        class_labels.append(0)

        # Перекрытие балками внутри bbox (имитация столбов, стен)
        if self.augment and len(bboxes) > 0 and random.random() < self.occlusion_prob:
            # Применяем к 1-2 случайным bbox
            num_occluded = min(random.randint(1, 2), len(bboxes))
            occluded_indices = random.sample(range(len(bboxes)), num_occluded)

            for occ_idx in occluded_indices:
                bbox = bboxes[occ_idx]
                image = self._apply_beam_occlusion(image, bbox)

        # Перекрытие ног укрытием (стена, ящик — видна только верхняя часть тела)
        if self.augment and len(bboxes) > 0 and random.random() < self.cover_prob:
            num_covered = min(random.randint(1, 2), len(bboxes))
            covered_indices = random.sample(range(len(bboxes)), num_covered)
            for cov_idx in covered_indices:
                image = self._apply_cover_occlusion(image, bboxes[cov_idx])

        # Вспышка от выстрела рядом с игроком
        if self.augment and len(bboxes) > 0 and random.random() < self.muzzle_flash_prob:
            image = self._apply_muzzle_flash(image, bboxes)

        # Эффект получения урона (красный экран)
        if self.augment and random.random() < self.blood_prob:
            image = self._apply_blood_screen(image)

        # Флэш-эффект (слеповая граната)
        if self.augment and random.random() < self.flash_prob:
            image = self._apply_flash_effect(image)

        # Пустые bbox для albumentations
        if len(bboxes) == 0:
            bboxes_for_transform = [[0.5, 0.5, 0.01, 0.01]]
            labels_for_transform = [0]
            is_dummy = True
        else:
            bboxes_for_transform = bboxes
            labels_for_transform = class_labels
            is_dummy = False

        # Albumentations
        transformed = self.transform(
            image=image, bboxes=bboxes_for_transform, class_labels=labels_for_transform
        )

        image = transformed['image']
        out_bboxes = transformed['bboxes']
        out_labels = transformed['class_labels']

        # Убираем dummy bbox
        if is_dummy:
            out_bboxes = []
            out_labels = []

        # Конвертируем в тензоры
        if len(out_bboxes) == 0:
            bboxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
        else:
            bboxes_tensor = torch.tensor(out_bboxes, dtype=torch.float32)
            labels_tensor = torch.tensor(out_labels, dtype=torch.long)

        return {
            'image': image,
            'bboxes': bboxes_tensor,
            'class_labels': labels_tensor
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function для DataLoader.

    Объединяет батч в формате:
    {
        'images': Tensor[B, 3, H, W],
        'targets': List[Tensor[N_i, 6]]  # каждый элемент: [batch_idx, class_id, x, y, w, h]
    }
    """
    images = []
    targets = []

    for batch_idx, sample in enumerate(batch):
        images.append(sample['image'])

        bboxes = sample['bboxes']
        class_labels = sample['class_labels']

        if len(bboxes) > 0:
            # Формат YOLO: [batch_idx, class_id, x_center, y_center, width, height]
            batch_indices = torch.full((len(bboxes), 1), batch_idx, dtype=torch.float32)
            class_ids = class_labels.unsqueeze(1).float()
            target = torch.cat([batch_indices, class_ids, bboxes], dim=1)
            targets.append(target)

    images = torch.stack(images, dim=0)

    if len(targets) > 0:
        targets = torch.cat(targets, dim=0)
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32)

    return {
        'images': images,
        'targets': targets
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='YOLO Dataset utilities')
    parser.add_argument('--extract-crops', action='store_true',
                        help='Extract player crops from labeled dataset')
    parser.add_argument('--extract-coco', action='store_true',
                        help='Extract RGBA crops from COCO segmentation dataset')
    parser.add_argument('--test-synthetic', action='store_true',
                        help='Test SyntheticYOLODataset')
    parser.add_argument('--images', type=str, default=r'D:\yolo_dataset\images',
                        help='Images directory')
    parser.add_argument('--labels', type=str, default=r'D:\yolo_dataset\labels',
                        help='Labels directory')
    parser.add_argument('--backgrounds', type=str, default=r'D:\backgrounds',
                        help='Backgrounds directory')
    parser.add_argument('--crops', type=str, default=r'D:\yolo_dataset\player_crops',
                        help='Player crops directory (old RGB format)')
    parser.add_argument('--output', type=str, default=r'D:\yolo_dataset\player_crops',
                        help='Output directory for extracted crops')
    parser.add_argument('--coco-dir', type=str,
                        default=r'C:\Users\misas\Downloads\cs2.v6i.coco-segmentation',
                        help='COCO segmentation dataset directory')
    parser.add_argument('--rgba-output', type=str,
                        default=r'D:\yolo_dataset\crops_rgba',
                        help='Output directory for RGBA crops')
    parser.add_argument('--rgba-crops', type=str, default=None,
                        help='RGBA crops directory (body/, head/ subdirs)')
    args = parser.parse_args()

    if args.extract_crops:
        print("="*60)
        print("Extracting player crops from dataset")
        print("="*60)
        extract_player_crops(
            images_dir=args.images,
            labels_dir=args.labels,
            output_dir=args.output,
            padding=0.05
        )

    elif args.extract_coco:
        print("="*60)
        print("Extracting RGBA crops from COCO segmentation dataset")
        print("="*60)
        extract_coco_rgba_crops(
            coco_dir=args.coco_dir,
            output_dir=args.rgba_output,
            splits=['train', 'valid', 'test'],
            padding=0.05,
            blur_edge_sigma=1.0,
        )

    elif args.test_synthetic:
        print("="*60)
        print("Testing SyntheticYOLODataset")
        print("="*60)

        rgba_dir = args.rgba_crops
        dataset = SyntheticYOLODataset(
            backgrounds_dir=args.backgrounds,
            player_crops_dir=args.crops if rgba_dir is None else None,
            img_size=640,
            augment=True,
            min_players=1,
            max_players=5,
            negative_ratio=0.15,
            flash_prob=0.15,
            occlusion_prob=0.3,
            scale_range=(1.5, 3.0),
            rgba_crops_dir=rgba_dir,
        )

        print(f"Dataset size: {len(dataset)} backgrounds")

        # Тест загрузки
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Bboxes: {sample['bboxes'].shape}")
        print(f"  Class labels: {sample['class_labels']}")

        # Тест DataLoader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)
        batch = next(iter(loader))
        print(f"\nBatch test:")
        print(f"  Images shape: {batch['images'].shape}")
        print(f"  Targets shape: {batch['targets'].shape}")
        print(f"  Targets:\n{batch['targets'][:5]}")

        # Визуализация нескольких сэмплов
        CLASS_COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}  # body=green, head=red
        CLASS_NAMES = {0: 'body', 1: 'head'}

        print("\nSaving sample visualizations...")
        output_vis = Path(args.backgrounds).parent / 'synthetic_samples'
        output_vis.mkdir(exist_ok=True)

        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            img = sample['image'].permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            for bbox, cls in zip(sample['bboxes'], sample['class_labels']):
                xc, yc, bw, bh = bbox.tolist()
                cls_id = int(cls.item())
                x1 = int((xc - bw/2) * 640)
                y1 = int((yc - bh/2) * 640)
                x2 = int((xc + bw/2) * 640)
                y2 = int((yc + bh/2) * 640)
                color = CLASS_COLORS.get(cls_id, (255, 255, 0))
                label = CLASS_NAMES.get(cls_id, f'cls{cls_id}')
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imwrite(str(output_vis / f"sample_{i:03d}.jpg"), img)

        print(f"Saved {min(10, len(dataset))} visualizations to {output_vis}")

        print("\n" + "="*60)
        print("Synthetic dataset test passed!")
        print("="*60)

    else:
        print("Usage:")
        print("  Extract COCO:      python dataset_yolo.py --extract-coco")
        print("  Test synthetic:    python dataset_yolo.py --test-synthetic --rgba-crops D:\\yolo_dataset\\crops_rgba")
        print("  Extract RGB crops: python dataset_yolo.py --extract-crops --images ... --labels ...")
