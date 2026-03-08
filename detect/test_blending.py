"""
Визуальный тест блендинга: сравнение simple / feather / poisson.

Извлекает кропы игроков из скриншотов (по фиолетовым углам),
вставляет на фоны тремя методами, сохраняет сравнение.
"""

import cv2
import numpy as np
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gt_find import detect_purple_corners


def extract_crops_from_screenshots(screenshots_dir, max_crops=20):
    """Извлекает кропы игроков из скриншотов по фиолетовым углам."""
    screenshots_path = Path(screenshots_dir)
    files = sorted(list(screenshots_path.glob('*.jpg')) + list(screenshots_path.glob('*.png')))

    crops = []
    for img_path in files:
        if len(crops) >= max_crops:
            break

        annotation = detect_purple_corners(str(img_path), return_full_coords=True)
        if annotation is None:
            continue

        parts = annotation.strip().split()
        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        x1 = max(0, int((xc - w / 2) * img_w))
        y1 = max(0, int((yc - h / 2) * img_h))
        x2 = min(img_w, int((xc + w / 2) * img_w))
        y2 = min(img_h, int((yc + h / 2) * img_h))

        crop = img[y1:y2, x1:x2]
        if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
            crops.append(crop)
            print(f"  Crop {len(crops)}: {crop.shape[1]}x{crop.shape[0]} from {img_path.name}")

    print(f"\nExtracted {len(crops)} crops total")
    return crops


def paste_simple(image, crop, x, y):
    """Простая вставка (без блендинга)."""
    h, w = image.shape[:2]
    ch, cw = crop.shape[:2]
    x2 = min(x + cw, w)
    y2 = min(y + ch, h)
    result = image.copy()
    result[y:y2, x:x2] = crop[:y2 - y, :x2 - x]
    return result


def paste_feather(image, crop, x, y, feather_px=8):
    """Вставка с мягкими краями."""
    h, w = image.shape[:2]
    ch, cw = crop.shape[:2]
    x2 = min(x + cw, w)
    y2 = min(y + ch, h)
    aw = x2 - x
    ah = y2 - y
    if aw <= 0 or ah <= 0:
        return image.copy()

    crop_region = crop[:ah, :aw]

    mask = np.ones((ah, aw), dtype=np.float32)
    feather = min(feather_px, aw // 4, ah // 4)
    feather = max(feather, 1)
    for i in range(feather):
        alpha = (i + 1) / (feather + 1)
        mask[i, :] *= alpha
        mask[-(i + 1), :] *= alpha
        mask[:, i] *= alpha
        mask[:, -(i + 1)] *= alpha

    mask_3ch = mask[..., np.newaxis]
    result = image.copy()
    bg = result[y:y2, x:x2].astype(np.float32)
    blended = crop_region.astype(np.float32) * mask_3ch + bg * (1 - mask_3ch)
    result[y:y2, x:x2] = blended.astype(np.uint8)
    return result


def paste_poisson(image, crop, x, y):
    """Poisson blending через cv2.seamlessClone."""
    h, w = image.shape[:2]
    ch, cw = crop.shape[:2]

    if x < 0 or y < 0 or x + cw > w or y + ch > h:
        return paste_feather(image, crop, x, y)

    center_x = x + cw // 2
    center_y = y + ch // 2

    mask = np.ones((ch, cw), dtype=np.uint8) * 255
    mask[:2, :] = 0
    mask[-2:, :] = 0
    mask[:, :2] = 0
    mask[:, -2:] = 0

    try:
        return cv2.seamlessClone(crop, image.copy(), mask,
                                  (center_x, center_y), cv2.NORMAL_CLONE)
    except cv2.error:
        return paste_feather(image, crop, x, y)


def create_comparison(bg, crops, output_path, num_crops=3):
    """Создаёт сравнительное изображение: 3 метода рядом."""
    h, w = bg.shape[:2]

    # Выбираем кропы и позиции (одинаковые для всех методов)
    placements = []
    selected = random.sample(crops, min(num_crops, len(crops)))
    for crop in selected:
        ch, cw = crop.shape[:2]
        # Масштабируем если слишком большой
        max_dim = min(w, h) // 3
        if max(ch, cw) > max_dim:
            scale = max_dim / max(ch, cw)
            crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)))
            ch, cw = crop.shape[:2]

        px = random.randint(10, max(11, w - cw - 10))
        py = random.randint(10, max(11, h - ch - 10))
        placements.append((crop, px, py))

    # Применяем 3 метода
    img_simple = bg.copy()
    img_feather = bg.copy()
    img_poisson = bg.copy()

    for crop, px, py in placements:
        img_simple = paste_simple(img_simple, crop, px, py)
        img_feather = paste_feather(img_feather, crop, px, py)
        img_poisson = paste_poisson(img_poisson, crop, px, py)

    # Добавляем подписи
    font = cv2.FONT_HERSHEY_SIMPLEX
    for img, label in [(img_simple, "SIMPLE (raw paste)"),
                        (img_feather, "FEATHER (soft edges)"),
                        (img_poisson, "POISSON (seamless)")]:
        cv2.putText(img, label, (10, 30), font, 0.8, (0, 0, 0), 4)
        cv2.putText(img, label, (10, 30), font, 0.8, (0, 255, 0), 2)

    # Масштабируем для конкатенации (одинаковая высота)
    target_h = 720
    results = []
    for img in [img_simple, img_feather, img_poisson]:
        scale = target_h / img.shape[0]
        resized = cv2.resize(img, (int(img.shape[1] * scale), target_h))
        results.append(resized)

    comparison = np.hstack(results)
    cv2.imwrite(str(output_path), comparison)
    return comparison


def main():
    screenshots_dir = r"D:\screenshots2\1-1b9c68ed-f2e3-4f54-9a56-b62cb542f859-1-1"
    backgrounds_dir = r"D:\backgrounds"
    output_dir = Path(r"D:\blending_test")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Blending Comparison Test")
    print("=" * 60)

    # 1. Извлекаем кропы
    print(f"\n1. Extracting crops from {screenshots_dir}...")
    crops = extract_crops_from_screenshots(screenshots_dir, max_crops=15)

    if len(crops) == 0:
        print("No crops extracted! Check screenshots directory.")
        return

    # 2. Загружаем фоны
    bg_path = Path(backgrounds_dir)
    bg_files = sorted(list(bg_path.glob('*.jpg')) + list(bg_path.glob('*.png')))
    print(f"\n2. Found {len(bg_files)} backgrounds")

    if len(bg_files) == 0:
        print("No backgrounds found!")
        return

    # 3. Генерируем сравнения
    print(f"\n3. Generating comparisons...")
    num_comparisons = min(8, len(bg_files))

    for i in range(num_comparisons):
        bg = cv2.imread(str(bg_files[i]))
        if bg is None:
            continue

        output_path = output_dir / f"comparison_{i:03d}.jpg"
        num_crops = random.randint(1, min(3, len(crops)))
        create_comparison(bg, crops, output_path, num_crops=num_crops)
        print(f"  Saved: {output_path.name} ({num_crops} crops)")

    # 4. Отдельно: крупный план одного кропа на одном фоне
    print(f"\n4. Close-up comparison...")
    bg = cv2.imread(str(bg_files[0]))
    crop = crops[0]

    # Масштабируем кроп
    ch, cw = crop.shape[:2]
    scale = 1.0
    crop_scaled = cv2.resize(crop, (int(cw * scale), int(ch * scale)))
    ch, cw = crop_scaled.shape[:2]

    # Центрируем
    bh, bw = bg.shape[:2]
    px = bw // 2 - cw // 2
    py = bh // 2 - ch // 2

    img_s = paste_simple(bg, crop_scaled, px, py)
    img_f = paste_feather(bg, crop_scaled, px, py)
    img_p = paste_poisson(bg, crop_scaled, px, py)

    # Crop region for close-up
    margin = 40
    rx1 = max(0, px - margin)
    ry1 = max(0, py - margin)
    rx2 = min(bw, px + cw + margin)
    ry2 = min(bh, py + ch + margin)

    closeups = []
    for img, label in [(img_s, "SIMPLE"), (img_f, "FEATHER"), (img_p, "POISSON")]:
        region = img[ry1:ry2, rx1:rx2].copy()
        # Масштабируем для наглядности
        region = cv2.resize(region, (region.shape[1] * 2, region.shape[0] * 2),
                           interpolation=cv2.INTER_NEAREST)
        cv2.putText(region, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(region, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        closeups.append(region)

    # Выравниваем высоту
    max_h = max(c.shape[0] for c in closeups)
    padded = []
    for c in closeups:
        if c.shape[0] < max_h:
            pad = np.zeros((max_h - c.shape[0], c.shape[1], 3), dtype=np.uint8)
            c = np.vstack([c, pad])
        padded.append(c)

    closeup_compare = np.hstack(padded)
    closeup_path = output_dir / "closeup_comparison.jpg"
    cv2.imwrite(str(closeup_path), closeup_compare)
    print(f"  Saved: {closeup_path.name}")

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_dir}")
    print(f"  {num_comparisons} full comparisons + 1 close-up")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
