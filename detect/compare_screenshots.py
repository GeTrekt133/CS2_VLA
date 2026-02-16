"""
Скрипт для сравнения скриншотов с GT боксами и чистых скриншотов.

Сравнивает изображения с одинаковыми именами из двух папок:
- screenshots/ - с отрисованными GT боксами (для извлечения координат)
- screenshots_clean/ - чистые скриншоты (для обучения)

Использует SSIM (Structural Similarity Index) для проверки соответствия.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import json
from skimage.metrics import structural_similarity as ssim


# =============================================================================
# Configuration
# =============================================================================

# Пути к директориям
SCREENSHOTS_GT_DIR = r"D:\screenshots2"
SCREENSHOTS_CLEAN_DIR = r"D:\screenshots2_clean"
OUTPUT_DIR = r"c:\Users\misas\CS2_NN\detect\comparison_output"

# SSIM threshold для определения "одинаковых" изображений
# Значения близкие к 1.0 означают высокое сходство
# GT боксы занимают небольшую часть изображения, поэтому SSIM должен быть высоким
SSIM_THRESHOLD = 0.85

# Размер для ресайза при сравнении (для ускорения)
COMPARE_SIZE = (640, 360)  # 16:9 aspect ratio


@dataclass
class ComparisonResult:
    """Результат сравнения пары изображений."""
    filename: str
    ssim_score: float
    is_match: bool
    gt_path: str
    clean_path: str
    error: Optional[str] = None


def load_image(path: Path, resize: Tuple[int, int] = None) -> Optional[np.ndarray]:
    """
    Загружает изображение.

    Args:
        path: путь к изображению
        resize: опциональный размер для ресайза (width, height)

    Returns:
        np.ndarray или None при ошибке
    """
    if not path.exists():
        return None

    image = cv2.imread(str(path))
    if image is None:
        return None

    if resize:
        image = cv2.resize(image, resize)

    return image


def compute_ssim(image1: np.ndarray, image2: np.ndarray, multichannel: bool = True) -> float:
    """
    Вычисляет SSIM между двумя изображениями.

    Args:
        image1: первое изображение (BGR)
        image2: второе изображение (BGR)
        multichannel: использовать все каналы

    Returns:
        float: SSIM score в диапазоне [-1, 1], где 1 = идентичные
    """
    # Конвертируем в grayscale для базового сравнения
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # SSIM по grayscale
    score, _ = ssim(gray1, gray2, full=True)

    return score


def compute_ssim_multichannel(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Вычисляет SSIM по всем каналам (более точно для цветных изображений).

    Args:
        image1: первое изображение (BGR)
        image2: второе изображение (BGR)

    Returns:
        float: средний SSIM score по всем каналам
    """
    scores = []
    for i in range(3):  # B, G, R channels
        score, _ = ssim(image1[:, :, i], image2[:, :, i], full=True)
        scores.append(score)

    return np.mean(scores)


def compute_difference_mask(image1: np.ndarray, image2: np.ndarray,
                           threshold: int = 30) -> np.ndarray:
    """
    Вычисляет маску различий между изображениями.

    Полезно для локализации GT боксов (области отличия).

    Args:
        image1: изображение с GT боксами
        image2: чистое изображение
        threshold: порог для бинаризации различий

    Returns:
        np.ndarray: бинарная маска различий
    """
    # Абсолютная разница
    diff = cv2.absdiff(image1, image2)

    # Конвертируем в grayscale
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Бинаризация
    _, mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)

    # Морфологические операции для очистки
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def find_bbox_from_difference(gt_image: np.ndarray, clean_image: np.ndarray,
                              min_area: int = 100) -> List[Tuple[int, int, int, int]]:
    """
    Находит bounding boxes из различий между GT и чистым изображением.

    Args:
        gt_image: изображение с GT боксами
        clean_image: чистое изображение
        min_area: минимальная площадь контура

    Returns:
        List[Tuple]: список bbox (x, y, w, h)
    """
    # Получаем маску различий
    diff_mask = compute_difference_mask(gt_image, clean_image)

    # Находим контуры
    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, w, h))

    return bboxes


def compare_image_pair(gt_path: Path, clean_path: Path,
                       resize: Tuple[int, int] = COMPARE_SIZE) -> ComparisonResult:
    """
    Сравнивает пару изображений.

    Args:
        gt_path: путь к изображению с GT боксами
        clean_path: путь к чистому изображению
        resize: размер для сравнения

    Returns:
        ComparisonResult
    """
    filename = gt_path.name

    # Загружаем изображения
    gt_image = load_image(gt_path, resize)
    clean_image = load_image(clean_path, resize)

    if gt_image is None:
        return ComparisonResult(
            filename=filename,
            ssim_score=0.0,
            is_match=False,
            gt_path=str(gt_path),
            clean_path=str(clean_path),
            error=f"Failed to load GT image: {gt_path}"
        )

    if clean_image is None:
        return ComparisonResult(
            filename=filename,
            ssim_score=0.0,
            is_match=False,
            gt_path=str(gt_path),
            clean_path=str(clean_path),
            error=f"Failed to load clean image: {clean_path}"
        )

    # Проверяем размеры
    if gt_image.shape != clean_image.shape:
        return ComparisonResult(
            filename=filename,
            ssim_score=0.0,
            is_match=False,
            gt_path=str(gt_path),
            clean_path=str(clean_path),
            error=f"Size mismatch: GT={gt_image.shape}, Clean={clean_image.shape}"
        )

    # Вычисляем SSIM
    ssim_score = compute_ssim(gt_image, clean_image)

    return ComparisonResult(
        filename=filename,
        ssim_score=ssim_score,
        is_match=ssim_score >= SSIM_THRESHOLD,
        gt_path=str(gt_path),
        clean_path=str(clean_path)
    )


def save_comparison_visualization(gt_path: str, clean_path: str, output_path: str, ssim_score: float):
    """
    Сохраняет визуализацию сравнения пары (GT слева, Clean справа, SSIM в заголовке).

    Args:
        gt_path: путь к GT изображению
        clean_path: путь к clean изображению
        output_path: путь для сохранения
        ssim_score: SSIM score
    """
    gt_img = cv2.imread(gt_path)
    clean_img = cv2.imread(clean_path)

    if gt_img is None or clean_img is None:
        return

    # Ресайзим для компактности
    h, w = 360, 640
    gt_resized = cv2.resize(gt_img, (w, h))
    clean_resized = cv2.resize(clean_img, (w, h))

    # Склеиваем горизонтально
    combined = np.hstack([gt_resized, clean_resized])

    # Добавляем заголовок с SSIM
    header_h = 40
    header = np.zeros((header_h, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, f"GT (left) | Clean (right) | SSIM: {ssim_score:.4f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    final = np.vstack([header, combined])
    cv2.imwrite(output_path, final)


def compare_directories(gt_dir: str, clean_dir: str,
                        output_dir: str = None,
                        extensions: List[str] = ['.jpg', '.png'],
                        save_visualizations: bool = True) -> Dict:
    """
    Сравнивает все изображения из двух директорий.

    Args:
        gt_dir: директория с GT изображениями
        clean_dir: директория с чистыми изображениями
        output_dir: директория для сохранения визуализаций
        extensions: расширения файлов для обработки
        save_visualizations: сохранять визуализации итеративно

    Returns:
        Dict с результатами и статистикой
    """
    gt_path = Path(gt_dir)
    clean_path = Path(clean_dir)

    if not gt_path.exists():
        raise ValueError(f"GT directory does not exist: {gt_dir}")
    if not clean_path.exists():
        raise ValueError(f"Clean directory does not exist: {clean_dir}")

    # Создаём папки для визуализаций
    vis_matched_dir = None
    vis_mismatched_dir = None
    if save_visualizations and output_dir:
        vis_matched_dir = Path(output_dir) / "matched"
        vis_mismatched_dir = Path(output_dir) / "mismatched"
        vis_matched_dir.mkdir(parents=True, exist_ok=True)
        vis_mismatched_dir.mkdir(parents=True, exist_ok=True)

    # Находим все изображения в GT директории (рекурсивно)
    gt_files = []
    for ext in extensions:
        gt_files.extend(gt_path.glob(f"*{ext}"))  # в корне
        gt_files.extend(gt_path.glob(f"**/*{ext}"))  # рекурсивно

    print(f"Found {len(gt_files)} images in GT directory")

    results = []
    matched = 0
    mismatched = 0
    errors = 0
    missing_clean = 0

    ssim_scores = []

    for gt_file in tqdm(gt_files, desc="Comparing images"):
        # Соответствующий чистый файл (сохраняем относительный путь)
        relative_path = gt_file.relative_to(gt_path)
        clean_file = clean_path / relative_path

        # Извлекаем game_id из родительской папки
        game_id = gt_file.parent.name
        # Имя файла: tick_XXXXX_Y.jpg -> берём tick и user_id
        filename_stem = gt_file.stem  # tick_XXXXX_Y
        # Формируем имя: {game_id}_{tick}_{user}.jpg
        vis_filename = f"{game_id}_{filename_stem}.jpg"

        if not clean_file.exists():
            missing_clean += 1
            results.append(ComparisonResult(
                filename=gt_file.name,
                ssim_score=0.0,
                is_match=False,
                gt_path=str(gt_file),
                clean_path=str(clean_file),
                error=f"Clean image not found: {clean_file}"
            ))
            continue

        # Сравниваем
        result = compare_image_pair(gt_file, clean_file)
        results.append(result)

        if result.error:
            errors += 1
        elif result.is_match:
            matched += 1
            ssim_scores.append(result.ssim_score)
            # Сохраняем в matched
            if vis_matched_dir:
                vis_path = vis_matched_dir / vis_filename
                save_comparison_visualization(str(gt_file), str(clean_file), str(vis_path), result.ssim_score)
        else:
            mismatched += 1
            ssim_scores.append(result.ssim_score)
            # Сохраняем в mismatched
            if vis_mismatched_dir:
                vis_path = vis_mismatched_dir / vis_filename
                save_comparison_visualization(str(gt_file), str(clean_file), str(vis_path), result.ssim_score)

    # Статистика
    stats = {
        'total': len(gt_files),
        'matched': matched,
        'mismatched': mismatched,
        'errors': errors,
        'missing_clean': missing_clean,
        'ssim_threshold': SSIM_THRESHOLD,
        'ssim_mean': np.mean(ssim_scores) if ssim_scores else 0.0,
        'ssim_std': np.std(ssim_scores) if ssim_scores else 0.0,
        'ssim_min': np.min(ssim_scores) if ssim_scores else 0.0,
        'ssim_max': np.max(ssim_scores) if ssim_scores else 0.0,
    }

    return {
        'results': results,
        'stats': stats
    }


def visualize_comparison(gt_path: str, clean_path: str, output_path: str = None):
    """
    Визуализирует сравнение пары изображений.

    Создаёт изображение с:
    - GT изображением
    - Чистым изображением
    - Маской различий
    - SSIM score

    Args:
        gt_path: путь к GT изображению
        clean_path: путь к чистому изображению
        output_path: путь для сохранения (опционально)
    """
    import matplotlib.pyplot as plt

    gt_image = cv2.imread(gt_path)
    clean_image = cv2.imread(clean_path)

    if gt_image is None or clean_image is None:
        print("Error loading images")
        return

    # Ресайзим для сравнения
    gt_resized = cv2.resize(gt_image, COMPARE_SIZE)
    clean_resized = cv2.resize(clean_image, COMPARE_SIZE)

    # Вычисляем SSIM
    ssim_score = compute_ssim(gt_resized, clean_resized)

    # Маска различий
    diff_mask = compute_difference_mask(gt_resized, clean_resized)

    # Конвертируем BGR → RGB для matplotlib
    gt_rgb = cv2.cvtColor(gt_resized, cv2.COLOR_BGR2RGB)
    clean_rgb = cv2.cvtColor(clean_resized, cv2.COLOR_BGR2RGB)

    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(gt_rgb)
    axes[0].set_title("GT (with boxes)")
    axes[0].axis('off')

    axes[1].imshow(clean_rgb)
    axes[1].set_title("Clean")
    axes[1].axis('off')

    axes[2].imshow(diff_mask, cmap='hot')
    axes[2].set_title(f"Difference (SSIM: {ssim_score:.4f})")
    axes[2].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()


def extract_gt_bboxes_from_pairs(gt_dir: str, clean_dir: str, output_dir: str,
                                  min_area: int = 100):
    """
    Извлекает GT боксы из различий между парами изображений.

    Args:
        gt_dir: директория с GT изображениями
        clean_dir: директория с чистыми изображениями
        output_dir: директория для сохранения результатов
        min_area: минимальная площадь бокса
    """
    gt_path = Path(gt_dir)
    clean_path = Path(clean_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Находим все изображения
    gt_files = list(gt_path.glob("*.jpg")) + list(gt_path.glob("*.png"))

    print(f"Processing {len(gt_files)} image pairs...")

    all_bboxes = {}

    for gt_file in tqdm(gt_files, desc="Extracting bboxes"):
        clean_file = clean_path / gt_file.name

        if not clean_file.exists():
            continue

        gt_image = cv2.imread(str(gt_file))
        clean_image = cv2.imread(str(clean_file))

        if gt_image is None or clean_image is None:
            continue

        # Ресайзим clean к размеру GT если нужно
        if gt_image.shape != clean_image.shape:
            clean_image = cv2.resize(clean_image, (gt_image.shape[1], gt_image.shape[0]))

        # Находим боксы из различий
        bboxes = find_bbox_from_difference(gt_image, clean_image, min_area)

        if bboxes:
            all_bboxes[gt_file.stem] = bboxes

            # Сохраняем визуализацию
            vis_image = clean_image.copy()
            for (x, y, w, h) in bboxes:
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            vis_path = output_path / f"{gt_file.stem}_extracted.jpg"
            cv2.imwrite(str(vis_path), vis_image)

    # Сохраняем JSON с боксами
    json_path = output_path / "extracted_bboxes.json"
    with open(json_path, 'w') as f:
        json.dump(all_bboxes, f, indent=2)

    print(f"\nExtracted bboxes from {len(all_bboxes)} images")
    print(f"Results saved to {output_path}")


def print_report(comparison_result: Dict):
    """
    Выводит отчёт о сравнении.

    Args:
        comparison_result: результат compare_directories()
    """
    stats = comparison_result['stats']

    print("\n" + "=" * 60)
    print("COMPARISON REPORT")
    print("=" * 60)

    print(f"\nTotal images: {stats['total']}")
    print(f"Matched (SSIM >= {stats['ssim_threshold']}): {stats['matched']}")
    print(f"Mismatched: {stats['mismatched']}")
    print(f"Errors: {stats['errors']}")
    print(f"Missing clean images: {stats['missing_clean']}")

    print(f"\nSSIM Statistics:")
    print(f"  Mean: {stats['ssim_mean']:.4f}")
    print(f"  Std:  {stats['ssim_std']:.4f}")
    print(f"  Min:  {stats['ssim_min']:.4f}")
    print(f"  Max:  {stats['ssim_max']:.4f}")

    # Показываем проблемные изображения
    results = comparison_result['results']
    problematic = [r for r in results if not r.is_match and r.error is None]

    if problematic:
        print(f"\nProblematic pairs (SSIM < {stats['ssim_threshold']}):")
        for r in sorted(problematic, key=lambda x: x.ssim_score)[:10]:
            print(f"  {r.filename}: SSIM = {r.ssim_score:.4f}")

    errors = [r for r in results if r.error]
    if errors:
        print(f"\nErrors:")
        for r in errors[:10]:
            print(f"  {r.filename}: {r.error}")

    print("\n" + "=" * 60)


def main():
    """Основная функция."""
    global SSIM_THRESHOLD
    import argparse

    parser = argparse.ArgumentParser(description='Compare GT and clean screenshots')
    parser.add_argument('--gt', type=str, default=SCREENSHOTS_GT_DIR,
                        help='Directory with GT boxes')
    parser.add_argument('--clean', type=str, default=SCREENSHOTS_CLEAN_DIR,
                        help='Directory with clean screenshots')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='Output directory')
    parser.add_argument('--threshold', type=float, default=SSIM_THRESHOLD,
                        help='SSIM threshold for matching')
    args = parser.parse_args()

    gt_dir = args.gt
    clean_dir = args.clean
    output_dir = args.output
    SSIM_THRESHOLD = args.threshold

    # Проверяем пути
    if "PATH_TO" in gt_dir:
        print("Please provide paths via arguments or config!")
        print("  --gt PATH      - directory with GT boxes")
        print("  --clean PATH   - directory with clean screenshots")
        print("  --output PATH  - output directory")
        return

    # Проверяем существование директорий
    gt_path = Path(gt_dir)
    clean_path = Path(clean_dir)

    print(f"GT directory: {gt_dir}")
    print(f"Clean directory: {clean_dir}")
    print(f"GT exists: {gt_path.exists()}")
    print(f"Clean exists: {clean_path.exists()}")

    if gt_path.exists():
        # Показываем содержимое
        all_files = list(gt_path.iterdir())
        print(f"Items in GT dir: {len(all_files)}")
        for f in all_files[:10]:
            print(f"  {f.name} {'[DIR]' if f.is_dir() else ''}")

    # Создаём output директорию заранее
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Сравниваем директории (с итеративной визуализацией)
    print("\nComparing directories...")
    print(f"Visualizations will be saved to: {output_path / 'matched'} and {output_path / 'mismatched'}")
    result = compare_directories(gt_dir, clean_dir, output_dir=output_dir, save_visualizations=True)

    # Выводим отчёт
    print_report(result)

    # JSON с результатами
    results_json = output_path / "comparison_results.json"
    with open(results_json, 'w') as f:
        # Конвертируем dataclass в dict, приводим numpy типы к Python native
        stats_native = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in result['stats'].items()}
        data = {
            'stats': stats_native,
            'results': [
                {
                    'filename': r.filename,
                    'ssim_score': float(r.ssim_score),
                    'is_match': bool(r.is_match),
                    'gt_path': r.gt_path,
                    'clean_path': r.clean_path,
                    'error': r.error
                }
                for r in result['results']
            ]
        }
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {results_json}")
    print(f"Visualizations saved to: {output_path / 'matched'} and {output_path / 'mismatched'}")


if __name__ == "__main__":
    main()
