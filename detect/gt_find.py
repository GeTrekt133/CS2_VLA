import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm


def detect_purple_corners(image_path, output_image_path=None, output_txt_path=None, crop_size=322, save_annotations=False, return_full_coords=False):
    """
    Детектирует фиолетовые углы на изображении и вычисляет bounding box.
    Использует HSV цветовое пространство для надёжной детекции purple (F600FF).
    Ищет 4 угловых маркера и строит bbox по их крайним точкам.

    Всегда class_id=0 (player).

    Args:
        image_path: путь к исходному изображению
        output_image_path: путь для сохранения изображения с нарисованным bbox
        output_txt_path: путь для сохранения YOLO аннотаций
        crop_size: размер центральной квадратной области для анализа (по умолчанию 322)
        save_annotations: сохранять ли txt файлы с аннотациями (по умолчанию False)
        return_full_coords: если True, возвращает координаты относительно полного изображения

    Returns:
        yolo_annotation: строка в формате YOLO (class x_center y_center width height)
        Если return_full_coords=True, координаты нормализованы относительно полного изображения
    """

    try:
        image_full = cv2.imread(str(image_path))
        if image_full is None:
            raise ValueError(f"Не удалось загрузить изображение")
    except Exception as e:
        print(f"   Не удалось загрузить: {image_path}")
        return None

    full_height, full_width = image_full.shape[:2]

    # Вырезаем центральную область crop_size x crop_size для поиска маркеров
    center_y, center_x = full_height // 2, full_width // 2
    half_crop = crop_size // 2

    y_start = max(0, center_y - half_crop)
    y_end = min(full_height, center_y + half_crop)
    x_start = max(0, center_x - half_crop)
    x_end = min(full_width, center_x + half_crop)

    image_crop = image_full[y_start:y_end, x_start:x_end]
    crop_height, crop_width = image_crop.shape[:2]

    # Конвертируем BGR в HSV для детекции цвета
    image_hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)

    # Purple (F600FF) в HSV: RGB(246,0,255) → Hue ≈ 149 в OpenCV (0-180 scale)
    # Расширенный диапазон для надёжности: H=[140,160], высокая Saturation, высокий Value
    lower_purple = np.array([140, 100, 100])
    upper_purple = np.array([160, 255, 255])

    # Создаём маску по HSV диапазону
    mask = cv2.inRange(image_hsv, lower_purple, upper_purple)

    # Исключаем центральную область (прицел)
    crosshair_size = 16
    ch_half = crosshair_size // 2
    ch_center_y, ch_center_x = crop_height // 2, crop_width // 2
    mask[ch_center_y - ch_half:ch_center_y + ch_half,
         ch_center_x - ch_half:ch_center_x + ch_half] = 0

    # Морфологическая операция для заполнения пробелов
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Находим контуры
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтруем по площади
    min_area = 8
    max_area = 2000  # Угловые маркеры не должны быть огромными
    filtered_contours = [cnt for cnt in contours
                         if min_area < cv2.contourArea(cnt) < max_area]

    if len(filtered_contours) == 0:
        print("   No purple regions found")
        return None

    # Собираем центры всех контуров (в координатах кропа)
    centers_crop = []
    for cnt in filtered_contours:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers_crop.append((cx, cy, cnt))

    if len(centers_crop) < 2:
        print("   Not enough markers found")
        return None

    # Находим крайние точки из всех контуров для построения bbox (в координатах кропа)
    all_points = np.vstack(filtered_contours)
    x_min_crop, y_min_crop, bbox_w, bbox_h = cv2.boundingRect(all_points)
    x_max_crop = x_min_crop + bbox_w
    y_max_crop = y_min_crop + bbox_h

    # Небольшая корректировка границ (уголки могут быть чуть внутри)
    margin = 1
    x_min_crop = max(0, x_min_crop - margin)
    y_min_crop = max(0, y_min_crop - margin)
    x_max_crop = min(crop_width, x_max_crop + margin)
    y_max_crop = min(crop_height, y_max_crop + margin)

    class_id = 0  # single class: player

    if return_full_coords:
        # Пересчитываем координаты относительно полного изображения
        x_min_full = x_start + x_min_crop
        y_min_full = y_start + y_min_crop
        x_max_full = x_start + x_max_crop
        y_max_full = y_start + y_max_crop

        bbox_width = x_max_full - x_min_full
        bbox_height = y_max_full - y_min_full

        # YOLO формат: нормализованные координаты относительно ПОЛНОГО изображения
        x_center = (x_min_full + bbox_width / 2) / full_width
        y_center = (y_min_full + bbox_height / 2) / full_height
        norm_width = bbox_width / full_width
        norm_height = bbox_height / full_height
    else:
        # Координаты относительно кропа (старое поведение)
        bbox_width = x_max_crop - x_min_crop
        bbox_height = y_max_crop - y_min_crop

        x_center = (x_min_crop + bbox_width / 2) / crop_width
        y_center = (y_min_crop + bbox_height / 2) / crop_height
        norm_width = bbox_width / crop_width
        norm_height = bbox_height / crop_height

    yolo_annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"

    print(f"   Found {len(filtered_contours)} markers -> YOLO: {yolo_annotation}")

    # Сохраняем изображение с bbox (на кропе для debug)
    if output_image_path is not None:
        output_image = image_crop.copy()
        # Фиолетовый цвет: F600FFFF в RGBA = (246, 0, 255) → BGR = (255, 0, 246)
        cv2.rectangle(output_image, (x_min_crop, y_min_crop), (x_max_crop, y_max_crop), (255, 0, 246), 2)

        # Отмечаем найденные маркеры
        for cx, cy, _ in centers_crop:
            cv2.circle(output_image, (cx, cy), 3, (0, 0, 255), -1)

        cv2.imwrite(str(output_image_path), output_image)

    # Сохраняем аннотацию если нужно
    if save_annotations and output_txt_path is not None:
        with open(str(output_txt_path), 'w') as f:
            f.write(yolo_annotation)

    return yolo_annotation


def draw_bbox(image_path, yolo_annotation=None, output_path=None, show=True):
    """
    Отрисовывает bounding box на изображении.
    
    Args:
        image_path: путь к исходному изображению
        yolo_annotation: YOLO аннотация (или None для автоматического детектирования)
        output_path: путь для сохранения результата
        show: показывать ли окно с результатом
    
    Returns:
        image_with_bbox: изображение с нарисованным bounding box
    """
    
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError("Не удалось загрузить изображение")
    except Exception as e:
        print(f"❌ Ошибка при загрузке изображения: {e}")
        return None
    
    height, width = image.shape[:2]
    
    # Если аннотация не передана, детектируем сами
    if yolo_annotation is None:
        yolo_annotation = detect_purple_corners(image_path)
    
    if yolo_annotation is None:
        return None
    
    # Парсим YOLO аннотацию
    parts = yolo_annotation.strip().split()
    class_id = int(parts[0])
    x_center_norm = float(parts[1])
    y_center_norm = float(parts[2])
    width_norm = float(parts[3])
    height_norm = float(parts[4])
    
    # Конвертируем обратно в пиксели
    x_center = int(x_center_norm * width)
    y_center = int(y_center_norm * height)
    bbox_width = int(width_norm * width)
    bbox_height = int(height_norm * height)
    
    x_min = x_center - bbox_width // 2
    y_min = y_center - bbox_height // 2
    x_max = x_min + bbox_width
    y_max = y_min + bbox_height
    
    # Копируем изображение для рисования
    result_image = image.copy()
    
    # Рисуем основной bounding box (фиолетовый)
    cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (255, 0, 246), 2)
    
    # Рисуем центр
    cv2.circle(result_image, (x_center, y_center), 5, (0, 0, 255), -1)
    
    # Рисуем углы
    corner_size = 8
    corners = [
        (x_min, y_min),
        (x_max, y_min),
        (x_min, y_max),
        (x_max, y_max)
    ]
    for corner in corners:
        cv2.circle(result_image, corner, corner_size, (255, 0, 0), 2)
    
    # Добавляем текстовую информацию (фиолетовый)
    text_info = f"Class: {class_id} | Size: {bbox_width}x{bbox_height}"
    cv2.putText(result_image, text_info, (x_min, y_min - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 246), 2)
    
    text_center = f"Center: ({x_center_norm:.3f}, {y_center_norm:.3f})"
    cv2.putText(result_image, text_center, (x_min, y_min - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Сохраняем результат
    if output_path is None:
        input_path = Path(image_path)
        output_path = input_path.parent / f"{input_path.stem}_annotated.png"
    
    cv2.imwrite(str(output_path), result_image)
    print(f"✅ Аннотированное изображение сохранено: {output_path}")
    
    # Показываем окно если требуется
    if show:
        cv2.imshow("Bounding Box", result_image)
        print("   Нажмите любую клавишу для закрытия окна...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result_image


def process_batch(input_dir, output_dir, crop_size=322, save_annotations=False):
    """
    Обрабатывает все изображения в директории рекурсивно.

    Args:
        input_dir: путь к директории с изображениями
        output_dir: путь к директории для сохранения результатов
        crop_size: размер центральной квадратной области для анализа
        save_annotations: сохранять ли txt файлы с аннотациями

    Returns:
        dict: статистика обработки
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Создаём выходную директорию
    output_path.mkdir(parents=True, exist_ok=True)

    # Поддерживаемые форматы изображений
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # Находим все изображения рекурсивно
    all_images = []
    for ext in image_extensions:
        all_images.extend(input_path.rglob(f'*{ext}'))
        all_images.extend(input_path.rglob(f'*{ext.upper()}'))

    # Убираем дубликаты
    all_images = list(set(all_images))
    all_images.sort()

    stats = {
        'total': len(all_images),
        'success': 0,
        'failed': 0,
        'no_detection': 0
    }

    print(f"\n{'='*60}")
    print(f"Batch Processing")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Crop size: {crop_size}x{crop_size}")
    print(f"Total images found: {stats['total']}")
    print(f"{'='*60}\n")

    for idx, image_path in enumerate(all_images, 1):
        print(f"\n[{idx}/{stats['total']}] Processing: {image_path.name}")

        # Генерируем уникальное имя для выходного файла
        # Сохраняем структуру папок в имени файла
        relative_path = image_path.relative_to(input_path)
        safe_name = str(relative_path).replace('\\', '_').replace('/', '_')
        safe_name = Path(safe_name).stem

        output_image_path = output_path / f"{safe_name}_bbox.jpg"
        output_txt_path = output_path / f"{safe_name}.txt" if save_annotations else None

        try:
            annotation = detect_purple_corners(
                image_path,
                output_image_path=output_image_path,
                output_txt_path=output_txt_path,
                crop_size=crop_size,
                save_annotations=save_annotations
            )

            if annotation:
                stats['success'] += 1
            else:
                stats['no_detection'] += 1

        except Exception as e:
            print(f"   Error: {e}")
            stats['failed'] += 1

    print(f"\n{'='*60}")
    print(f"Processing Complete")
    print(f"{'='*60}")
    print(f"Total processed: {stats['total']}")
    print(f"Success: {stats['success']}")
    print(f"No detection: {stats['no_detection']}")
    print(f"Failed: {stats['failed']}")
    print(f"{'='*60}\n")

    return stats


def process_from_comparison_results(
    comparison_json_path: str,
    output_dir: str,
    ssim_threshold: float = 0.8,
    crop_size: int = 322,
    save_annotations: bool = True,
    resize_size: int = 640
):
    """
    Обрабатывает изображения на основе результатов сравнения из comparison_results.json.
    Берёт только пары с SSIM > threshold и ищет GT боксы на GT изображениях,
    затем сохраняет аннотации для clean изображений.

    Все bbox сохраняются с class_id=0 (player).

    Сохраняет:
    - images/ - полные clean изображения (оригинальный размер)
    - images_resized/ - полные clean изображения ресайзнутые до resize_size x resize_size
    - labels/ - YOLO аннотации (координаты относительно полного изображения)
    - debug/ - debug визуализации (кроп с bbox)

    Args:
        comparison_json_path: путь к comparison_results.json
        output_dir: директория для сохранения результатов
        ssim_threshold: минимальный SSIM для обработки (по умолчанию 0.8)
        crop_size: размер центральной области для поиска маркеров
        save_annotations: сохранять ли txt файлы с аннотациями
        resize_size: размер для ресайза (по умолчанию 640)

    Returns:
        dict: статистика обработки
    """
    # Загружаем результаты сравнения
    with open(comparison_json_path, 'r') as f:
        comparison_data = json.load(f)

    results = comparison_data['results']
    print(f"Loaded {len(results)} comparison results")

    # Фильтруем по SSIM threshold
    valid_pairs = [r for r in results if r['ssim_score'] >= ssim_threshold and r['error'] is None]
    print(f"Found {len(valid_pairs)} pairs with SSIM >= {ssim_threshold}")

    if not valid_pairs:
        print("No valid pairs to process!")
        return {'total': 0, 'success': 0, 'failed': 0, 'no_detection': 0}

    # Создаём выходные директории
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_resized_dir = output_path / "images_resized"
    labels_dir = output_path / "labels"
    debug_dir = output_path / "debug"

    images_dir.mkdir(parents=True, exist_ok=True)
    images_resized_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        'total': len(valid_pairs),
        'success': 0,
        'failed': 0,
        'no_detection': 0
    }

    print(f"\n{'='*60}")
    print(f"Processing with SSIM filtering")
    print(f"{'='*60}")
    print(f"SSIM threshold: {ssim_threshold}")
    print(f"Valid pairs: {stats['total']}")
    print(f"Output: {output_dir}")
    print(f"Resize: {resize_size}x{resize_size}")
    print(f"{'='*60}\n")

    for pair in tqdm(valid_pairs, desc="Processing GT images"):
        gt_path = Path(pair['gt_path'])
        clean_path = Path(pair['clean_path'])

        # Извлекаем game_id и filename для именования
        game_id = gt_path.parent.name
        filename_stem = gt_path.stem  # tick_XXXXX_Y

        # Формируем выходные имена: {game_id}_{tick}_{user}
        out_name = f"{game_id}_{filename_stem}"

        try:
            # Детектируем GT боксы на GT изображении
            # return_full_coords=True для координат относительно полного изображения
            # class_id=0 (player)
            annotation = detect_purple_corners(
                gt_path,
                output_image_path=debug_dir / f"{out_name}_debug.jpg",
                output_txt_path=None,
                crop_size=crop_size,
                save_annotations=False,
                return_full_coords=True  # Координаты относительно полного изображения
            )

            if annotation:
                # Загружаем полное clean изображение
                clean_img = cv2.imread(str(clean_path))
                if clean_img is not None:
                    # Сохраняем полное clean изображение (оригинальный размер)
                    cv2.imwrite(str(images_dir / f"{out_name}.jpg"), clean_img)

                    # Сохраняем ресайзнутую версию
                    resized_img = cv2.resize(clean_img, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(str(images_resized_dir / f"{out_name}.jpg"), resized_img)

                    # Сохраняем аннотацию (координаты нормализованы, работают для любого размера)
                    if save_annotations:
                        with open(labels_dir / f"{out_name}.txt", 'w') as f:
                            f.write(annotation)

                    stats['success'] += 1
                else:
                    stats['failed'] += 1
            else:
                stats['no_detection'] += 1

        except Exception as e:
            print(f"Error processing {gt_path.name}: {e}")
            stats['failed'] += 1

    print(f"\n{'='*60}")
    print(f"Processing Complete")
    print(f"{'='*60}")
    print(f"Total processed: {stats['total']}")
    print(f"Success: {stats['success']}")
    print(f"No detection: {stats['no_detection']}")
    print(f"Failed: {stats['failed']}")
    print(f"{'='*60}")
    print(f"\nOutput structure:")
    print(f"  {images_dir}/ - full clean images (original size)")
    print(f"  {images_resized_dir}/ - full clean images ({resize_size}x{resize_size})")
    print(f"  {labels_dir}/ - YOLO annotations (normalized coords)")
    print(f"  {debug_dir}/ - debug visualizations (crop with bbox)")

    return stats


if __name__ == "__main__":
    # Обработка на основе результатов сравнения
    COMPARISON_JSON = r"c:\Users\misas\CS2_NN\detect\comparison_output\comparison_results.json"
    OUTPUT_DIR = r"D:\yolo_dataset"

    process_from_comparison_results(
        comparison_json_path=COMPARISON_JSON,
        output_dir=OUTPUT_DIR,
        ssim_threshold=0.8,
        crop_size=322,
        save_annotations=True,
        resize_size=640
    )
