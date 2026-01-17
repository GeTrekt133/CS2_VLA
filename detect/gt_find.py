import cv2
import numpy as np
from pathlib import Path


def detect_cyan_corners(image_path, output_image_path=None, output_txt_path=None, crop_size=322, save_annotations=False):
    """
    Детектирует голубые углы на изображении и вычисляет bounding box.
    Использует HSV цветовое пространство для надёжной детекции cyan.
    Ищет 4 угловых маркера и строит bbox по их крайним точкам.

    Args:
        image_path: путь к исходному изображению
        output_image_path: путь для сохранения изображения с нарисованным bbox
        output_txt_path: путь для сохранения YOLO аннотаций
        crop_size: размер центральной квадратной области для анализа (по умолчанию 322)
        save_annotations: сохранять ли txt файлы с аннотациями (по умолчанию False)

    Returns:
        yolo_annotation: строка в формате YOLO (class x_center y_center width height)
    """

    try:
        image_full = cv2.imread(str(image_path))
        if image_full is None:
            raise ValueError(f"Не удалось загрузить изображение")
    except Exception as e:
        print(f"   Не удалось загрузить: {image_path}")
        return None

    full_height, full_width = image_full.shape[:2]

    # Вырезаем центральную область crop_size x crop_size
    center_y, center_x = full_height // 2, full_width // 2
    half_crop = crop_size // 2

    y_start = max(0, center_y - half_crop)
    y_end = min(full_height, center_y + half_crop)
    x_start = max(0, center_x - half_crop)
    x_end = min(full_width, center_x + half_crop)

    image = image_full[y_start:y_end, x_start:x_end]
    height, width = image.shape[:2]

    # Конвертируем BGR в HSV для детекции цвета
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Cyan в HSV: Hue ~80-100 (OpenCV использует 0-180), высокая Saturation, средний+ Value
    # Расширенный диапазон для надёжности
    lower_cyan = np.array([75, 100, 100])
    upper_cyan = np.array([105, 255, 255])

    # Создаём маску по HSV диапазону
    mask = cv2.inRange(image_hsv, lower_cyan, upper_cyan)

    # Исключаем центральную область (прицел)
    crosshair_size = 16
    ch_half = crosshair_size // 2
    ch_center_y, ch_center_x = height // 2, width // 2
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
        print("   No cyan regions found")
        return None

    # Собираем центры всех контуров
    centers = []
    for cnt in filtered_contours:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy, cnt))

    if len(centers) < 2:
        print("   Not enough markers found")
        return None

    # Находим крайние точки из всех контуров для построения bbox
    all_points = np.vstack(filtered_contours)
    x_min, y_min, bbox_w, bbox_h = cv2.boundingRect(all_points)
    x_max = x_min + bbox_w
    y_max = y_min + bbox_h

    # Небольшая корректировка границ (уголки могут быть чуть внутри)
    margin = 1
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(width, x_max + margin)
    y_max = min(height, y_max + margin)

    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    # YOLO формат: нормализованные координаты
    x_center = (x_min + bbox_width / 2) / width
    y_center = (y_min + bbox_height / 2) / height
    norm_width = bbox_width / width
    norm_height = bbox_height / height

    yolo_annotation = f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"

    print(f"   Found {len(filtered_contours)} markers -> YOLO: {yolo_annotation}")

    # Сохраняем изображение с bbox
    if output_image_path is None:
        input_path = Path(image_path)
        output_image_path = input_path.parent / f"{input_path.stem}_bbox.jpg"

    output_image = image.copy()
    cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

    # Отмечаем найденные маркеры
    for cx, cy, _ in centers:
        cv2.circle(output_image, (cx, cy), 3, (0, 0, 255), -1)

    cv2.imwrite(str(output_image_path), output_image)

    # Сохраняем аннотацию если нужно
    if save_annotations:
        if output_txt_path is None:
            input_path = Path(image_path)
            output_txt_path = input_path.parent / f"{input_path.stem}.txt"

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
        yolo_annotation = detect_cyan_corners(image_path)
    
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
    
    # Рисуем основной bounding box
    cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
    
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
    
    # Добавляем текстовую информацию
    text_info = f"Class: {class_id} | Size: {bbox_width}x{bbox_height}"
    cv2.putText(result_image, text_info, (x_min, y_min - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
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
            annotation = detect_cyan_corners(
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


if __name__ == "__main__":
    # Batch обработка всех изображений
    INPUT_DIR = r"D:\screenshots"
    OUTPUT_DIR = r"D:\screenshots_output2"

    process_batch(INPUT_DIR, OUTPUT_DIR, crop_size=322, save_annotations=False)
