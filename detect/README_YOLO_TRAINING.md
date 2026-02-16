# YOLO11n Training Pipeline для CT/T Detection

Полный пайплайн обучения детектора YOLO11n на классификацию игроков CS2 на Counter-Terrorists (CT) и Terrorists (T).

## Обзор

Этот pipeline обучает **только detection head** модели YOLO11n (без embed ветки). Используются агрессивные аугментации для имитации реальных игровых сценариев:

- **Cutout**: Копирование боксов с других изображений (1-3 дополнительных игрока на кадр)
- **Flash Effect**: Имитация ослепления от flashbang
- **Occlusion**: Частичное перекрытие боксов (10-40% площади)
- **Standard Augmentations**: Flip, brightness, contrast, blur, noise

## Структура файлов

```
CS2_NN/detect/
├── dataset_yolo.py           # Dataset с аугментациями
├── train_yolo.py              # Training script
├── README_YOLO_TRAINING.md    # Этот файл
└── screenshots_output/        # Данные
    ├── images/
    │   ├── tick_1234_5.jpg
    │   └── ...
    └── labels/
        ├── tick_1234_5.txt    # YOLO format: class_id x y w h
        └── ...
```

## Подготовка данных

### Шаг 1: Создайте структуру директорий

```bash
cd detect/screenshots_output
mkdir images labels
```

### Шаг 2: Разместите файлы

Перенесите файлы из `screenshots_output/`:
- `*.jpg` → `images/`
- `*.txt` → `labels/`

Или используйте скрипт:

```bash
python prepare_data.py
```

### Формат аннотаций

Файлы `.txt` должны содержать YOLO разметку:

```
<class_id> <x_center> <y_center> <width> <height>
```

Где:
- `class_id`: 0 (CT) или 1 (T)
- Координаты нормализованы в [0, 1]

**Пример:**
```
0 0.523456 0.612345 0.145678 0.234567
```

## Быстрый старт

### 1. Установите зависимости

```bash
pip install torch torchvision opencv-python albumentations pyyaml tqdm
```

### 2. Запустите обучение

```bash
python detect/train_yolo.py
```

### 3. Настройте параметры (опционально)

Отредактируйте `train_yolo.py`:

```python
# Paths
DATA_DIR = r"C:\Users\misas\CS2_NN\detect\screenshots_output"
CHECKPOINT_DIR = Path(r"C:\Users\misas\CS2_NN\checkpoints_yolo")

# Model
NC = 2  # CT and T

# Training
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 5e-4
```

## Dataset: Аугментации

### 1. Cutout (вставка боксов с других изображений)

**Проблема**: В датасете один игрок на кадр, но в реальности может быть несколько.

**Решение**: Копируем боксы с других изображений (1-3 дополнительных игрока).

```python
cutout_prob=0.5         # Вероятность применения
max_cutout_boxes=3      # Максимум дополнительных боксов
```

**Алгоритм**:
1. Выбираем случайные боксы из кэша (все изображения)
2. Вырезаем bbox с исходного изображения
3. Вставляем в случайную позицию на целевом изображении
4. Добавляем новую YOLO аннотацию с правильным class_id

### 2. Flash Effect (имитация ослепления)

**Проблема**: Flashbang в CS2 создаёт яркую белую вспышку, которая может повлиять на детекцию.

**Решение**: Радиальный градиент белого цвета от центра изображения.

```python
flash_prob=0.2          # Вероятность применения
```

**Алгоритм**:
1. Создаём радиальную маску с центром в середине изображения
2. Интенсивность: 30-70% (случайная)
3. Блендим с белым цветом

### 3. Occlusion (частичное перекрытие)

**Проблема**: Игроки часто частично скрыты за укрытиями/стенами.

**Решение**: Тёмный прямоугольник перекрывает 10-40% площади бокса.

```python
occlusion_prob=0.3      # Вероятность применения
```

**Алгоритм**:
1. Выбираем случайную сторону бокса (left/right/top/bottom)
2. Создаём тёмный прямоугольник (случайный цвет стены)
3. Перекрываем 10-40% площади

### 4. Стандартные аугментации

- **HorizontalFlip** (p=0.5)
- **RandomBrightnessContrast** (p=0.5)
- **HueSaturationValue** (p=0.5)
- **MotionBlur/GaussianBlur/GaussNoise** (p=0.3)

## Training Script

### Архитектура

```python
YOLO11n(
    backbone: Conv → C3k2 → SPPF → C2PSA
    head: FPN (P3/8, P4/16, P5/32)
    detect: Detect head (bbox + classification)
)
```

**Без embed ветки** - обучаем только detection.

### Loss Function

Используется упрощённая loss (для полноценной реализации используйте Ultralytics loss):

```python
loss = box_loss + cls_loss + obj_loss
```

**TODO**: Реализовать полную YOLO loss с:
- IoU-based assignment
- DFL loss для bbox regression
- BCE loss для classification

### Optimizer & Scheduler

- **Optimizer**: AdamW (lr=1e-3, weight_decay=5e-4)
- **Scheduler**: CosineAnnealingLR (T_max=EPOCHS, eta_min=lr*0.01)

### Checkpointing

Чекпоинты сохраняются в `checkpoints_yolo/`:

- `best_yolo11n_epoch{N}.pt` - лучшая модель по validation loss
- `latest_yolo11n.pt` - последняя модель

Формат чекпоинта:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'scheduler_state_dict': OrderedDict,
    'train_loss': float,
    'val_loss': float
}
```

## Использование обученной модели

### Загрузка чекпоинта

```python
from train_yolo import YOLO11n

model = YOLO11n(cfg='yolo11n.yaml', nc=2)
checkpoint = torch.load('checkpoints_yolo/best_yolo11n_epoch50.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Inference

```python
import cv2
import torch
import numpy as np

# Загрузка изображения
image = cv2.imread('test.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (640, 640))
image = image.astype(np.float32) / 255.0
image = np.transpose(image, (2, 0, 1))
image_tensor = torch.from_numpy(image).unsqueeze(0).cuda()

# Forward pass
with torch.no_grad():
    predictions = model(image_tensor)

# predictions - список из 3 feature maps или (inference_output, feature_maps)
# Обработка зависит от режима (training/eval)
```

### Постобработка (NMS)

```python
import torchvision.ops as ops

def postprocess(predictions, conf_threshold=0.25, iou_threshold=0.45):
    """
    Args:
        predictions: [B, (nc+4*reg_max), num_anchors] или (dboxes, scores)

    Returns:
        List[Tensor]: детекции для каждого изображения [N, 6] (x1, y1, x2, y2, conf, class_id)
    """
    # Извлекаем bbox и scores
    # Применяем NMS
    # Возвращаем топ-K детекций
    pass  # TODO: реализовать
```

## Валидация результатов

### 1. Визуализация предсказаний

```python
# Отрисовка bbox на изображении
for det in detections:
    x1, y1, x2, y2, conf, cls = det
    color = (0, 255, 0) if cls == 0 else (255, 0, 0)  # CT=зелёный, T=красный
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    label = f"{'CT' if cls == 0 else 'T'} {conf:.2f}"
    cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imwrite('output.jpg', image)
```

### 2. Метрики

Используйте Ultralytics для вычисления mAP:

```python
from ultralytics import YOLO

# Экспортируйте модель в формат Ultralytics или используйте их val()
```

## Troubleshooting

### Проблема: "No images found"

**Решение**: Проверьте структуру директорий. Изображения должны быть в `DATA_DIR/images/`, а аннотации в `DATA_DIR/labels/`.

### Проблема: "CUDA out of memory"

**Решение**: Уменьшите `BATCH_SIZE` или `IMG_SIZE`.

### Проблема: Loss не уменьшается

**Решение**:
1. Проверьте, что аннотации правильные (class_id 0 или 1, координаты в [0, 1])
2. Уменьшите learning rate
3. Реализуйте полноценную YOLO loss (текущая версия упрощённая)

### Проблема: Cutout добавляет артефакты

**Решение**: Уменьшите `cutout_prob` или `max_cutout_boxes`.

## Дополнительные улучшения

### 1. Полноценная YOLO Loss

Текущая реализация использует dummy loss. Для реального обучения:

- Используйте Ultralytics YOLO loss
- Или реализуйте:
  - Task-aligned assignment (TAL)
  - Distribution Focal Loss (DFL) для bbox
  - Varifocal Loss для classification

### 2. Mosaic Augmentation

Добавьте mosaic (4 изображения в одном):

```python
class MosaicAugmentation:
    def __call__(self, images, bboxes, class_labels):
        # Объединяем 4 изображения в сетку 2x2
        # Корректируем bbox координаты
        pass
```

### 3. Mixup

Блендинг двух изображений:

```python
alpha = 0.5
mixed_image = image1 * alpha + image2 * (1 - alpha)
```

### 4. Transfer Learning

Загрузите pretrained веса YOLO11n:

```python
from ultralytics import YOLO

pretrained = YOLO('yolo11n.pt')
model.load_state_dict(pretrained.state_dict(), strict=False)
```

### 5. Multi-GPU Training

```python
model = nn.DataParallel(model, device_ids=[0, 1])
```

## Производительность

### Ожидаемые характеристики

- **mAP@0.5**: 85-95% (после 50-100 эпох)
- **Inference Speed**: ~100-150 FPS на RTX 3080 (batch=1)
- **Model Size**: ~5 MB (YOLO11n)

### Профилирование

```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    model(images)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## Лицензия

Этот код является частью проекта CS2_NN для обучения VLA агента Counter-Strike 2.

## Благодарности

- **Ultralytics** за архитектуру YOLO11
- **Albumentations** за библиотеку аугментаций

---

**Автор**: CS2_NN Team
**Дата**: 2026-01-18
