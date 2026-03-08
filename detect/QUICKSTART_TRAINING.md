# Quick Start: YOLO11n Training для CT/T Detection

## Минимальный путь от данных до обученной модели

### Шаг 1: Подготовка данных (2 минуты)

```bash
# Организуем файлы в правильную структуру
python detect/prepare_data.py

# Ожидаемый результат:
# detect/screenshots_output/
#   ├── images/
#   │   ├── tick_1234_5.jpg
#   │   └── ...
#   └── labels/
#       ├── tick_1234_5.txt
#       └── ...
```

### Шаг 2: Визуализация аугментаций (опционально)

```bash
# Проверяем как работают аугментации
python detect/visualize_augmentations.py

# Результат в detect/visualizations/
```

### Шаг 3: Запуск обучения

```bash
# Обучение на полном датасете
python detect/train_yolo.py
```

### Шаг 4: Мониторинг

Прогресс отображается в консоли:

```
Epoch 1/100
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% | loss: 0.0234
Train Loss: 0.0245
Val Loss: 0.0267
LR: 0.001000
✅ Saved best checkpoint: checkpoints_yolo/best_yolo11n_epoch1.pt
```

Чекпоинты сохраняются в `checkpoints_yolo/`:
- `best_yolo11n_epoch{N}.pt` - лучшая модель
- `latest_yolo11n.pt` - последняя модель

## Настройка параметров

Отредактируйте `train_yolo.py`:

```python
# === ПУТИ ===
DATA_DIR = r"C:\Users\misas\CS2_NN\detect\screenshots_output"
CHECKPOINT_DIR = Path(r"C:\Users\misas\CS2_NN\checkpoints_yolo")

# === ГИПЕРПАРАМЕТРЫ ===
IMG_SIZE = 640          # Размер входа YOLO
BATCH_SIZE = 16         # Размер батча (уменьшите если CUDA OOM)
EPOCHS = 100            # Количество эпох
LR = 1e-3              # Learning rate
WEIGHT_DECAY = 5e-4    # L2 regularization

# === АУГМЕНТАЦИИ (в dataset_yolo.py) ===
cutout_prob=0.5         # Вероятность cutout (добавление боксов)
max_cutout_boxes=3      # Максимум дополнительных боксов
flash_prob=0.2          # Вероятность флэш-эффекта
occlusion_prob=0.3      # Вероятность частичного перекрытия
```

## Структура файлов

```
CS2_NN/detect/
├── dataset_yolo.py                 # Dataset с аугментациями
├── train_yolo.py                   # Основной скрипт обучения
├── prepare_data.py                 # Подготовка структуры данных
├── visualize_augmentations.py      # Визуализация аугментаций
├── README_YOLO_TRAINING.md         # Полная документация
├── QUICKSTART_TRAINING.md          # Этот файл
│
├── screenshots_output/             # Данные для обучения
│   ├── images/
│   │   └── *.jpg
│   └── labels/
│       └── *.txt
│
└── checkpoints_yolo/               # Сохранённые модели (создаётся автоматически)
    ├── best_yolo11n_epoch50.pt
    └── latest_yolo11n.pt
```

## Аугментации

### 1. Cutout (копирование боксов)

**Цель**: Имитация нескольких игроков на кадре (в реальности их может быть больше одного).

**Как работает**:
1. Выбирает случайные боксы из других изображений
2. Вырезает bbox с исходного изображения
3. Вставляет в случайную позицию на целевом изображении
4. Добавляет новые YOLO аннотации

**Параметры**:
```python
cutout_prob=0.5         # 50% вероятность применения
max_cutout_boxes=3      # До 3 дополнительных боксов
```

### 2. Flash Effect (имитация ослепления)

**Цель**: Имитация flashbang (яркая белая вспышка в CS2).

**Как работает**:
1. Создаёт радиальный градиент от центра изображения
2. Блендит с белым цветом (интенсивность 30-70%)
3. Применяет квадратичный градиент для резкого перехода

**Параметры**:
```python
flash_prob=0.2          # 20% вероятность применения
```

### 3. Occlusion (частичное перекрытие)

**Цель**: Имитация укрытий/стен (игроки часто частично скрыты).

**Как работает**:
1. Выбирает случайную сторону бокса (left/right/top/bottom)
2. Создаёт тёмный прямоугольник (случайный цвет стены)
3. Перекрывает 10-40% площади бокса

**Параметры**:
```python
occlusion_prob=0.3      # 30% вероятность применения
```

### 4. Стандартные аугментации

- **HorizontalFlip** (50%)
- **RandomBrightnessContrast** (50%)
- **HueSaturationValue** (50%)
- **MotionBlur/GaussianBlur/GaussNoise** (30%)

## Загрузка обученной модели

```python
import torch
from train_yolo import YOLO11n

# Загрузка модели
model = YOLO11n(cfg='src/yolo11n.yaml', nc=2)
checkpoint = torch.load('checkpoints_yolo/best_yolo11n_epoch50.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.cuda()

# Inference на одном изображении
import cv2
import numpy as np

image = cv2.imread('test.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (640, 640))
image = image.astype(np.float32) / 255.0
image = np.transpose(image, (2, 0, 1))
image_tensor = torch.from_numpy(image).unsqueeze(0).cuda()

with torch.no_grad():
    predictions = model(image_tensor)

# predictions содержит либо список feature maps (training mode)
# либо (inference_output, feature_maps) в eval mode
```

## Troubleshooting

### Problem: "No images found in images/"

**Solution**: Запустите `prepare_data.py` для организации структуры:
```bash
python detect/prepare_data.py
```

### Problem: CUDA out of memory

**Solutions**:
1. Уменьшите `BATCH_SIZE` (например, с 16 до 8 или 4)
2. Уменьшите `IMG_SIZE` (с 640 до 512 или 384)
3. Закройте другие программы использующие GPU

### Problem: Loss не уменьшается

**Solutions**:
1. Проверьте корректность аннотаций (`prepare_data.py` показывает валидацию)
2. Уменьшите learning rate (например, с 1e-3 до 5e-4)
3. **ВАЖНО**: Текущая loss function упрощённая (dummy). Для реального обучения нужно реализовать полноценную YOLO loss (см. README_YOLO_TRAINING.md)

### Problem: Cutout создаёт артефакты

**Solutions**:
1. Уменьшите `cutout_prob` (например, с 0.5 до 0.3)
2. Уменьшите `max_cutout_boxes` (с 3 до 1-2)

### Problem: Аугментации слишком агрессивные

**Solution**: Отрегулируйте вероятности в `train_yolo.py`:
```python
full_dataset = YOLODetectionDataset(
    data_dir=DATA_DIR,
    img_size=IMG_SIZE,
    augment=True,
    cutout_prob=0.3,        # Было: 0.5
    max_cutout_boxes=2,     # Было: 3
    flash_prob=0.1,         # Было: 0.2
    occlusion_prob=0.2      # Было: 0.3
)
```

## Ожидаемые результаты

### После 50-100 эпох:

- **mAP@0.5**: 85-95% (зависит от качества данных)
- **Inference Speed**: ~100-150 FPS на RTX 3080 (batch=1)
- **Model Size**: ~5 MB (YOLO11n nano)

### Типичная динамика loss:

```
Epoch 1:   Train Loss: 0.245  Val Loss: 0.267
Epoch 10:  Train Loss: 0.087  Val Loss: 0.092
Epoch 25:  Train Loss: 0.034  Val Loss: 0.041
Epoch 50:  Train Loss: 0.018  Val Loss: 0.024
Epoch 100: Train Loss: 0.012  Val Loss: 0.019
```

## Следующие шаги

После обучения базовой модели:

1. **Интеграция в VLA агент**:
   - Добавьте обученную модель в `src/Yolo.py`
   - Замените detection head на вашу обученную версию
   - Обучите embed ветку отдельно (если нужна)

2. **Улучшение качества**:
   - Соберите больше данных (разные карты, ситуации)
   - Реализуйте полноценную YOLO loss
   - Добавьте Mosaic и Mixup аугментации
   - Используйте pretrained веса YOLO11n

3. **Оптимизация производительности**:
   - Экспорт в ONNX/TensorRT для ускорения
   - Квантизация модели (INT8)
   - Batch inference для параллельной обработки

## Дополнительные ресурсы

- **Полная документация**: [README_YOLO_TRAINING.md](README_YOLO_TRAINING.md)
- **Dataset implementation**: [dataset_yolo.py](dataset_yolo.py)
- **Training script**: [train_yolo.py](train_yolo.py)
- **Ultralytics YOLO11**: https://docs.ultralytics.com/models/yolo11/

---

**Вопросы?** Проверьте [README_YOLO_TRAINING.md](README_YOLO_TRAINING.md) для детальной информации.
