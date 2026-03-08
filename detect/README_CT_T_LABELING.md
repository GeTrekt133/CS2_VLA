# CT/T Labeling для YOLO Dataset

Система автоматической разметки игроков Counter-Strike 2 на два класса: **CT (Counter-Terrorist)** и **T (Terrorist)** для обучения YOLO11n детектора.

## Обзор

Пайплайн состоит из следующих этапов:

1. **Парсинг демо** → `spotted_alive_players.csv`
2. **Кластеризация команд** → `team_mapping.json` (K-means по позициям спавна)
3. **Захват скриншотов** → `tick_{tick}_{user_id}.jpg`
4. **Детекция bbox** → YOLO аннотации с `class_id=0` (CT) или `class_id=1` (T)

## Быстрый старт

### Вариант 1: Автоматический пайплайн (рекомендуется)

```bash
python detect/run_pipeline.py
```

Этот скрипт выполнит все шаги автоматически.

### Вариант 2: Пошаговый запуск

#### Шаг 1: Парсинг демо (если ещё не сделано)

```bash
python detect/parse_demo.py
```

Создаёт `spotted_alive_players.csv` с позициями игроков.

#### Шаг 2: Кластеризация команд

```bash
python detect/team_classifier.py
```

Выполняет K-means кластеризацию игроков по начальным позициям спавна и создаёт:
- `team_mapping.json` - маппинг `steamid → команда`
- `team_clustering_visualization.png` - визуализация кластеров для проверки

**Важно**: Проверьте визуализацию! Если кластеризация выглядит неправильно, можно вручную отредактировать `team_mapping.json`.

#### Шаг 3: Захват скриншотов (если ещё не сделано)

```bash
python detect/screen_capture.py
```

Создаёт скриншоты в формате `tick_{tick}_{user_id}.jpg`.

#### Шаг 4: Детекция bbox с разметкой команд

```python
from detect.gt_find import process_batch

stats = process_batch(
    input_dir="detect/screenshots",
    output_dir="detect/screenshots_output",
    crop_size=322,
    save_annotations=True,
    csv_path="spotted_alive_players.csv",
    team_mapping_path="team_mapping.json"
)
```

Или используйте существующий код в `gt_find.py` (в `__main__` секции).

## Структура файлов

```
CS2_NN/
├── spotted_alive_players.csv          # Данные из демо
├── team_mapping.json                   # Маппинг steamid → команда
├── team_clustering_visualization.png  # Визуализация кластеров
└── detect/
    ├── parse_demo.py                  # Парсер демо (без изменений)
    ├── screen_capture.py              # Захват скриншотов (без изменений)
    ├── team_classifier.py             # [НОВЫЙ] Кластеризация команд
    ├── gt_find.py                     # [ИЗМЕНЁН] Детекция с CT/T метками
    ├── run_pipeline.py                # [НОВЫЙ] Автоматический пайплайн
    ├── screenshots/                   # Входные скриншоты
    │   └── tick_12345_5.jpg
    └── screenshots_output/            # Результаты
        ├── tick_12345_5_bbox.jpg      # Визуализация (фиолетовый bbox)
        └── tick_12345_5.txt           # YOLO аннотация
```

## Формат YOLO аннотаций

Файлы `.txt` содержат строки в формате:

```
<class_id> <x_center> <y_center> <width> <height>
```

Где:
- `class_id`: `0` для CT, `1` для T
- Координаты нормализованы в диапазоне [0, 1]

**Пример:**

```
0 0.523456 0.612345 0.145678 0.234567  # CT игрок
1 0.789012 0.345678 0.123456 0.198765  # T игрок
```

## Изменения в коде

### 1. team_classifier.py (новый файл)

Функции:
- `classify_teams_by_spawn(csv_path, initial_ticks=5)` - K-means кластеризация
- `visualize_clusters(...)` - визуализация результатов
- `save_team_mapping(team_mapping, output_path)` - сохранение в JSON
- `load_team_mapping(json_path)` - загрузка из JSON

### 2. gt_find.py (изменения)

**Добавлено:**
- `load_team_mapping(json_path)` - загрузка маппинга команд
- `extract_tick_userid_from_filename(filename)` - парсинг имени файла
- `lookup_steamid_from_csv(csv_df, tick, user_id)` - поиск steamid

**Изменено:**
- `detect_cyan_corners()`: добавлены параметры `team_mapping` и `steamid`
- Логика определения `class_id` на основе команды игрока
- Цвет bbox изменён с cyan `(0, 255, 255)` на purple `(255, 0, 246)`
- `process_batch()`: добавлена интеграция с CSV и team_mapping

### 3. Цвет bbox

**Было:** Голубой (cyan) `(0, 255, 255)` в BGR
**Стало:** Фиолетовый (purple) `(255, 0, 246)` в BGR

Соответствует `F600FFFF` в RGBA формате.

## Валидация результатов

### 1. Проверка кластеризации

Откройте `team_clustering_visualization.png`:
- **Синие точки** = CT (должны быть в одной зоне карты)
- **Красные точки** = T (должны быть в противоположной зоне)
- Игроки должны чётко разделиться на две группы

Если разделение неправильное:
1. Попробуйте увеличить `initial_ticks` в `team_classifier.py`
2. Вручную отредактируйте `team_mapping.json`

### 2. Проверка аннотаций

Случайно выберите несколько файлов `.txt`:

```bash
# Подсчёт классов
grep "^0 " detect/screenshots_output/*.txt | wc -l  # CT count
grep "^1 " detect/screenshots_output/*.txt | wc -l  # T count
```

Соотношение CT/T должно быть примерно 50/50 (зависит от карты и раунда).

### 3. Проверка визуализации

Откройте несколько `*_bbox.jpg` файлов:
- Цвет bbox должен быть **фиолетовым**
- Bbox должны плотно обрамлять персонажей

## Использование для обучения YOLO

### Создайте dataset.yaml:

```yaml
path: /path/to/CS2_NN/detect/screenshots_output
train: images
val: images

nc: 2
names:
  0: CT
  1: T
```

### Запустите обучение:

```bash
yolo train model=yolo11n.pt data=dataset.yaml epochs=100 imgsz=640
```

## Устранение проблем

### Проблема: "CSV file not found"
**Решение:** Запустите `parse_demo.py` для создания CSV

### Проблема: Неправильное разделение команд
**Решение:**
1. Увеличьте `initial_ticks` в `team_classifier.py`
2. Проверьте визуализацию
3. Вручную исправьте `team_mapping.json`

### Проблема: "No detection" для большинства изображений
**Решение:** Убедитесь, что на скриншотах видны cyan маркеры игроков (cl_ent_bbox в игре)

### Проблема: Steamid не найден в CSV
**Решение:** Проверьте формат имён файлов (`tick_{tick}_{user_id}.jpg`) и offset user_id (+1 в имени файла)

## Примеры использования

### Обработка отдельного изображения:

```python
from detect.gt_find import detect_cyan_corners, load_team_mapping

team_mapping = load_team_mapping("team_mapping.json")

annotation = detect_cyan_corners(
    image_path="screenshot.jpg",
    save_annotations=True,
    team_mapping=team_mapping,
    steamid=76561198123456789
)

print(annotation)  # "0 0.5 0.6 0.2 0.3" (CT) или "1 ..." (T)
```

### Пакетная обработка:

```python
from detect.gt_find import process_batch

stats = process_batch(
    input_dir="screenshots",
    output_dir="output",
    save_annotations=True,
    csv_path="spotted_alive_players.csv",
    team_mapping_path="team_mapping.json"
)

print(f"Success: {stats['success']}, Failed: {stats['failed']}")
```

## Параметры конфигурации

### team_classifier.py:

- `initial_ticks`: Количество начальных тиков для анализа (default: 5)
  - Меньше → быстрее, но менее точно
  - Больше → медленнее, но точнее (рекомендуется 5-10)

### gt_find.py:

- `crop_size`: Размер центрального кропа для детекции (default: 322)
- `save_annotations`: Сохранять ли `.txt` файлы (default: False)

## Формат team_mapping.json

```json
{
  "76561198386265483": "CT",
  "76561199012345678": "T",
  "76561198987654321": "CT",
  ...
}
```

Ключи: steamid (string)
Значения: "CT" или "T"

## Лицензия

Этот код является частью проекта CS2_NN для обучения нейросетевого агента Counter-Strike 2.
