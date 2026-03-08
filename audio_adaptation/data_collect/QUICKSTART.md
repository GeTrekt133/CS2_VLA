# Quick Start Guide - CS2 Dataset Collection

Быстрый старт для сбора данных из CS2 демо.

---

## 1. Установка (1 минута)

```bash
cd audio_adaptation/data_collect

# Установить зависимости
pip install -r requirements.txt

# Тест что все работает
python test_pipeline.py
```

---

## 2. Настройка OBS (если нужно видео)

1. Скачать [OBS Studio](https://obsproject.com/)
2. Запустить OBS → Tools → WebSocket Server Settings
3. Включить WebSocket server
4. Порт: `4455`, Пароль: `secret`

---

## 3. Узнать свой Steam ID

Зайди на [steamid.io](https://steamid.io/) → введи профиль → скопируй **steamID64**

Пример: `76561198386265483`

---

## 4. Запуск (выбери вариант)

### Вариант A: Только парсинг (БЫСТРО, без записи)

Если у тебя уже есть .dem файлы и нужно только парсить в JSON:

```bash
python universal_demo_parser.py \
  --demo-dir "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo" \
  --output-json "D:\CS2_Dataset\dataset.json" \
  --steam-id 76561198386265483
```

**Результат**: JSON файл с states + events для BC и RL обучения.

---

### Вариант B: Запись + парсинг (ПОЛНЫЙ ДАТАСЕТ)

Если нужно записать видео + аудио + парсить:

```bash
python full_dataset_pipeline.py \
  --demo-dir "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo" \
  --output-dir "D:\CS2_Dataset" \
  --steam-id 76561198386265483 \
  --record-video \
  --record-audio \
  --parse-demo
```

**Важно**:
- Запусти CS2 перед этим
- Нажми **Escape** чтобы остановить в любой момент
- Записывается на **нормальной скорости** (БЕЗ ускорения)

---

## 5. Проверка результата

После парсинга проверь JSON:

```bash
python test_pipeline.py --json "D:\CS2_Dataset\dataset.json"
```

Должен вывести:
```
✓ 10 demos
✓ 120 rounds
✓ 15000 states
✓ 500 events
```

---

## 6. Использование в обучении

### Behavioral Cloning (BC)

```python
from src.DatasetIntent import DatasetIntent

dataset = DatasetIntent(
    demo_list_path="D:/CS2_Dataset/dataset.json",
    is_train=True,
    radar_window=32,
    scene_window=16
)
```

### Offline RL

```python
import json

# Загрузи данные
with open("D:/CS2_Dataset/dataset.json", 'r') as f:
    data = json.load(f)

# Извлеки события для наград
for demo in data['demos']:
    for round_data in demo['rounds']:
        events = round_data['events']  # Список: kill, death, damage, etc.

        # Вычисли награды
        for event in events:
            if event['type'] == 'kill':
                reward = 1.0
            elif event['type'] == 'death':
                reward = -1.0
            # и т.д.
```

---

## Troubleshooting

### ❌ Ошибка: `ModuleNotFoundError: No module named 'demoparser2'`

```bash
pip install demoparser2 pandas tqdm
```

### ❌ Ошибка: `OBS не подключается`

1. Запусти OBS
2. Проверь WebSocket настройки (Tools → WebSocket Server Settings)
3. Проверь порт/пароль

### ❌ Ошибка: `Escape не работает`

Запусти с правами администратора:
```bash
# Windows (PowerShell)
Start-Process python -ArgumentList "full_dataset_pipeline.py ..." -Verb RunAs
```

### ❌ Ошибка: `Demo file not found`

Проверь путь к демо папке:
```bash
dir "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\*.dem"
```

---

## Следующие шаги

После сбора данных:

1. ✅ Обучить YOLO: `python detect/train_yolo.py`
2. ✅ Обучить BC: `python audio_adaptation/src/Train.py`
3. ✅ Добавить RL (см. план)

---

## Структура файлов

```
audio_adaptation/data_collect/
├── universal_demo_parser.py        # Парсинг демо → JSON
├── screen_capture_with_audio.py    # Запись видео + аудио (OBS)
├── full_dataset_pipeline.py        # Полный пайплайн
├── test_pipeline.py                # Тесты
├── requirements.txt                # Зависимости
├── README.md                       # Полная документация
└── QUICKSTART.md                   # Этот файл
```

---

## Примеры команд

```bash
# Только парсинг (без записи)
python universal_demo_parser.py \
  --demo-dir "C:/demos" \
  --output-json "D:/dataset.json" \
  --steam-id 76561198386265483

# Запись + парсинг (полный пайплайн)
python full_dataset_pipeline.py \
  --demo-dir "C:/demos" \
  --output-dir "D:/CS2_Dataset" \
  --steam-id 76561198386265483 \
  --record-video --record-audio --parse-demo

# Только запись (потом парсинг отдельно)
python full_dataset_pipeline.py \
  --demo-dir "C:/demos" \
  --output-dir "D:/CS2_Dataset" \
  --steam-id 76561198386265483 \
  --record-video --record-audio

# Тест
python test_pipeline.py --demo "C:/demos/match.dem" --steam-id 76561198386265483
```

---

## FAQ

**Q: Сколько времени занимает запись одной демки?**

A: Зависит от длины матча. ~20-30 минут на full match (30 раундов) БЕЗ ускорения.

**Q: Можно ли ускорить?**

A: НЕТ! Иначе аудио не синхронизируется с тиками. Для быстрого датасета используй только парсинг без записи.

**Q: Зачем аудио если у меня уже есть видео?**

A: Аудио важно для spatial awareness (шаги врагов, выстрелы, и т.д.). Модель будет лучше понимать где враги.

**Q: Формат для offline RL готов?**

A: ✅ Да! Events (kill, death, damage, bomb) уже парсятся. Можно сразу вычислять rewards.

---

## Дополнительная помощь

- Полная документация: [README.md](README.md)
- План обучения: [wiggly-chasing-pike.md](../../.claude/plans/wiggly-chasing-pike.md)
- Проблемы: создай issue на GitHub
