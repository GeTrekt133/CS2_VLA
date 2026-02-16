# CS2 Dataset Collection Pipeline

Полноценная система для сбора данных из демо-записей CS2 для обучения **Behavioral Cloning** и **Offline RL**.

## Возможности

✅ **Запись видео + аудио** из демо (через OBS + WASAPI)
✅ **Парсинг демо** в JSON (states, actions, events)
✅ **Escape listener** - остановка по нажатию Escape
✅ **Нормальная скорость** - БЕЗ demo_timescale (правильная синхронизация аудио)
✅ **Offline RL events** - kills, deaths, damage, bomb, round outcomes
✅ **Универсальный формат** - готов для BC и RL обучения

---

## Установка

### 1. Python зависимости

```bash
pip install demoparser2 pandas tqdm obsws-python pyautogui keyboard
```

### 2. OBS Studio

1. Скачать [OBS Studio](https://obsproject.com/)
2. Установить [obs-websocket](https://github.com/obsproject/obs-websocket/releases) (обычно включен в OBS 28+)
3. Настроить WebSocket:
   - Tools → WebSocket Server Settings
   - Enable WebSocket server
   - Port: `4455`
   - Password: `secret` (или свой, укажи в коде)

### 3. Audio Capture

Убедись что установлен модуль `audio_capture.py` (уже есть в папке).

---

## Структура выходных данных

```
OUTPUT_DIR/
├── videos/              # Видео записи
│   └── demo_name/
│       ├── round_0.mp4
│       ├── round_1.mp4
│       └── ...
├── audio/               # Аудио (16kHz mono WAV)
│   └── demo_name/
│       ├── round_0.wav
│       ├── round_1.wav
│       └── ...
├── parsed/              # Парсинг каждой демки отдельно
│   ├── demo_1.json
│   ├── demo_2.json
│   └── ...
└── dataset.json         # Объединенный датасет (все демо)
```

### Формат JSON

```json
{
  "demos": [
    {
      "demo_name": "match_001",
      "player_steam_id": 76561198386265483,
      "rounds": [
        {
          "round_id": 0,
          "start_tick": 1000,
          "end_tick": 8500,
          "winner": "CT",
          "player_team": "T",
          "score": [3, 2],

          "events": [
            {"type": "kill", "tick": 1500, "victim_id": 123, "weapon": "ak47", "headshot": true},
            {"type": "death", "tick": 2000, "killer_id": 456, "weapon": "awp"},
            {"type": "damage_dealt", "tick": 1450, "victim_id": 123, "damage": 27},
            {"type": "bomb_plant", "tick": 5000},
            {"type": "round_end", "tick": 8500, "winner": "CT"}
          ],

          "states": [
            {
              "tick": 1000,
              "keys": ["W", "MOUSE_LEFT"],
              "mouse": [45.2, -10.5],
              "hp": 100,
              "armor": 100,
              "weapon": "ak-47",
              "ammo": 30,
              "side": "T",
              "ct_alive": 5,
              "t_alive": 5,
              "round_time_left": 115,
              "bomb_planted": false,
              "score": [3, 2]
            }
          ]
        }
      ]
    }
  ]
}
```

---

## Использование

### Вариант 1: Полный пайплайн (запись + парсинг)

```bash
python full_dataset_pipeline.py \
  --demo-dir "C:/Program Files (x86)/Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo" \
  --output-dir "D:/CS2_Dataset" \
  --steam-id 76561198386265483 \
  --record-video \
  --record-audio \
  --parse-demo
```

**Важно**: Нажми **Escape** в любой момент чтобы остановить запись!

---

### Вариант 2: Только запись (потом парсинг отдельно)

#### Шаг 1: Запись

```bash
python full_dataset_pipeline.py \
  --demo-dir "C:/path/to/demos" \
  --output-dir "D:/CS2_Dataset" \
  --steam-id 76561198386265483 \
  --record-video \
  --record-audio
```

#### Шаг 2: Парсинг (после записи)

```bash
python full_dataset_pipeline.py \
  --demo-dir "C:/path/to/demos" \
  --output-dir "D:/CS2_Dataset" \
  --steam-id 76561198386265483 \
  --parse-demo
```

---

### Вариант 3: Только парсинг (без записи)

Если у тебя уже есть демо и нужно только парсить:

```bash
python universal_demo_parser.py \
  --demo-dir "C:/path/to/demos" \
  --output-json "D:/CS2_Dataset/dataset.json" \
  --steam-id 76561198386265483 \
  --skip-rounds 3
```

---

## Параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--demo-dir` | Папка с .dem файлами | **Обязательно** |
| `--output-dir` | Папка для выходных данных | **Обязательно** |
| `--steam-id` | Steam ID целевого игрока | **Обязательно** |
| `--skip-rounds` | Пропустить первые N раундов (warmup) | `3` |
| `--start-index` | Начать с N-ой демки (для записи) | `0` |
| `--record-video` | Записывать видео (OBS) | `False` |
| `--record-audio` | Записывать аудио (WASAPI) | `False` |
| `--parse-demo` | Парсить демо в JSON | `False` |

---

## Как найти свой Steam ID?

1. Зайди на [steamid.io](https://steamid.io/)
2. Введи свой профиль Steam
3. Скопируй **steamID64** (например: `76561198386265483`)

---

## Events для Offline RL

Парсер автоматически собирает события для вычисления наград:

### Sparse Events (разреженные)

- `kill` - убийство врага
- `death` - смерть игрока
- `bomb_plant` - установка бомбы
- `bomb_defuse` - разминирование
- `round_end` - конец раунда (winner: CT/T)

### Dense Events (плотные)

- `damage_dealt` - нанесенный урон (tick, victim_id, damage, hitgroup)
- `damage_taken` - полученный урон (tick, attacker_id, damage, hitgroup)

---

## Примеры вычисления наград

После парсинга можно вычислить награды из `events`:

```python
# Sparse rewards
REWARD_KILL = 1.0
REWARD_DEATH = -1.0
REWARD_ROUND_WIN = 2.0

# Dense rewards
REWARD_DAMAGE = 0.01  # per 1 HP

# Example:
for event in round_data['events']:
    if event['type'] == 'kill':
        reward += REWARD_KILL
    elif event['type'] == 'death':
        reward += REWARD_DEATH
    elif event['type'] == 'damage_dealt':
        reward += REWARD_DAMAGE * event['damage']
```

---

## Troubleshooting

### OBS не подключается

1. Проверь что OBS запущен
2. Проверь настройки WebSocket (Tools → WebSocket Server Settings)
3. Проверь порт и пароль в коде

### Демо не записывается

1. Убедись что CS2 запущен
2. Проверь что путь к демке правильный
3. Проверь что Steam ID совпадает

### Аудио не синхронизировано

- ✅ **ИСПРАВЛЕНО**: Убрано `demo_timescale 4`
- Теперь записывается на нормальной скорости (x1)
- Аудио и видео синхронизированы с тиками игры

### Escape не останавливает

- Запускай скрипт **с правами администратора**
- `keyboard` модуль требует прав для глобальных хоткеев

---

## Интеграция с обучением

### Behavioral Cloning (BC)

```python
from src.DatasetIntent import DatasetIntent

# Используй парсинг напрямую
dataset = DatasetIntent(
    demo_list_path="D:/CS2_Dataset/dataset.json",
    is_train=True,
    radar_window=32,
    scene_window=16
)
```

### Offline RL

```python
from audio_adaptation.src.RewardCalculator import RewardCalculator

reward_calc = RewardCalculator()

# Вычисли награды из events
for round_data in demo['rounds']:
    events = round_data['events']
    rewards = reward_calc.compute_rewards(events)
```

---

## Следующие шаги

После сбора данных:

1. ✅ Обучить YOLO детекцию ([detect/train_yolo.py](../../detect/train_yolo.py))
2. ✅ Обучить BC модель ([audio_adaptation/src/Train.py](../src/Train.py))
3. ✅ Добавить Offline RL (см. план в [C:\Users\misas\.claude\plans\wiggly-chasing-pike.md](../../.claude/plans/wiggly-chasing-pike.md))

---

## Changelog

### v2.0 (2024-02-11)

- ✅ Убрано ускорение (demo_timescale 4 → 1)
- ✅ Добавлен Escape listener для остановки
- ✅ Парсинг events для offline RL
- ✅ Универсальный формат JSON (BC + RL)
- ✅ Правильная синхронизация аудио
- ✅ Полный пайплайн (запись + парсинг)

### v1.0 (старый dataset_collect.py)

- Только BC данные
- demo_timescale 4 (ускорение)
- Нет events для RL
- Нет Escape listener

---

## Лицензия

MIT
