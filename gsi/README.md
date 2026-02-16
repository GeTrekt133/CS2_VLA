# CS2 Game State Integration (GSI)

Модуль для получения данных из CS2 в реальном времени через Game State Integration.

## Структура

```
gsi/
├── config/
│   ├── gsi_config.py                    # Конфигурация и константы
│   └── gamestate_integration_cs2nn.cfg  # Конфиг для CS2
├── server/
│   └── http_server.py                   # HTTP сервер для приёма GSI данных
├── adapter/
│   └── state_adapter.py                 # Конвертация GSI → state_vec (95)
├── buffer/
│   └── state_buffer.py                  # Буферы для состояний и кадров
├── collector/
│   └── data_collector.py                # Сбор данных для обучения
├── inference/
│   └── inference_engine.py              # Инференс в реальном времени
├── run_test.py                          # Тест GSI подключения
├── run_collector.py                     # Запуск сбора данных
├── run_inference.py                     # Запуск инференса
└── install_config.py                    # Установка конфига в CS2
```

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install mss opencv-python numpy torch
```

### 2. Установка GSI конфига в CS2

**Автоматически:**
```bash
python gsi/install_config.py
```

**Вручную:**
Скопируйте файл `gsi/config/gamestate_integration_cs2nn.cfg` в папку:
```
C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg\
```

**Перезапустите CS2** после установки конфига.

### 3. Проверка работы GSI

```bash
python gsi/run_test.py
```

Запустите CS2, зайдите в матч — вы должны увидеть данные о состоянии игры.

## Режимы работы

### Сбор данных (для обучения)

```bash
# Базовый запуск
python gsi/run_collector.py

# С параметрами
python gsi/run_collector.py --output ./my_data --port 3000

# Без скриншотов (только game state)
python gsi/run_collector.py --no-screenshots
```

Данные сохраняются в формате `train_dataset.json`, совместимом с `DatasetIntent.py`.

### Инференс (запуск модели)

```bash
python gsi/run_inference.py --checkpoint ./checkpoints2/run_xxx/epoch_10.pth

# На CPU
python gsi/run_inference.py --checkpoint ./checkpoint.pth --device cpu
```

## Использование в коде

### Простой GSI сервер

```python
from gsi.server.http_server import GSIServer, GSIPayload
from gsi.adapter.state_adapter import GSIStateAdapter

def on_update(payload: GSIPayload):
    adapter = GSIStateAdapter()
    snapshot = adapter.parse(payload)

    if snapshot:
        print(f"HP: {snapshot.hp}, Weapon: {snapshot.weapon}")

        # Получить state_vec для модели
        state_vec = adapter.to_state_vec(snapshot)
        print(f"State vec shape: {state_vec.shape}")  # (95,)

server = GSIServer(port=3000)
server.register_callback(on_update)
server.start()

# ... игра ...

server.stop()
```

### Сбор данных

```python
from gsi.collector.data_collector import GSIDataCollector
from gsi.config.gsi_config import GSIConfig

config = GSIConfig(
    output_dir="./collected_data",
    save_screenshots=True,
    screenshot_interval=4,  # каждые 4 тика
)

collector = GSIDataCollector(config)
collector.start()

# ... играем в CS2 ...
# Ctrl+C или:

collector.stop()
collector.save()  # Сохранить в dataset.json
```

### Инференс

```python
from gsi.inference.inference_engine import GSIInferenceEngine, InferenceResult

def on_prediction(result: InferenceResult):
    print(f"Mouse: {result.mouse_delta}")
    print(f"Keys: {result.predicted_keys}")
    print(f"Value: {result.value:.2f}")

engine = GSIInferenceEngine(
    checkpoint_path="./checkpoints2/run_xxx/epoch_10.pth",
    device="cuda"
)
engine.register_callback(on_prediction)
engine.start()

# ... игра ...

engine.stop()
```

## Формат данных

### GameStateSnapshot

```python
@dataclass
class GameStateSnapshot:
    timestamp: float
    tick: int

    # Скаляры (9)
    hp: int           # 0-100
    armor: int        # 0-100
    helmet: bool
    ammo: int         # текущая обойма
    ct_alive: int     # 0-5
    t_alive: int      # 0-5
    round_time_left: float  # секунды
    bomb_planted: bool
    freeze_time: bool

    # Категориальные
    side: str         # "CT" или "T"
    weapon: str       # "ak-47", "m4a4", etc.
    weapon_list: List[str]  # инвентарь

    # Дополнительно
    round_num: int
    score_ct: int
    score_t: int
    phase: str
    money: int
```

### state_vec (95 параметров)

```
[0:9]   - hp, armor, helmet, ammo, ct_alive, t_alive,
          round_time_left, bomb_planted, freeze_time
[9:11]  - side one-hot [CT, T]
[11:53] - weapon one-hot (42 класса)
[53:95] - weapon_list multi-hot (42 класса)
```

## Ограничения GSI

| Данные | Доступно | Примечание |
|--------|----------|------------|
| HP, Armor, Helmet | Да | |
| Weapon, Ammo | Да | |
| Inventory | Да | |
| Side (CT/T) | Да | |
| Alive players | Да | через allplayers |
| Round time | Да | phase_countdowns |
| Bomb status | Да | |
| Freeze time | Да | |
| **Mouse position** | **Нет** | GSI не предоставляет |
| **Key presses** | **Нет** | GSI не предоставляет |
| **Exact tick** | **Нет** | Эстимация из timestamps |

**Важно:** GSI не предоставляет данные о нажатиях клавиш и положении мыши.
Собранные через GSI данные подходят для state-based анализа, но не для точного
imitation learning действий.

## Troubleshooting

### GSI не работает

1. Убедитесь, что конфиг установлен в правильную папку
2. Перезапустите CS2 после установки конфига
3. Проверьте, что порт 3000 не занят:
   ```bash
   netstat -an | findstr 3000
   ```

### Нет скриншотов

1. Установите mss: `pip install mss`
2. Запустите от имени администратора

### Модель не загружается

1. Проверьте путь к чекпоинту
2. Убедитесь, что есть CUDA (или используйте `--device cpu`)

## Совместимость

- Windows 10/11
- CS2 (Counter-Strike 2)
- Python 3.10+
- PyTorch 2.0+
