# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS2 AI Agent — нейросетевой агент для игры в Counter-Strike 2. Обучается на демо-записях предсказывать действия игрока (движение мыши и нажатия клавиш) на основе визуальной информации.

## Architecture

### Core Components (src/)

1. **RadarEncoder.py** — EfficientNet-B0 энкодер для миникарты (радара)
   - Вход: (B, 3, 224, 224) — кропнутый радар
   - Выход: (B, 512) — эмбеддинг

2. **Yolo.py** — YOLOv11 детектор + эмбеддер сцены
   - `DetectionModel` — основная модель с двумя выходами:
     - детекции (bbox персонажей)
     - эмбеддинг сцены (B, 2048) через `self.embeds`
   - Конфигурация в `yolo11n.yaml`

3. **TemporalTransformer.py** — `TemporalCrossTransformer`
   - Агрегирует: radar_seq, scene_seq, detections, action_history, state
   - Cross-attention с register tokens
   - Выходы: policy_mouse (2), policy_keys (20), value (1)

4. **DatasetIntent.py** — основной датасет (используется в обучении)
   - Формирует intent-based таргеты с агрегацией по временному окну T
   - Ключевые параметры: `radar_window=128`, `scene_window=16`, `actions_window=16`

5. **Dataset.py** — альтернативный датасет (tick-level, не используется)

6. **Dataloader.py** — `ExactMatchBucketSampler` для батчинга сэмплов с одинаковой длиной последовательностей

### Data Flow

```
[Radar Images] → RadarEncoder → radar_embeds (B, T_radar, 512)
[Scene Images] → YOLO.embeds → scene_embeds (B, T_scene, 2048)
[Scene Images] → YOLO.detect → detections (B, 1, 100)
[Action History] → action_seq (B, 16, 22)
[Game State] → state_vec (B, 95)
                    ↓
            TemporalCrossTransformer
                    ↓
    policy_mouse, policy_keys, value
```

### Key Dimensions

- `radar_seq`: max 129 frames (128 * 64 ticks window + current)
- `scene_seq`: 16 frames
- `actions_seq`: 16 intent windows
- `state_dim`: 95 (9 scalars + 2 side + 42 weapon + 42 weapon_list)
- `detection_dim`: 100 (20 detections * 5 features)

## Training

```bash
python src/Train.py
```

### Key Training Parameters (in Train.py)

- `BATCH_SIZE = 4`
- `LR = 1e-4`
- `W_KEYS = 100` — вес BCE loss для клавиш
- `MAX_DET = 20` — максимум детекций

### Loss Functions

- MSE для mouse delta (yaw, pitch)
- BCEWithLogitsLoss для intent клавиш (20 действий)

### Checkpoints

Сохраняются в `./checkpoints2/<run_name>/`:
- `radar_encoder`, `yolo`, `temporal_model`, `optimizer`

## Data Collection (data_collect_v2/)

Автоматический сбор данных из CS2 demo-записей: кадры + аудио + game state.

- **cs2_cmd.py** — прямой ввод команд в консоль CS2 через scan codes (без cfg/bind)
- **demo_control.py** — управление воспроизведением демо (playdemo, seek, spec_player)
- **capture.py** — захват экрана через WGC (Windows Graphics Capture)
- **audio.py** — запись аудио через WASAPI loopback (pyaudiowpatch) + ffmpeg resample
- **collector.py** — SingleWindowCollector: оркестрирует запись раундов
- **run.py** — CLI entry point (`--test`, `--list-devices`, `--validate`, сбор данных)

Поддержка Sandboxie (`--sandbox CS2`) для изоляции инстансов.

### Multi-Instance Bot Farm (future)

Для параллельного запуска 10-15 инстансов CS2 на сервере (RTX 3090 Ti, 256GB RAM):
- **Option A**: ASTER Multiseat — изолированные seats с ViGEm input
- **Option B**: RDP Multi-Session — concurrent RDP sessions с GPU acceleration
- Подробности: см. memory/orchestration.md
- ОС для фермы: Windows 10 IoT Enterprise LTSC 2021

## Important Notes

- YOLO backbone заморожен, обучается только `yolo.embeds`
- При загрузке чекпоинтов используется `strict=False`
- Датасет требует JSON с путями к демо и состояниями

Запуск python осуществляется командой - `C:/Users/misas/AppData/Local/Programs/Python/Python313/python.exe`
