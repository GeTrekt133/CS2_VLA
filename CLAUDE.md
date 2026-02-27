# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS2 AI Agent — нейросетевой агент для игры в Counter-Strike 2. Обучается на демо-записях предсказывать действия игрока (движение мыши и нажатия клавиш) на основе визуальной информации + аудио.

## Architecture

### Active Implementation: `final_model/`

Итоговая архитектура ~50.8M параметров. Perception/Control ratio: **1.66x** ✅

```
Component                              Params    Status
───────────────────────────────────────────────────────
RadarEncoder (EffB0 blocks 1-5)        ~1.0M    trainable
YOLO (YOLOv11l backbone + FPN)         ~25.4M   FROZEN
YOLO embeds head                       ~1.6M    trainable
StereoAudioEncoder (MN10 / mn10_as)   ~4.88M   trainable  ← стерео
TemporalTransformer (d=384, L=4)       ~19.0M   trainable
FlowActionHead                         ~0.3M    trainable
───────────────────────────────────────────────────────
Total:     ~52.2M   |   Trainable: ~26.8M   |   Ratio: 1.72x ✅
```

### Component Details (`final_model/`)

#### 1. RadarEncoder (`RadarEncoder.py`)
- EfficientNet-B0 blocks 1–5 (~1.0M); раньше был 1-4 (~0.35M)
- Вход: `(B, 3, 224, 224)` — кропнутый радар
- Выход: `(B, 512)` — эмбеддинг
- Crop box: `(10, 25, 140, 170)` из оригинального кадра

#### 2. YOLO (`Yolo.py`) — YOLOv11l
- Backbone (слои 0–16) + FPN: **заморожены**
- Embed branch: P3 (layer 16, 256ch, 80×80) + P5 skip (layer 10, 512ch, 20×20)
  - UNet-like head: P3 down-to-20×20, skip-add P5 → InvRes → pool → `(B, 512)`
- Ключевые методы:
  - `extract_features(img)` — останавливается на layer 16, возвращает `(p3, p5)` → **кешируется**
  - `embed_from_features(p3, p5)` — trainable embed head → `(B, 512)`, **всегда с gradient**
  - `forward_detections_only(img)` — только детекции, без embed
- Детекции: 2 класса (`body=0`, `head=1`), nc=2

#### 3. StereoAudioEncoder (`AudioEncoder.py`) — MobileNetV3-Large (mn10_as)
- MobileNetV3-Large с `width_mult=1.0` — та же архитектура что EfficientAT mn10_as (~4.88M)
- **Вход: `(B, 2, T)` стерео @ 16kHz**, T=256,000 (16 sec) — оба канала для ILD/HRTF
- `StereoMelFrontend`: `(B, 2, T)` → `(B, 2, 128, 1600)` — 128 mel, hop=160 (10ms), нормализация per channel
- Backbone (без global pool): `(B, 2, 128, 1600)` → `(B, 960, 4, 50)`
  - 5 × stride-2 = 32x downsampling в обоих измерениях
  - 960 каналов — финальный `1×1 Conv` (160→960, = 6×160)
- Freq pool → `(B, 960, 50)`, Temporal pool 50→32 → `(B, 32, 960)`, Project → `(B, 32, 512)`
- → linspace до `(B, 16, 512)` в Train.py
- Pretrained веса: `load_pretrained_mn10as(path)` — адаптирует первый conv (1ch→2ch: w/2)
  - URL: `https://github.com/fschmid56/EfficientAT/releases/download/v0.0.1/mn10_as.pt`
  - Все остальные 16 слоёв грузятся as-is (AudioSet pretraining сохраняется)
- MACs: ~0.9B для 16sec стерео (vs QuartzNet ~1.6B моно)
- Зачем стерео: CS2 HRTF → шаги/выстрелы имеют L-R разницу → направление угрозы

#### 4. TemporalTransformer (`TemporalTransformer.py`) — Unified
- `d_model=384`, `num_heads=6`, `depth=4`, `ff_mult=4`, pre-LayerNorm
- **ModalityCompressor**: cross-attention сжатие 64→16 через 16 learnable query tokens
  - Используется для scene (64→16) и action (64→16)
- Все 6 модальностей конкатенируются → единый shared self-attention
- Итоговый контекст: **81 токен** (с audio) или **65 токен** (без audio):
  ```
  radar:     16 tokens  (32 raw → linspace 16)
  scene:     16 tokens  (64 raw → ModalityCompressor 64→16)
  audio:     16 tokens  (32 raw → linspace 16)   ← optional
  detection: 16 tokens  (64 raw → linspace 16)
  action:    16 tokens  (64 raw → ModalityCompressor 64→16)
  state:      1 token   (95-dim → proj 384)
  ─────────────────────────────────────────
  Total:     81 tokens (with audio)
  ```
- Выходы: `policy_mouse (2)`, `policy_keys (20)`, `value (1)`

#### 5. FlowActionHead (`TemporalTransformer.py`)
- Flow Matching для предсказания mouse delta (yaw, pitch)
- `context_dim=384`, `hidden_dim=256`, `noise_scale=0.3`

### Backbone Caching (ключевая оптимизация)

YOLO backbone (frozen) кешируется на диск/CPU для ускорения обучения в ~7x:

```python
# В Train.py:
with torch.no_grad():
    p3, p5 = yolo.extract_features(img_tensor)   # P3: float16, CPU → cache
    det_vec = get_det_vector(frame_path, yolo, ...)  # detection → cache

# Каждый шаг (с gradient):
scene_embed = yolo.embed_from_features(p3, p5)    # trainable, 1.6M params
```

- P3 features: `(N, 256, 80, 80)` → кешируется в float16 (≈6MB/frame)
- Detection vec: `(max_det * 5,)` = `(100,)` float32
- Ключ кеша: path к frame файлу (in-memory dict или h5)

### Data Flow

```
[Radar 224×224]  → RadarEncoder (trainable, 1.0M) → (B, 32, 512) → linspace → (B, 16, 512)
[Scene 640×640]  → YOLO.extract_features (frozen)  → p3, p5 cached
                 → YOLO.embed_from_features (trainable, 1.6M) → (B, 64, 512)
                                                               → ModalityCompressor → (B, 16, 512)
[Scene 640×640]  → YOLO.forward_detections_only (frozen) → (B, 64, 100) → linspace → (B, 16, 100)
[Audio 16sec stereo] → StereoAudioEncoder (trainable, 4.88M) → (B, 32, 512) → linspace → (B, 16, 512)
                       StereoMelFrontend → (B, 2, 128, 1600)
                       MN10 backbone (no global pool) → (B, 960, 4, 50)
                       freq pool + temporal pool → (B, 32, 960) → project → (B, 32, 512)
[Action history] → DatasetIntent → (B, 64, 22)
                                  → ModalityCompressor → (B, 16, 22)
[Game State]     → (B, 95)
                        ↓
            TemporalTransformer d=384, L=4
            (B, 81, 384) unified self-attention
                        ↓
    FlowActionHead → policy_mouse (2)   [flow matching]
    KeysHead       → policy_keys (20)   [BCEWithLogits]
    ValueHead      → value (1)          [for RL/critic]
```

### Sequence Dimensions (DatasetIntent.py)

```python
SEQ_LEN        = 16   # унифицированный размер всех последовательностей
SCENE_SEQ_LEN  = 64   # сцена: 64 кадра (~4 sec @ ~16Hz) → сжимается до 16
ACTION_SEQ_LEN = 64   # история действий: 64 окна (~4 sec) → сжимается до 16
```

| Модальность  | Raw              | After transform | Покрытие |
|--------------|------------------|-----------------|----------|
| radar        | 32 frames @1Hz   | linspace → 16   | 32 sec   |
| scene        | 64 frames @16Hz  | cross-attn → 16 | 4 sec    |
| audio        | 32 stereo embeds @0.5s | linspace → 16 | 16 sec |
| detection    | 64 frames @16Hz  | linspace → 16   | 4 sec    |
| action       | 64 windows       | cross-attn → 16 | 4 sec    |
| state        | 95 scalars       | proj → 1 token  | current  |

- `state_dim`: 95 (9 scalars + 2 side + 42 weapon + 42 weapon_list)
- `detection_dim`: 100 (20 detections × 5 features: x1,y1,x2,y2,conf)
- `action_dim`: 22 (2 mouse + 20 keys)

## Training

```bash
cd final_model && python Train.py
```

### Key Training Parameters

```python
BATCH_SIZE = 4
LR         = 1e-4
W_KEYS     = 100     # вес BCE loss для клавиш (класс имбаланс)
MAX_DET    = 20      # максимум детекций на кадр
SEED       = 42
```

### Loss Functions

| Выход        | Loss                  | Примечание                         |
|--------------|-----------------------|------------------------------------|
| mouse delta  | Flow Matching (MSE v) | yaw + pitch через velocity field   |
| keys (20)    | BCEWithLogitsLoss     | × W_KEYS=100 из-за редкости нажатий|
| value        | (будущий critic loss) | пока не используется в BC          |

### Frozen vs Trainable

```python
# FROZEN (не обновляются):
yolo.backbone         # слои 0–16 (~24M)
yolo.detect_head      # detect head (~1.4M)

# TRAINABLE:
yolo.embed_head       # ~1.6M   (UNet embed branch)
radar_encoder         # ~1.0M   (EffB0 blocks 1-5)
audio_encoder         # ~4.88M  (MN10 / mn10_as, stereo MobileNetV3)
temporal_model        # ~19.0M  (d=384, L=4)
flow_head             # ~0.3M   (FlowActionHead)
```

### Checkpoints

Сохраняются в `./checkpoints2/<run_name>/`:
- `radar_encoder`, `yolo`, `temporal_model`, `flow_head`, `audio_encoder`, `optimizer`
- Загрузка с `strict=False`

### Metrics (per epoch)

- `flow_loss` — основная mouse loss (flow matching)
- `bce_loss` — keys loss (weighted BCE)
- Per-class precision/recall для каждой из 20 клавиш (через sklearn если доступен)

## Legacy Code (audio_adaptation/src/)

Старая версия архитектуры (~42.8M), не используется в обучении:
- `TemporalCrossTransformer` — d=512, L=6, отдельные radar_encoder + scene_encoder (38M!)
- `scene_window=16` вместо 64 (только 1 sec покрытия)
- Нет AudioEncoder, нет ModalityCompressor
- Есть `ExactMatchBucketSampler` (не нужен в final_model — все seq фиксированного размера 16)

## Data Collection (data_collect_v2/)

Автоматический сбор данных из CS2 demo-записей: кадры + аудио + game state.

- **cs2_cmd.py** — прямой ввод команд в консоль CS2 через scan codes (без cfg/bind)
- **demo_control.py** — управление воспроизведением демо (playdemo, seek, spec_player)
- **capture.py** — захват экрана через WGC (Windows Graphics Capture)
- **audio.py** — запись аудио через WASAPI loopback (pyaudiowpatch) + ffmpeg resample
- **collector.py** — SingleWindowCollector: оркестрирует запись раундов
- **run.py** — CLI entry point (`--test`, `--list-devices`, `--validate`, сбор данных)
- **universal_demo_parser.py** — парсинг .dem файлов, извлечение 20 типов событий

Поддержка Sandboxie (`--sandbox CS2`) для изоляции инстансов.

### Known Bugs (не исправлены)

- `audio_adaptation/src/Train.py` строки 371 и 377: `audio_speedup_factor=4.0` → должно быть `1.0`

### Multi-Instance Bot Farm (future)

Для параллельного запуска 10-15 инстансов CS2 на сервере (RTX 3090 Ti, 256GB RAM):
- **Option A**: ASTER Multiseat — изолированные seats с ViGEm input
- **Option B**: RDP Multi-Session — concurrent RDP sessions с GPU acceleration
- Подробности: `memory/orchestration.md`
- ОС для фермы: Windows 10 IoT Enterprise LTSC 2021

## Important Notes

- Основной код для обучения — `final_model/`, не `audio_adaptation/src/`
- YOLO backbone заморожен, обучается только `yolo.embed_head` (~1.6M)
- Кешировать backbone features (P3/P5) на диск/CPU — критично для скорости
- `DataLoader` в `final_model` не требует `ExactMatchBucketSampler` (все seq = 16)
- Датасет требует JSON с путями к демо и состояниям

Запуск python: `C:/Users/misas/AppData/Local/Programs/Python/Python313/python.exe`
