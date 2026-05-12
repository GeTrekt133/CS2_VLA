# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS2 AI Agent — нейросетевой агент для игры в Counter-Strike 2. Обучается на демо-записях предсказывать действия игрока (движение мыши и нажатия клавиш) на основе визуальной информации + аудио.

## Architecture

### Active Implementation: `final_model_v3/`

Stage 3 — Hierarchical Joint Mouse+Keys with UnifiedActionHead.

```
Component                              Params    Status
───────────────────────────────────────────────────────
RadarEncoder (EffB0 blocks 1-5)        ~1.0M    trainable
YOLO (YOLOv11l backbone + FPN)         ~25.4M   FROZEN
YOLO embeds head                       ~1.6M    trainable
StereoAudioEncoder (MN10 / mn10_as)    ~4.88M   trainable
TemporalTransformer (d=384, L=4)       ~20.9M   trainable (register self-attn, no keys_head)
PlannerHead (joint mouse+keys traj)    ~0.53M   trainable
UnifiedActionHead (joint mouse+keys)   ~0.78M   trainable
───────────────────────────────────────────────────────
Total:     ~55.1M   |   Trainable: ~29.7M
```

### Hierarchical Architecture (Stage 3)

```
TemporalCrossTransformer
        ↓
unified_context (B, 81, 384)
        ↓
cross_attn: 3 register tokens → context
        ↓
register self-attention (3 → 3, cross-task coupling)
        ↓
(mouse_embed, keys_embed, value_embed)  — all (B, 384)
        ↓
┌───────┴──────────────────────┐
↓                              ↓
PlannerHead                UnifiedActionHead
concat(mouse, keys)        mouse_embed + keys_embed +
→ (B, 768)                 goal_attn(detections) +
→ MLP 768→512→272          planner_feat(512-dim)
→ (B, 16, 17)             → shared backbone → (B, 256)
  2 mouse +                ├→ mouse bins X (19) + Y (13)
  15 keys                  ├→ mouse offset regression
  (granades excluded)      └→ keys logits (20, all keys)
```

### Component Details (`final_model_v3/`)

#### 1. RadarEncoder (`RadarEncoder.py`)
- EfficientNet-B0 blocks 1–5 (~1.0M)
- Вход: `(B, 3, 224, 224)` — кропнутый радар
- Выход: `(B, 512)` — эмбеддинг

#### 2. YOLO (`Yolo.py`) — YOLOv11l
- Backbone (слои 0–16) + FPN: **заморожены** (~25.4M)
- Embed head: trainable (~1.6M), `(p3, p5) → (B, 512)`
- Train2GPU: backbone runs on GPU 0 (scorer process), embed head on GPU 1

#### 3. StereoAudioEncoder (`AudioEncoder.py`) — MobileNetV3-Large (mn10_as)
- ~4.88M, стерео вход `(B, 2, T)` @ 16kHz, T=256,000 (16 sec)
- Pretrained: `mn10_as.pt` (AudioSet), first conv adapted 1ch→2ch

#### 4. TemporalTransformer (`TemporalTransformer.py`) — ~20.9M
- `d_model=384`, `num_heads=8`, `depth=4`, `ff_mult=4`, pre-LayerNorm
- ModalityCompressor: cross-attention 64→16 (scene, action)
- 81 tokens unified self-attention (with audio)
- 3 register tokens (mouse/keys/value) → cross-attn to context → **register self-attention**
- Returns `(mouse_embed, keys_embed, value_embed)` — heads applied externally
- **Removed**: ModeHead, policy_keys_head, apply_keys_head

#### 5. PlannerHead (`TemporalTransformer.py`) — ~0.53M
- Joint trajectory: `concat(mouse_embed, keys_embed)` → `(B, 768)` → MLP → `(B, 16, 17)`
- 16 waypoints × (2 mouse raw cumulative + 15 keys binary)
- TRAJECTORY_KEYS: 15 keys (movement + combat + weapons + utility, granades excluded)
- Loss: SmoothL1 (mouse) + BCEWithLogits (keys) + smoothness penalty
- Gradient flows end-to-end (no detach) to controller

#### 6. UnifiedActionHead (`TemporalTransformer.py`) — ~0.78M
- Replaces DiscretizedMouseHead + policy_keys_head
- Inputs: mouse_embed(384) + keys_embed(384) + goal_attn(256) + planner_feat(512) = 1536
- Shared backbone: `Linear(1536, 256) → GELU → LayerNorm`
- Mouse: discretized bins (19 X, 13 Y) + per-bin offset regression
- Keys: `Linear(256, 384) → GELU → Linear(384, 20)` — all 20 keys including granades
- **Soft ordinal CE** (SORD, CVPR 2019): Gaussian soft labels σ_x=1.5, σ_y=0.75
- Balanced class weights + zero bin curriculum (0.5 → 0.06 over 150k steps)

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
| detection    | 64 frames @stride=T | linspace → 16 | T-synced |
| action       | 64 windows       | cross-attn → 16 | 4 sec    |
| state        | 100 scalars      | proj → 1 token  | current  |

- `state_dim`: 100 (12 scalars + 2 side + 43 weapon + 43 weapon_list)
  - 12 scalars: hp, armor, helmet, ammo, ct_alive, t_alive, round_time_left, bomb_planted, freeze_time, defuser, score_ct, score_t
- `detection_dim`: 100 (20 detections × 5 features: x1,y1,x2,y2,conf)
- `action_dim`: 22 (2 mouse + 20 keys)

## Training

```bash
cd final_model_v3 && python Train2GPU.py   # 2-GPU async pipeline
```

### Key Training Parameters

```python
BATCH_SIZE = 4
ACCUM_STEPS = 4       # effective BS=16
LR         = 2e-4     # temporal_model
LR_VISUAL  = 5e-4     # YOLO embed + radar
LR_AUDIO   = 1e-4     # audio encoder (pretrained, careful)
LR_MOUSE   = 1e-3     # unified_head + planner_head
MAX_DET    = 20
SEED       = 42
```

### T Parameter (Intent Shift / FPS Emulation)

- `T_min=1, T_max=6` → `allowed_T = [1, 2, 3, 4, 5, 6]`
- T=1 → 64 FPS, T=2 → 32 FPS, T=4 → 16 FPS, T=6 → ~10.7 FPS
- Randomly sampled per-sample (deterministic by idx)
- **Mouse delta normalized by T**: `target = delta / (T * MOUSE_SCALE)` — per-tick rate
- At inference: `predicted_rate * T_inference` to get actual delta
- Action history mouse also normalized by window size for consistency

### Loss Functions (Hierarchical)

```python
# Planner: dense trajectory supervision (primary anti-collapse signal)
L_planner = SmoothL1(mouse) + W_TRAJ_KEYS * BCE(keys) + W_SMOOTH * smoothness
# W_TRAJ_KEYS=1.0, W_SMOOTH=0.1

# Controller: current-tick prediction (joint mouse + keys)
L_ctrl_mouse = soft_ordinal_CE(bins_x, sigma_x=1.5) + soft_ordinal_CE(bins_y, sigma_y=0.75)
             + reg_weight * SmoothL1(offsets)
L_ctrl_keys  = focal_BCE(keys, gamma=1.5, pos_weight=balanced)

# Total
loss = W_PLANNER * L_planner + W_CTRL_MOUSE * L_ctrl_mouse + W_CTRL_KEYS * L_ctrl_keys
# W_PLANNER=1.0, W_CTRL_MOUSE=0.2, W_CTRL_KEYS=10.0
```

| Loss component | Approach | Details |
|---|---|---|
| Planner mouse | SmoothL1 | Raw cumulative trajectory, 16 waypoints |
| Planner keys | BCEWithLogits | 15 keys (granades excluded) |
| Controller mouse | Soft ordinal CE (SORD) | Gaussian soft labels, balanced weights, zero bin curriculum |
| Controller keys | Focal BCE | gamma=1.5, sqrt-dampened pos_weight cap=5 |
| Planner→Controller | End-to-end | No detach, gradient flows through planner_feat |

### Curriculum schedules

- **State dropout**: p_drop 1.0 → 0.0 (phase1=10k, ramp=50k)
- **Zero bin weight**: 0.5 → 0.06 (ramp=150k) — forces idle detection first
- **Regression weight**: 0.5 epoch 0 → 1.0 epoch 1+

### Frozen vs Trainable

```python
# FROZEN:
yolo.backbone         # слои 0–16 (~24M)
yolo.detect_head      # detect head (~1.4M)

# TRAINABLE:
yolo.embed_head       # ~1.6M   (UNet embed branch)
radar_encoder         # ~1.0M   (EffB0 blocks 1-5)
audio_encoder         # ~4.88M  (MN10 / mn10_as, stereo)
temporal_model        # ~20.9M  (d=384, L=4, register self-attn)
planner_head          # ~0.53M  (joint trajectory MLP)
unified_head          # ~0.78M  (joint mouse bins + keys)
```

### Mixed Precision (AMP)

- `torch.amp.GradScaler` + `torch.amp.autocast(device_type='cuda', dtype=torch.float16)`
- Training: autocast around forward+loss, scaler for backward/step
- Evaluation: autocast only (no scaler needed under no_grad)
- Fallback: CPU training works without AMP (scaler=None)

### DataLoader

- `num_workers=2, persistent_workers=True` — параллельная загрузка данных
- Custom `collate_fn`: стакает тензоры, собирает lists для paths/strings

### Checkpoints

Сохраняются в `./checkpoints_final/<run_name>/checkpoints/`:
- `radar_encoder`, `yolo`, `temporal_model`, `unified_head`, `planner_head`, `audio_encoder`, `optimizer`, `scheduler`
- Загрузка с `strict=False`

### Metrics & Plots

Логи: `./checkpoints_final/<run_name>/logs/metrics.txt`
Графики: `./checkpoints_final/<run_name>/plots/step_XXXXXX_{train|val}/`

Каждые METRICS_EVERY (5k) шагов (train) и EVAL_EVERY (30k) шагов (val):
- `bin_distribution.png` — GT vs Pred bin distribution
- `keys_precision_recall.png` — per-key P/R bar chart
- `mouse_scatter.png` — predicted vs GT delta scatter

Text metrics:
- `acc@1/2/3` — accuracy within ±K bins (ordinal-aware)
- `mae_bin` — mean absolute error in bin index space
- `move_detect` — binary accuracy: idle vs movement detection
- Per-bin accuracy, entropy, MSE

## Legacy Code

### `final_model/` — Stage 1 (FlowActionHead)

Предыдущая архитектура с FlowActionHead для mouse prediction (flow matching).
Работала, но заменена на hierarchical Stage 3 для joint mouse+keys prediction.

### `audio_adaptation/src/` — Old baseline

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

- Основной код для обучения — `final_model_v3/`, не `final_model/` или `audio_adaptation/src/`
- YOLO backbone заморожен, обучается только `yolo.embed_head` (~1.6M)
- Кешировать backbone features (P3/P5) на диск/CPU — критично для скорости
- `DataLoader` в `final_model` не требует `ExactMatchBucketSampler` (все seq = 16)
- Датасет требует JSON с путями к демо и состояниям

Запуск python: `C:/Users/misas/AppData/Local/Programs/Python/Python313/python.exe`
