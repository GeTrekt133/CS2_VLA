# Perception vs Control: ИСПРАВЛЕННЫЙ АНАЛИЗ (с QuartzNet)

## Текущее состояние (ПОЛНОЕ)

### Параметры по компонентам

```
RadarEncoder (EfficientNet-B0):     348,212 params (  0.7%)
YOLO (YOLOv11n):                  2,645,136 params (  5.6%)
  └─ YOLO embeds:                    21,056 params (  0.0%)
AudioEncoder (QuartzNet):         1,278,720 params (  2.7%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERCEPTION TOTAL:                 4,272,068 params (  9.1%)

TemporalTransformer (Control):   42,809,879 params ( 90.9%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL MODEL:                     47,081,947 params (100.0%)

Perception / Control Ratio: 0.10x  ⚠️ КРИТИЧЕСКИ НИЗКОЕ
```

## Приоритеты для CS2 (по важности)

### 🎯 1. YOLO (Scene Vision) - КРИТИЧЕСКИ ВАЖЕН ⭐⭐⭐⭐⭐

**Текущий:** YOLOv11n (2.6M params)
**Рекомендация:** YOLOv11m (20M params) - **8x increase**

**Почему это САМОЕ ВАЖНОЕ:**
- Главный источник визуальной информации о сцене
- Детектирует врагов (body + head)
- Распознает оружие, утилиты, объекты
- Feature map используется для scene embedding (2048-dim)
- В CS2 визуальная информация = 80% всех данных для принятия решений

**Текущие проблемы:**
- YOLOv11n — слишком маленький backbone для сложных сцен
- Только 21K trainable params в embeds
- Backbone полностью заморожен

---

### 🎯 2. AudioEncoder (QuartzNet) - ОЧЕНЬ ВАЖЕН ⭐⭐⭐⭐

**Текущий:** 1.3M params
**Рекомендация:** Оставить как есть или увеличить до 1.5-2M

**Почему это важно:**
- Footsteps (определение позиции врагов)
- Gunshots (тип оружия, направление)
- Bomb plant/defuse sounds
- Utility sounds (smokes, flashes, molotovs)
- Реакция на звук быстрее чем на визуал (часто слышишь раньше чем видишь)

**Текущее состояние:**
- QuartzNet architecture с 3 blocks (B1-B3)
- 60 embeddings @ 0.5 sec step (30 sec window)
- Архитектура адекватная, параметров достаточно

---

### 🎯 3. RadarEncoder - СРЕДНЯЯ ВАЖНОСТЬ ⭐⭐⭐

**Текущий:** EfficientNet-B0 (3 blocks) - 0.35M params
**Рекомендация:** EfficientNet-B0 (full) - 5M params - **14x increase**

**Почему менее критичен:**
- Радар — только overview карты и позиций teammate
- Не дает детальной информации о врагах
- Полезен для tactical awareness, но не для immediate action
- Можно принимать решения без радара, но не без YOLO

**Рекомендация:**
- Использовать полный EfficientNet-B0 (все 8 blocks)
- Небольшое увеличение (5M), но даст лучшие spatial features

---

### 🎯 4. TemporalTransformer - ОПТИМИЗИРОВАТЬ ⭐⭐

**Текущий:** d_model=512, depth=6, num_heads=8 → 42.8M params
**Рекомендация:** d_model=384, depth=4, num_heads=8 → ~24M params

**Почему слишком большой:**
- Текущие perception features слабые (4M total)
- Мощный transformer не может компенсировать плохие входные данные
- 90% параметров в control при 9% в perception — дисбаланс
- Depth=6 избыточен для temporal aggregation

---

## ПРАВИЛЬНАЯ РЕКОМЕНДАЦИЯ

### Целевая архитектура

```python
# === PERCEPTION (ПРИОРИТЕТЫ) ===

# 1. YOLO: YOLOv11m (ГЛАВНЫЙ ПРИОРИТЕТ)
yolo = DetectionModel(cfg='yolo11m.yaml', ch=3, nc=2)  # ~20M params

# Улучшенный embedding head
yolo.embeds = nn.Sequential(
    nn.Conv2d(256, 256, 3, padding=1, groups=256, bias=False),  # Depthwise
    nn.BatchNorm2d(256),
    nn.SiLU(),
    nn.Conv2d(256, 512, 1, bias=False),  # Pointwise expand
    nn.BatchNorm2d(512),
    nn.SiLU(),
    nn.Conv2d(512, 512, 3, padding=1, groups=512, bias=False),
    nn.BatchNorm2d(512),
    nn.SiLU(),
    nn.Conv2d(512, 1024, 1, bias=False),  # Final expand
    nn.AdaptiveAvgPool2d((4, 4)),
    nn.Flatten(),
    nn.Linear(1024 * 4 * 4, 2048),  # Match scene_dim
    nn.LayerNorm(2048)
)  # ~500K params в embeds

# Разморозить последние 3-4 layers YOLO для fine-tuning
# +500K trainable params

# YOLO TOTAL: ~21M params (~1M trainable из них)


# 2. AudioEncoder: QuartzNet (оставить как есть)
audio_encoder = AudioEncoder(
    sample_rate=16000,
    n_mels=64,
    embed_dim=512,
    target_embeddings=60
)  # 1.3M params


# 3. RadarEncoder: EfficientNet-B0 full
class RadarEncoderEffB0(nn.Module):
    def __init__(self, ...):
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Использовать ВСЕ блоки вместо первых 3
        self.blocks = backbone.features  # ALL blocks
        ...
# 5M params


# === PERCEPTION TOTAL: ~27M params ===


# === CONTROL (ОПТИМИЗАЦИЯ) ===

# 4. TemporalTransformer: оптимизированный
temporal = TemporalCrossTransformer(
    radar_dim=512, radar_seq=129,
    scene_dim=2048, scene_seq=16,
    detection_dim=100, detection_seq=1,
    actions_dim=22, actions_seq=16,
    state_dim=95,
    audio_dim=512, audio_seq=60,  # Добавить audio!
    d_model=384,    # ↓ from 512
    depth=4,        # ↓ from 6
    num_heads=8,
    ff_mult=4,
    dropout=0.1
)  # ~24M params

# === CONTROL TOTAL: ~24M params ===
```

### Итоговый результат

```
PERCEPTION:
  - YOLO (YOLOv11m):        21.0M (41%)  ⭐ ГЛАВНЫЙ
  - AudioEncoder:            1.3M ( 3%)  ⭐ footsteps/gunshots
  - RadarEncoder:            5.0M (10%)  ⭐ tactical awareness
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TOTAL PERCEPTION:         27.3M (53%)

CONTROL:
  - TemporalTransformer:    24.0M (47%)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TOTAL CONTROL:            24.0M (47%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL MODEL:                51.3M (100%)

Perception / Control Ratio: 1.14x  ✅ ОТЛИЧНО!
```

### Сравнение: До → После

| Компонент | Текущий | Рекомендуемый | Изменение |
|-----------|---------|---------------|-----------|
| **YOLO** | 2.6M (frozen) | 21.0M (1M trainable) | **+18.4M (+708%)** ⭐ |
| **AudioEncoder** | 1.3M | 1.3M | 0 |
| **RadarEncoder** | 0.35M | 5.0M | **+4.65M (+1329%)** |
| **TemporalTransformer** | 42.8M | 24.0M | **-18.8M (-44%)** |
| **TOTAL** | 47.1M | 51.3M | **+4.2M (+9%)** |
| **Perception %** | 9.1% | 53.2% | **+44.1%** ✅ |
| **Ratio** | 0.10x | 1.14x | **+1.04x** ✅ |

---

## Почему YOLOv11m, а не EfficientNet upgrade?

### Аргументы:

1. **CS2 = Visual Game**
   - 90% информации приходит из сцены (враги, оружие, карта)
   - Радар полезен, но вторичен

2. **YOLO дает:**
   - Детекции врагов (bbox + confidence)
   - Scene embedding (2048-dim feature map)
   - Spatial awareness
   - Object recognition

3. **YOLOv11m vs YOLOv11n:**
   - YOLOv11n: 2.6M params, 64-128-256 channels
   - YOLOv11m: ~20M params, 128-256-512 channels
   - Лучше детекция, лучше features, лучше generalization

4. **EfficientNet-B0 для радара достаточно:**
   - Радар = простая 2D карта (145x190)
   - Не требует столько параметров как scene
   - Full EfficientNet-B0 (5M) вполне достаточно

---

## Implementation Plan

### Этап 1: YOLO Upgrade (КРИТИЧЕСКИЙ) - 2-3 дня

```bash
# 1. Скачать YOLOv11m weights
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11m.pt

# 2. Создать yolo11m.yaml config
# (скопировать из ultralytics и адаптировать)

# 3. Обновить Yolo.py
# - Загрузка YOLOv11m вместо YOLOv11n
# - Улучшенный embeds head
# - Частичная разморозка последних layers

# 4. Обновить Train.py
# - Новый learning rate для YOLO layers
# - Warmup для YOLO fine-tuning
```

### Этап 2: Optimize TemporalTransformer - 1 день

```python
# Уменьшить d_model и depth
temporal = TemporalCrossTransformer(
    ...,
    d_model=384,  # was 512
    depth=4,      # was 6
    ...
)
```

### Этап 3: RadarEncoder Full - 1 день

```python
# Использовать полный EfficientNet-B0
self.blocks = backbone.features  # ALL blocks, not [1:5]
```

### Этап 4: YOLO Embeds Upgrade - 1 день

```python
# Более мощный embedding head (см. выше)
```

### Этап 5: YOLO Fine-tuning - 1 день

```python
# Разморозить последние layers для domain adaptation
def unfreeze_yolo_layers(yolo, num_layers=4):
    # Разморозить model[19-23] (последние C3k2 + Detect)
    ...
```

---

## Ожидаемые улучшения

### Метрики:

1. **Detection Quality:**
   - YOLOv11m: лучше детекция врагов (mAP ↑)
   - Меньше false positives/negatives

2. **Scene Understanding:**
   - Лучшие scene embeddings (2048-dim)
   - Богаче feature representations

3. **Generalization:**
   - Лучше работает на новых картах
   - Меньше overfitting (perception сильнее)

4. **Training Speed:**
   - Control легче (24M vs 43M) → быстрее обучение
   - Лучше gradient flow

5. **Action Quality:**
   - Более точные mouse movements
   - Лучшее key prediction (видит врагов раньше)

---

## Почему это правильно?

### Индустрия:

| Model | Main Perception | Control | Ratio |
|-------|----------------|---------|-------|
| **VPT** | ResNet (large) | Transformer 4L | 1.78x |
| **AlphaStar** | 3 Encoders (spatial++) | LSTM + Attn | 1.45x |
| **IMPALA** | ResNet 15L | Simple Policy | 2.00x |
| **Ваша (новая)** | YOLOv11m | Transformer 4L | **1.14x** ✅ |

### Философия:

> "Garbage in, garbage out"
>
> Даже самый мощный control не компенсирует слабые perception features.
>
> В играх (особенно FPS) визуальная информация критична.
> YOLO должен быть мощным, потому что он видит ВСЕ.

---

## TL;DR

### ❌ Старая рекомендация (НЕПРАВИЛЬНО):
- RadarEncoder: EfficientNet-B2/B3 (9-12M) ← слишком много для радара
- YOLO: YOLOv11n (2.6M) ← слишком мало для scene!

### ✅ Новая рекомендация (ПРАВИЛЬНО):
- **YOLO: YOLOv11m (21M)** ← ГЛАВНЫЙ ПРИОРИТЕТ ⭐
- AudioEncoder: 1.3M ← оставить
- RadarEncoder: EfficientNet-B0 full (5M) ← достаточно
- TemporalTransformer: оптимизировать до 24M

**Perception: 27M (53%) | Control: 24M (47%) | Ratio: 1.14x**

---

## Next Steps

1. ✅ Создать yolo11m.yaml config
2. ✅ Загрузить YOLOv11m pretrained weights
3. ✅ Обновить DetectionModel для YOLOv11m
4. ✅ Улучшить embeds head
5. ✅ Разморозить последние YOLO layers
6. ✅ Оптимизировать TemporalTransformer
7. ✅ Upgrade RadarEncoder to full EfficientNet-B0
8. 🔬 Обучить и сравнить метрики!

**Начинаем с YOLO upgrade — это ГЛАВНОЕ!**
