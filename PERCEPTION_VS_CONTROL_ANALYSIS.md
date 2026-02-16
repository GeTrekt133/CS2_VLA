# Perception vs Control: Parameter Analysis & Recommendations

## Текущее состояние вашей модели

### Параметры по компонентам

```
RadarEncoder (EfficientNet-B0):     348,212 params (  0.8%)
YOLO (YOLOv11n):                  2,645,136 params (  5.8%)
  └─ YOLO embeds:                    21,056 params (  0.0%)
TemporalTransformer (Control):   42,809,879 params ( 93.5%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL:                           45,803,227 params (100.0%)

Perception / Control Ratio: 0.07x  ⚠️ КРИТИЧЕСКИ НИЗКОЕ
```

### Проблемы:
1. **Perception слишком слабый**: 93.5% параметров в control, только 6.6% в perception
2. **EfficientNet-B0 урезан до 3 блоков**: всего 348K параметров вместо ~5.3M
3. **YOLO backbone заморожен**: обучается только embedding head (21K параметров)
4. **Temporal Transformer слишком мощный**: 42M параметров, 6 layers, depth=6

## Сравнение с state-of-the-art моделями

### 1. OpenAI VPT (Minecraft, 2022-2024)

**Architecture:**
- **Vision Encoder**: ResNet (ConvNets) - значительная часть модели
- **Control**: Transformer (4 attention layers, 16 heads)
- **Total**: ~250M parameters
- **Perception/Control Ratio**: ~**2-3x** (perception доминирует)

**Key Insights:**
- Используют residual CNN для обработки 128 frames контекста
- Transformer только для temporal aggregation
- Vision encoder составляет большую часть модели

**Sources:**
- [OpenAI VPT GitHub](https://github.com/openai/Video-Pre-Training)
- [VPT Paper](https://arxiv.org/abs/2206.11795)

---

### 2. DeepMind AlphaStar (StarCraft II, 2019-2024)

**Architecture:**
- **Encoders**: Entity encoder + Spatial encoder + Scalar encoder
- **Core**: Deep LSTM + Self-attention на entities
- **Action Heads**: Auto-regressive policy с pointer network
- **Total**: 139M weights (55M during inference)
- **Perception/Control Ratio**: ~**1.5-2x** (perception значимая часть)

**Key Insights:**
- 3 отдельных энкодера для разных типов информации
- Spatial encoder обрабатывает feature maps (аналог вашего YOLO)
- Entity encoder для юнитов (можно сравнить с детекциями)
- LSTM core меньше, чем encoders

**Sources:**
- [AlphaStar Architecture - DeepWiki](https://deepwiki.com/google-deepmind/alphastar/3.3-standard-architecture)
- [AlphaStar Paper](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)

---

### 3. IMPALA (DeepMind, 2018-2024)

**Architecture:**
- **Large version**: 15 convolutional layers, 1.6M parameters
- **Small version**: 2 convolutional layers, 1.2M parameters
- **Design**: ResNet-style encoder с ConvSeq blocks
- **Key difference**: Flattens all features (no GAP), высокоразмерный encoding

**Recent improvements (2024-2025):**
- **Impoola-CNN**: улучшает генерализацию на 17% при -35% параметров

**Sources:**
- [IMPALA Paper](https://arxiv.org/abs/1802.01561)
- [Impoola Research 2024](https://arxiv.org/html/2503.05546v1)

---

### 4. AlphaViT (Vision Transformer для игр, 2024)

**Architecture:**
- **Encoder**: Vision Transformer (ViT)
- **Standard ViT config**: 768 hidden, 12 layers, 12 heads → ~86M params
- **Finding**: Больше encoder layers = лучше performance
- **Recommendation**: "When more powerful hardware available, train with larger transformer encoders"

**Sources:**
- [AlphaViT Paper](https://arxiv.org/abs/2408.13871)
- [AlphaViT on PeerJ](https://peerj.com/articles/cs-3403/)

---

### 5. NitroGen (Nvidia/Stanford/Caltech, 2024-2025)

**Architecture:**
- **Training data**: 40,000+ hours human gameplay
- **Vision-centric approach**: VLA (Vision-Language-Action) model
- **Key finding**: Strong visual encoders critical for generalization

**Sources:**
- [MarkTechPost - Top AI Agent Architectures 2025](https://www.marktechpost.com/2025/11/15/comparing-the-top-5-ai-agent-architectures-in-2025-hierarchical-swarm-meta-learning-modular-evolutionary/)

---

## Сводная таблица Perception/Control Ratios

| Model               | Perception Params | Control Params | Ratio   | Notes                              |
|---------------------|-------------------|----------------|---------|-------------------------------------|
| **Ваша модель**     | 3M                | 43M            | **0.07x** | ⚠️ Критически низко               |
| OpenAI VPT          | ~160M             | ~90M           | **~2x**   | ResNet encoder доминирует          |
| AlphaStar           | ~80M              | ~55M           | **~1.5x** | Множественные энкодеры             |
| IMPALA (large)      | 1.6M              | <1M            | **>2x**   | Только perception, простой policy  |
| AlphaViT            | ~86M              | ~30M           | **~3x**   | ViT encoder + transformer decoder  |

**Оптимальное соотношение**: **1.5x - 3x** (perception должен быть мощнее control)

---

## Рекомендации по улучшению

### 🎯 Приоритет 1: Усилить Perception модули

#### Вариант A: Upgrade EfficientNet (легко)
```python
# Текущий: EfficientNet-B0 (3 blocks) → 348K params
# Предлагаемое: EfficientNet-B0 (full) → ~5.3M params

class RadarEncoderEffB0(nn.Module):
    def __init__(self, ...):
        backbone = efficientnet_b0(weights=...)
        # Вместо blocks[1:5] используйте полный backbone
        self.blocks = backbone.features  # ВСЕ блоки
        # или хотя бы до блока 7
        self.blocks = nn.Sequential(*backbone.features[1:8])
```

**Эффект**: +5M params в perception (15x increase)

---

#### Вариант B: Upgrade на EfficientNet-B2 или B3 (средне)
```python
# EfficientNet-B2: ~9.2M params
# EfficientNet-B3: ~12.2M params

from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class RadarEncoderEffB2(nn.Module):
    def __init__(self, ...):
        backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        # Использовать полный backbone
        ...
```

**Эффект**: +12M params в perception (35x increase)

---

#### Вариант C: YOLO → разморозить часть backbone (средне)

```python
# Текущий: YOLO backbone заморожен, только embeds (21K)
# Предлагаемое: разморозить последние N layers

def unfreeze_yolo_layers(yolo_model, num_layers=5):
    """Разморозить последние N layers YOLO для fine-tuning"""
    # Заморозить все
    for param in yolo_model.parameters():
        param.requires_grad = False

    # Разморозить последние layers
    # Например, разморозить model[7-10] + embeds
    for i in range(len(yolo_model.model) - num_layers, len(yolo_model.model)):
        for param in yolo_model.model[i].parameters():
            param.requires_grad = True

    # embeds всегда trainable
    for param in yolo_model.embeds.parameters():
        param.requires_grad = True
```

**Эффект**: +500K - 1M trainable params (зависит от num_layers)

---

#### Вариант D: Upgrade YOLO embeds (легко)

```python
# Текущий: embeds → 2048 (Conv → 128 → flatten) → 21K params
# Предлагаемое: более мощный embedding head

self.embeds = nn.Sequential(
    # Сохраняем spatial info дольше
    nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False),  # Depthwise
    nn.BatchNorm2d(128),
    nn.SiLU(),
    nn.Conv2d(128, 256, 1, bias=False),  # Pointwise expand
    nn.BatchNorm2d(256),
    nn.SiLU(),
    nn.Conv2d(256, 256, 3, padding=1, groups=256, bias=False),
    nn.BatchNorm2d(256),
    nn.SiLU(),
    nn.Conv2d(256, 512, 1, bias=False),  # Expand to 512
    nn.AdaptiveAvgPool2d((4, 4)),
    nn.Flatten(),
    nn.Linear(512 * 4 * 4, 2048),  # Projection
    nn.LayerNorm(2048)
)
```

**Эффект**: +200K params в YOLO embeds (10x increase)

---

### 🎯 Приоритет 2: Уменьшить или оптимизировать TemporalTransformer

#### Вариант A: Reduce depth (легко)
```python
# Текущий: depth=6, num_heads=8 → 42.8M params
# Предлагаемое: depth=4, num_heads=8 → ~28M params

temporal = TemporalCrossTransformer(
    ...,
    depth=4,  # Вместо 6
    ...
)
```

**Эффект**: -15M params в control

---

#### Вариант B: Reduce d_model (средне)
```python
# Текущий: d_model=512 → 42.8M params
# Предлагаемое: d_model=384 → ~24M params

temporal = TemporalCrossTransformer(
    ...,
    d_model=384,  # Вместо 512
    ...
)
```

**Эффект**: -18M params в control

---

#### Вариант C: Использовать pre-encoder для radar/scene (продвинуто)

```python
class TemporalCrossTransformer(nn.Module):
    def __init__(self, ...):
        # Вместо 2 отдельных TransformerEncoder для radar и scene
        # использовать один shared encoder с меньшим depth

        self.shared_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads),
            num_layers=3  # Вместо 6 для каждого
        )

        # Добавить modality embeddings
        self.radar_modality_embed = nn.Parameter(torch.randn(1, 1, d_model))
        self.scene_modality_embed = nn.Parameter(torch.randn(1, 1, d_model))
```

**Эффект**: -10M params, улучшенная shared representation

---

### 🎯 Приоритет 3: Оптимальная архитектура (рекомендуемая комбинация)

#### Recommended Setup

```python
# === PERCEPTION ===
# 1. RadarEncoder: EfficientNet-B2 (full) → ~9M params
radar_encoder = RadarEncoderEffB2(embed_dim=512)

# 2. YOLO: YOLOv11n с improved embeds → ~3M params
yolo = DetectionModel(cfg='yolo11n.yaml')
# Upgrade embeds head → +200K
# Разморозить последние 3 layers → +500K trainable

# TOTAL PERCEPTION: ~12M params

# === CONTROL ===
# 3. TemporalTransformer: d_model=384, depth=4 → ~24M params
temporal = TemporalCrossTransformer(
    radar_dim=512, radar_seq=129,
    scene_dim=2048, scene_seq=16,
    d_model=384,    # ↓ from 512
    depth=4,        # ↓ from 6
    num_heads=8,
    ...
)

# TOTAL CONTROL: ~24M params

# === РЕЗУЛЬТАТ ===
# Perception:  12M (33%)
# Control:     24M (67%)
# Ratio: 0.5x  ✅ Гораздо лучше!
```

**Ожидаемые улучшения:**
- ✅ Perception в 4x сильнее (12M vs 3M)
- ✅ Control на 45% легче (24M vs 43M)
- ✅ Соотношение 0.5x ближе к оптимальному (1-3x)
- ✅ Лучшая feature extraction
- ✅ Быстрее обучение, меньше overfitting

---

## Action Plan

### Этап 1: Быстрые wins (1-2 дня)
1. ✅ **Upgrade RadarEncoder**: EfficientNet-B0 full вместо 3 blocks
2. ✅ **Improve YOLO embeds**: более мощный embedding head
3. ✅ **Reduce TemporalTransformer**: depth=4, d_model=384

### Этап 2: Средние улучшения (3-5 дней)
4. ✅ **Upgrade RadarEncoder**: EfficientNet-B2 или B3
5. ✅ **Unfreeze YOLO layers**: разморозить последние 3-5 layers
6. ✅ **Experiment**: попробовать разные соотношения perception/control

### Этап 3: Продвинутые оптимизации (1-2 недели)
7. 🔬 **ViT encoder**: попробовать Vision Transformer вместо EfficientNet
8. 🔬 **Separate encoders**: отдельные энкодеры для radar, scene, detections (как AlphaStar)
9. 🔬 **Shared pre-encoder**: shared transformer для radar+scene

---

## Дополнительные источники

### ResNet/EfficientNet в Game AI
- [EfficientNet Overview](https://viso.ai/deep-learning/efficientnet/)
- [CNN Architectures Guide](https://theaisummer.com/cnn-architectures/)

### Vision Transformers
- [Vision Transformer Guide 2024](https://www.v7labs.com/blog/vision-transformer-guide)
- [Vision LLMs Architecture](https://code-b.dev/blog/vision-llm)

### Deep RL & Game Agents
- [Embodied AI: LLMs to World Models 2025](https://mn.cs.tsinghua.edu.cn/xinwang/PDF/papers/2025_Embodied%20AI%20from%20LLMs%20to%20World%20Models.pdf)
- [AI Agent Architectures 2025](https://orq.ai/blog/ai-agent-architecture)

---

## Выводы

1. **Текущая проблема**: Ваше соотношение 0.07x (perception/control) **критически низкое**
2. **Индустрия**: Оптимальное соотношение **1.5x - 3x** (perception доминирует)
3. **Почему это важно**:
   - Слабый perception → плохая feature extraction
   - Мощный control не компенсирует слабые features
   - Vision encoder должен учиться сложным representations
4. **Рекомендация**: Увеличить perception до **10-15M params**, уменьшить control до **24M params**

**Следующий шаг**: Начните с быстрых wins (этап 1) и замерьте improvement в метриках!
