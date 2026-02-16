# Анализ использования видеопамяти

## Executive Summary

| Конфигурация | Model Weights | Activations | Buffers | Cache | **Total VRAM** |
|--------------|---------------|-------------|---------|-------|----------------|
| PyTorch FP32 | ~120 MB | ~300 MB | ~50 MB | ~2 MB | **~472 MB** |
| TRT FP16 | ~60 MB | ~200 MB | ~50 MB | ~2 MB | **~312 MB** |

**Экономия с TRT FP16: ~160 MB (34% меньше)**

---

## Детальный анализ по компонентам

### 1. Model Weights (веса моделей)

#### PyTorch FP32 версия:

```python
RadarEncoder (EfficientNet-B0):
  - Параметры: ~5.3M
  - Размер FP32: 5.3M × 4 bytes = 21.2 MB

YOLO (YOLOv11n):
  - Параметры: ~2.6M
  - Размер FP32: 2.6M × 4 bytes = 10.4 MB

AudioEncoder (QuartzNet-based):
  - Параметры: ~6.7M
  - Размер FP32: 6.7M × 4 bytes = 26.8 MB

TemporalCrossTransformer:
  - Depth: 6 layers
  - Num heads: 8
  - d_model: 512
  - Параметры: ~12.5M
  - Размер FP32: 12.5M × 4 bytes = 50 MB

FlowActionHead:
  - Параметры: ~0.5M
  - Размер FP32: 0.5M × 4 bytes = 2 MB

─────────────────────────────────────────
Total PyTorch FP32: ~110 MB
```

#### TensorRT FP16 версия:

```python
Все модели в FP16 (половина от FP32):
  - RadarEncoder: 21.2 MB → 10.6 MB
  - YOLO: 10.4 MB → 5.2 MB
  - AudioEncoder: 26.8 MB → 13.4 MB
  - TemporalFlow: 52 MB → 26 MB

─────────────────────────────────────────
Total TRT FP16: ~55 MB

Экономия: ~55 MB (50% reduction)
```

---

### 2. Activation Memory (промежуточные активации)

Память для intermediate tensors во время forward pass:

#### PyTorch FP32:

```python
RadarEncoder forward:
  - Input: (1, 3, 224, 224) = 0.6 MB
  - Feature maps: ~20 MB (EfficientNet layers)
  - Output: (1, 512) = 2 KB

YOLO forward:
  - Input: (1, 3, 480, 640) = 3.7 MB
  - Feature pyramid: ~80 MB
  - Output: (1, 2048) = 8 KB

AudioEncoder forward:
  - Input: (1, 480000) = 1.9 MB
  - Mel-spectrogram: ~5 MB
  - Conv layers: ~30 MB
  - Output: (1, 60, 512) = 0.12 MB

TemporalTransformer forward:
  - Inputs (concatenated): ~2 MB
  - Attention matrices: ~40 MB (self-attention)
  - FFN intermediates: ~30 MB
  - Output: ~5 KB

FlowActionHead forward:
  - Flow matching iterations: ~5 MB
  - Output: ~20 bytes

─────────────────────────────────────────
Peak activation memory: ~300 MB

Примечание: С embedding cache активации
radar/scene уменьшаются (1 frame vs 32),
но TemporalTransformer остается основным
потребителем activation memory.
```

#### TensorRT FP16:

```python
TRT optimizations:
  - Kernel fusion (conv+bn+relu) → меньше intermediate tensors
  - FP16 activations → 50% меньше памяти
  - Memory pooling → переиспользование буферов

Estimated peak activation: ~200 MB

Экономия: ~100 MB (33% reduction)
```

---

### 3. Input/Output Buffers

```python
Frame buffers (CPU → GPU):
  - Scene buffer: 16 frames × (3, 480, 640) × 4 bytes = 59 MB
  - Radar buffer: 32 frames × (3, 224, 224) × 4 bytes = 19 MB
  - Audio buffer: 480000 samples × 4 bytes = 1.9 MB

Output buffers:
  - Policy tensors: negligible (~100 KB)

─────────────────────────────────────────
Total buffers: ~80 MB (PyTorch & TRT)

Примечание: Это максимум при полном буфере.
В реальности может быть меньше.
```

---

### 4. Embedding Cache

```python
Radar cache:
  - Capacity: 256 frames
  - Embedding dim: 512
  - Size: 256 × 512 × 4 bytes = 0.5 MB

Scene cache:
  - Capacity: 128 frames
  - Embedding dim: 2048
  - Size: 128 × 2048 × 4 bytes = 1 MB

Audio cache:
  - Dynamic allocation: ~0.1 MB

─────────────────────────────────────────
Total cache: ~1.6 MB (PyTorch & TRT)
```

---

### 5. TensorRT Engine Overhead

```python
TRT runtime overhead:
  - Engine context: ~5-10 MB per model
  - CUDA graph state: ~10 MB
  - Internal buffers: ~20 MB

─────────────────────────────────────────
Total TRT overhead: ~40 MB

Примечание: Это дополнительная память сверх
весов и активаций, но меньше чем PyTorch runtime.
```

---

## Сравнение конфигураций

### Baseline (PyTorch FP32, без cache):

```
Component               Memory (MB)   % of Total
────────────────────────────────────────────────
Model weights           110           23%
Activation memory       300           64%
Input buffers           50            11%
Runtime overhead        10            2%
────────────────────────────────────────────────
TOTAL                   470 MB        100%
```

### Optimized (TRT FP16 + Embedding Cache):

```
Component               Memory (MB)   % of Total
────────────────────────────────────────────────
Model weights (FP16)    55            18%
Activation memory       200           64%
Input buffers           50            16%
Embedding cache         1.6           0.5%
TRT runtime overhead    5             1.5%
────────────────────────────────────────────────
TOTAL                   311.6 MB      100%

Экономия: 158 MB (34% reduction)
```

---

## GPU Compatibility

### Минимальные требования:

| GPU Tier | VRAM | PyTorch FP32 | TRT FP16 | Рекомендация |
|----------|------|--------------|----------|--------------|
| Entry | 2 GB | ❌ Tight fit | ✅ Comfortable | GTX 1650, RTX 3050 |
| Mid-range | 4 GB | ✅ Comfortable | ✅ Plenty | RTX 3060, RTX 4060 |
| High-end | 6+ GB | ✅ Plenty | ✅ Plenty | RTX 3070+, RTX 4070+ |

**Рекомендация:** Минимум 3 GB VRAM для комфортной работы с TRT FP16.

---

## Memory Optimization Tips

### 1. Уменьшение buffer размеров:

```python
# В config.py
radar_buffer_size = 32  # Reduce to 16 → -10 MB
scene_buffer_size = 16  # Reduce to 8 → -30 MB
```

### 2. Отключение audio:

```python
use_audio = False  # Saves ~30 MB (AudioEncoder + buffers)
```

### 3. Уменьшение cache capacity:

```python
# В engine.py
radar_cache = GPUEmbeddingCache(
    capacity=128,  # вместо 256 → -0.25 MB
    ...
)
scene_cache = GPUEmbeddingCache(
    capacity=64,   # вместо 128 → -0.5 MB
    ...
)
```

---

## Профилирование реальной памяти

### Как проверить использование VRAM:

```bash
# Во время inference
nvidia-smi -l 1

# Или в Python
import torch
torch.cuda.memory_allocated() / 1024**2  # MB
torch.cuda.memory_reserved() / 1024**2   # MB
torch.cuda.max_memory_allocated() / 1024**2  # Peak
```

### Ожидаемый вывод:

```
PyTorch FP32:
  - Allocated: ~400-500 MB
  - Reserved: ~600-800 MB (с учетом fragmentation)
  - Peak: ~700 MB

TRT FP16:
  - Allocated: ~280-320 MB
  - Reserved: ~400-500 MB
  - Peak: ~450 MB
```

---

## Выводы

1. **TRT FP16 экономит ~160 MB VRAM (34%)** по сравнению с PyTorch FP32
2. **Embedding cache добавляет всего 1.6 MB** (negligible overhead)
3. **Основной потребитель памяти:** activation memory во время forward pass
4. **Минимальная GPU:** 3+ GB VRAM для TRT FP16 (комфортно)
5. **Рекомендуемая GPU:** 4+ GB VRAM для запаса

---

## FAQ

**Q: Можно ли уменьшить активации до FP16 в PyTorch?**
A: Да, через `torch.cuda.amp.autocast()`, но это не даст столько же оптимизации как TRT.

**Q: Почему TRT занимает 200 MB activations, а не 150 MB (50% от PyTorch)?**
A: Некоторые операции TRT все равно выполняет в FP32 для точности (например, softmax в attention).

**Q: Можно ли запустить на GPU с 2 GB VRAM?**
A: Теоретически да с TRT FP16 + без audio + уменьшенными буферами, но будет очень tight.

**Q: Влияет ли batch size на память?**
A: Да, но inference всегда с batch=1, так что это не проблема.

**Q: Как влияет на память увеличение sequence length?**
A: Линейно растет activation memory в TemporalTransformer (attention O(N²)).
