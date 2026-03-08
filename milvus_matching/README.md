# Video Similarity Search (Milvus)

Поиск похожих видео по эмбеддингам кадров с использованием Milvus.

## Структура

```
milvus_matching/
├── cluster_embeddings.py    # Кластеризация эмбеддингов (FAISS KMeans)
├── upload_to_milvus.py      # Загрузка центроидов в Milvus
├── query_matching_milvus.py # Поиск похожих видео через Milvus
├── query_matching.py        # Поиск через FAISS (legacy)
├── requirements.txt
├── .gitignore
└── README.md
```

## Пайплайн

### 1. Кластеризация эмбеддингов

```bash
python cluster_embeddings.py
```

Кластеризует эмбеддинги видео по AR-группам, сохраняет центроиды в `cluster_frames/ar_*/centroids.npy`.

**Конфигурация** (в файле):
- `EMBED_DIR` — папка с `.npy` файлами эмбеддингов
- `CSV_PATH` — CSV с метаданными видео (video_id, ar, filename)
- `CLUSTERS_PER_AR` — количество кластеров на AR (default: 150)
- `GPU_ID` — GPU для FAISS (default: 1)

### 2. Загрузка в Milvus

```bash
python upload_to_milvus.py
```

Загружает центроиды из `cluster_frames/` в коллекцию Dev Milvus.

**Конфигурация**:
- `MILVUS_HOST` / `MILVUS_PORT` — адрес Milvus сервера
- `COLLECTION_NAME` — имя коллекции (default: `video_centroids`)

### 3. Поиск похожих видео

```bash
python query_matching_milvus.py
```

Ищет совпадения query видео с кластерами в Milvus.

**Конфигурация**:
- `QUERY_EMBED_DIR` — папка с эмбеддингами query видео
- `QUERY_CSV_PATH` — CSV с метаданными query видео
- `THRESHOLDS` — пороги similarity для отчётов

**Выходные файлы** (в `query_results/`):
- `video_results.json` — полные результаты по видео
- `video_results.csv` — CSV с совпадениями
- `threshold_statistics.json` — статистика по порогам
- `threshold_summary.csv` — сводка по порогам
- `ar_threshold_stats.csv` — detection rate по AR

## Установка

```bash
pip install -r requirements.txt
```

## Схема данных в Milvus

Коллекция `video_centroids`:

| Field      | Type         | Description              |
|------------|--------------|--------------------------|
| id         | INT64 (PK)   | Auto-generated ID        |
| ar_id      | INT64        | AR group ID              |
| cluster_id | INT64        | Cluster ID within AR     |
| embedding  | FLOAT_VECTOR | Centroid vector (dim=1152) |

Index: FLAT, Metric: IP (Inner Product)
