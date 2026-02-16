import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)

# ===== КОНФИГУРАЦИЯ =====
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "video_centroids"

CLUSTER_DIR = Path("cluster_frames")  # Результаты кластеризации
EMBEDDING_DIM = 1152

# ===== ПОДКЛЮЧЕНИЕ К MILVUS =====
print("=== Connecting to Milvus ===")
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")


# ===== СОЗДАНИЕ КОЛЛЕКЦИИ =====
print("\n=== Creating collection ===")

# Удаляем старую коллекцию если существует
if utility.has_collection(COLLECTION_NAME):
    print(f"Dropping existing collection '{COLLECTION_NAME}'")
    utility.drop_collection(COLLECTION_NAME)

# Схема коллекции
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="ar_id", dtype=DataType.INT64),
    FieldSchema(name="cluster_id", dtype=DataType.INT64),
    FieldSchema(name="frame_path", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
]

schema = CollectionSchema(fields, description="Video cluster centroids")
collection = Collection(COLLECTION_NAME, schema)
print(f"Collection '{COLLECTION_NAME}' created")


# ===== ЗАГРУЗКА ЦЕНТРОИДОВ ИЗ ВСЕХ AR =====
print("\n=== Loading centroids from all AR clusters ===")

ar_data = []  # Список для batch insert

for ar_dir in sorted(CLUSTER_DIR.glob("ar_*")):
    ar_id = int(ar_dir.name.split("_")[1])
    centroids_path = ar_dir / "centroids.npy"

    if not centroids_path.exists():
        print(f"  AR {ar_id}: centroids not found, skipping")
        continue

    centroids = np.load(centroids_path).astype('float32')

    # Нормализуем векторы (как в оригинальном коде)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Избегаем деления на 0
    centroids = centroids / norms

    # Загружаем метаданные кластеров если есть
    extraction_log_path = ar_dir / "extraction_log.json"
    cluster_metadata = {}
    if extraction_log_path.exists():
        with open(extraction_log_path) as f:
            log = json.load(f)
            for key, info in log.items():
                cluster_id = int(key.split("_c")[1])
                cluster_metadata[cluster_id] = info

    # Собираем данные для вставки
    for local_id in range(len(centroids)):
        frame_path = cluster_metadata.get(local_id, {}).get("frame_path", "")
        ar_data.append({
            "ar_id": ar_id,
            "cluster_id": local_id,
            "frame_path": frame_path if frame_path else "",
            "embedding": centroids[local_id].tolist(),
        })

    print(f"  AR {ar_id}: {len(centroids)} centroids prepared")

print(f"\nTotal centroids to upload: {len(ar_data)}")


# ===== ВСТАВКА ДАННЫХ В MILVUS =====
print("\n=== Uploading to Milvus ===")

BATCH_SIZE = 1000
for batch_start in tqdm(range(0, len(ar_data), BATCH_SIZE), desc="Uploading batches"):
    batch = ar_data[batch_start:batch_start + BATCH_SIZE]

    # Формируем данные для вставки (по колонкам)
    insert_data = [
        [item["ar_id"] for item in batch],
        [item["cluster_id"] for item in batch],
        [item["frame_path"] for item in batch],
        [item["embedding"] for item in batch],
    ]

    collection.insert(insert_data)

# Flush для сохранения на диск
collection.flush()
print(f"Uploaded {len(ar_data)} centroids")


# ===== СОЗДАНИЕ ИНДЕКСА =====
print("\n=== Creating index ===")

index_params = {
    "metric_type": "IP",  # Inner Product (эквивалентно cosine для нормализованных векторов)
    "index_type": "FLAT",  # Точный поиск
    "params": {},
}

collection.create_index("embedding", index_params)
print("Index created")

# Загружаем коллекцию в память для поиска
collection.load()
print("Collection loaded into memory")


# ===== ПРОВЕРКА =====
print("\n=== Verification ===")
print(f"Collection '{COLLECTION_NAME}':")
print(f"  - Total entities: {collection.num_entities}")
print(f"  - Index: {collection.index().params}")

# Получаем статистику по AR
ar_counts = {}
results = collection.query(
    expr="ar_id >= 0",
    output_fields=["ar_id"],
    limit=16384,  # Milvus limit
)

# Если данных много, считаем по-другому
if len(results) >= 16384:
    print(f"  - AR distribution: (too many records, showing sample)")
else:
    for r in results:
        ar_id = r["ar_id"]
        ar_counts[ar_id] = ar_counts.get(ar_id, 0) + 1

    print(f"  - AR distribution:")
    for ar_id in sorted(ar_counts.keys()):
        print(f"      AR {ar_id}: {ar_counts[ar_id]} centroids")


print(f"""
{'='*60}
DONE!
{'='*60}
Collection '{COLLECTION_NAME}' is ready for queries.

To use in query_matching_milvus.py:
  - MILVUS_HOST = "{MILVUS_HOST}"
  - MILVUS_PORT = "{MILVUS_PORT}"
  - COLLECTION_NAME = "{COLLECTION_NAME}"
""")

connections.disconnect("default")
