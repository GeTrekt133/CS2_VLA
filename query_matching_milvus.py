import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from pymilvus import connections, Collection

# ===== КОНФИГУРАЦИЯ =====
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "video_centroids"

QUERY_EMBED_DIR = Path("/mnt/ml/msirotkin/vectorizing/test_embeds")
QUERY_CSV_PATH = Path("/mnt/ml/msirotkin/deleted_filenames_ar_november.csv")  # CSV с AR для query видео
CLUSTER_DIR = Path("cluster_frames")  # Для загрузки метаданных кластеров
OUTPUT_DIR = Path("query_results")

EMBEDDING_DIM = 1152
TOP_K = 1  # Берём только top-1 ближайший центроид для каждого кадра

OUTPUT_DIR.mkdir(exist_ok=True)


# ===== ПОДКЛЮЧЕНИЕ К MILVUS =====
print("=== Connecting to Milvus ===")
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

collection = Collection(COLLECTION_NAME)
collection.load()
print(f"Collection '{COLLECTION_NAME}' loaded, entities: {collection.num_entities}")


# ===== ЗАГРУЗКА ИНФОРМАЦИИ О QUERY ВИДЕО =====
print("\n=== Loading query video metadata ===")
query_df = pd.read_csv(QUERY_CSV_PATH)
query_video_to_ar = dict(zip(query_df['video_id'], query_df['ar']))

# Подсчёт видео по каждому AR в query датасете
query_ar_counts = query_df['ar'].value_counts().to_dict()
print(f"Query videos total: {len(query_df)}")
print(f"Unique ARs in query: {len(query_ar_counts)}")


# ===== ЗАГРУЗКА МЕТАДАННЫХ КЛАСТЕРОВ (для reference frames) =====
print("\n=== Loading cluster metadata ===")
ar_cluster_info = {}  # (ar_id, cluster_id) -> metadata

for ar_dir in sorted(CLUSTER_DIR.glob("ar_*")):
    ar_id = int(ar_dir.name.split("_")[1])
    extraction_log_path = ar_dir / "extraction_log.json"

    if extraction_log_path.exists():
        with open(extraction_log_path) as f:
            log = json.load(f)
            for key, info in log.items():
                cluster_id = int(key.split("_c")[1])
                ar_cluster_info[(ar_id, cluster_id)] = info

print(f"Loaded metadata for {len(ar_cluster_info)} clusters")


# ===== ЗАГРУЗКА И ОБРАБОТКА QUERY ЭМБЕДДИНГОВ =====
print("\n=== Processing query embeddings ===")

query_files = sorted(QUERY_EMBED_DIR.glob("*.npy"))
print(f"Found {len(query_files)} query files (videos)")

BATCH_SIZE = 10_000
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]

# Параметры поиска в Milvus
search_params = {
    "metric_type": "IP",
    "params": {},  # FLAT индекс не требует параметров
}

# Результаты на уровне видео (без фильтрации по порогу)
video_results = {}  # query_id -> {best_match_per_ar: {ar_id -> best_similarity}}

for query_file in tqdm(query_files, desc="Processing videos"):
    query_id = query_file.stem

    try:
        query_embeds = np.load(query_file).astype('float32')
    except Exception as e:
        print(f"Error loading {query_file}: {e}")
        continue

    if query_embeds.ndim == 1:
        query_embeds = query_embeds.reshape(1, -1)

    # Нормализуем векторы (как в оригинальном коде)
    norms = np.linalg.norm(query_embeds, axis=1, keepdims=True)
    norms[norms == 0] = 1
    query_embeds = query_embeds / norms

    # Для этого видео собираем лучший similarity для каждого AR
    best_match_per_ar = {}  # ar_id -> лучший match (максимальный similarity)

    # Поиск ближайших центроидов для каждого кадра
    for batch_start in range(0, len(query_embeds), BATCH_SIZE):
        batch = query_embeds[batch_start:batch_start + BATCH_SIZE]

        # Поиск в Milvus
        results = collection.search(
            data=batch.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=TOP_K,
            output_fields=["ar_id", "cluster_id", "frame_path"],
        )

        # Обрабатываем результаты
        for local_idx, hits in enumerate(results):
            if len(hits) == 0:
                continue

            hit = hits[0]  # Берём только top-1
            ar_id = hit.entity.get("ar_id")
            cluster_id = hit.entity.get("cluster_id")
            frame_path = hit.entity.get("frame_path")
            similarity = float(hit.distance)  # IP distance = similarity

            # Сохраняем лучший match для каждого AR (без фильтрации по порогу)
            if ar_id not in best_match_per_ar or similarity > best_match_per_ar[ar_id]["similarity"]:
                frame_idx = batch_start + local_idx
                best_match_per_ar[ar_id] = {
                    "ar_id": ar_id,
                    "cluster_id": cluster_id,
                    "similarity": similarity,
                    "frame_index": frame_idx,
                    "timestamp_sec": frame_idx * 5,
                }
                # Добавляем reference frame если есть
                if frame_path:
                    best_match_per_ar[ar_id]["reference_frame"] = frame_path
                elif (ar_id, cluster_id) in ar_cluster_info:
                    best_match_per_ar[ar_id]["reference_frame"] = ar_cluster_info[(ar_id, cluster_id)].get("frame_path")

    video_results[query_id] = {
        "total_frames": len(query_embeds),
        "best_matches": best_match_per_ar  # ar_id -> match info
    }


# ===== АГРЕГАЦИЯ РЕЗУЛЬТАТОВ ПО ПОРОГАМ =====
print("\n=== Aggregating results ===")

total_videos = len(video_results)
print(f"Total videos processed: {total_videos}")

# Статистика для каждого порога
threshold_stats = {}

for threshold in THRESHOLDS:
    print(f"\n{'='*60}")
    print(f"THRESHOLD = {threshold}")
    print('='*60)

    # Подсчёт по AR самого query видео (не по AR кластера)
    ar_matched_counts = {}  # query_ar -> сколько видео с этим AR имеют совпадение
    ar_matched_with_same = {}  # query_ar -> сколько видео совпали с кластером того же AR
    videos_with_match = 0
    videos_multi_ar = 0

    for query_id, result in video_results.items():
        # AR самого query видео
        query_ar = query_video_to_ar.get(query_id)

        matched_ars_at_threshold = []
        for cluster_ar_id, match_info in result["best_matches"].items():
            if match_info["similarity"] >= threshold:
                matched_ars_at_threshold.append(cluster_ar_id)

        # Если есть хотя бы одно совпадение
        if len(matched_ars_at_threshold) > 0:
            videos_with_match += 1
            if query_ar is not None:
                ar_matched_counts[query_ar] = ar_matched_counts.get(query_ar, 0) + 1
                # Проверяем, совпало ли с кластером того же AR
                if query_ar in matched_ars_at_threshold:
                    ar_matched_with_same[query_ar] = ar_matched_with_same.get(query_ar, 0) + 1

        if len(matched_ars_at_threshold) > 1:
            videos_multi_ar += 1

    videos_without_match = total_videos - videos_with_match

    # Процент относительно количества видео с каждым AR в query датасете
    print(f"\nDetection rate by query video AR (threshold >= {threshold}):")
    print(f"{'AR':<8} {'Matched':<10} {'Total':<10} {'Rate':<10} {'Same AR':<10}")
    print("-" * 48)
    for query_ar in sorted(query_ar_counts.keys()):
        total_ar_videos = query_ar_counts[query_ar]
        matched_count = ar_matched_counts.get(query_ar, 0)
        same_ar_count = ar_matched_with_same.get(query_ar, 0)
        rate = matched_count / total_ar_videos * 100 if total_ar_videos > 0 else 0
        print(f"AR {query_ar:<4} {matched_count:<10} {total_ar_videos:<10} {rate:.1f}%{'':<5} {same_ar_count}")

    print(f"\nOverall: {videos_with_match}/{total_videos} videos with matches ({videos_with_match/total_videos*100:.1f}%)")
    print(f"Videos with multiple ARs: {videos_multi_ar} ({videos_multi_ar/total_videos*100:.1f}%)")

    # Сохраняем статистику
    ar_rates = {}
    for query_ar in query_ar_counts.keys():
        total_ar_videos = query_ar_counts[query_ar]
        matched_count = ar_matched_counts.get(query_ar, 0)
        same_ar_count = ar_matched_with_same.get(query_ar, 0)
        ar_rates[str(query_ar)] = {
            "matched": matched_count,
            "total": total_ar_videos,
            "rate": matched_count / total_ar_videos * 100 if total_ar_videos > 0 else 0,
            "matched_same_ar": same_ar_count,
            "same_ar_rate": same_ar_count / total_ar_videos * 100 if total_ar_videos > 0 else 0
        }

    threshold_stats[threshold] = {
        "ar_stats": ar_rates,
        "videos_with_match": videos_with_match,
        "videos_without_match": videos_without_match,
        "videos_multi_ar": videos_multi_ar
    }

# Сводная таблица по порогам
print(f"\n{'='*60}")
print("SUMMARY ACROSS THRESHOLDS")
print('='*60)
print(f"{'Threshold':<12} {'With Match':<15} {'No Match':<15} {'Multi-AR':<12}")
print("-" * 54)
for threshold in THRESHOLDS:
    stats = threshold_stats[threshold]
    print(f"{threshold:<12} {stats['videos_with_match']:<15} {stats['videos_without_match']:<15} {stats['videos_multi_ar']:<12}")


# ===== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ =====
print("\n=== Saving results ===")

# Полные результаты по видео (конвертируем ключи ar_id в строки для JSON)
video_results_json = {}
for query_id, result in video_results.items():
    video_results_json[query_id] = {
        "total_frames": result["total_frames"],
        "best_matches": {str(k): v for k, v in result["best_matches"].items()}
    }

with open(OUTPUT_DIR / "video_results.json", "w") as f:
    json.dump(video_results_json, f, indent=2)

# Статистика по всем порогам
with open(OUTPUT_DIR / "threshold_statistics.json", "w") as f:
    json.dump({
        "total_videos": total_videos,
        "query_ar_counts": {str(k): v for k, v in query_ar_counts.items()},
        "thresholds": {str(t): stats for t, stats in threshold_stats.items()}
    }, f, indent=2)

# CSV для удобного просмотра - с колонками для каждого порога
summary_rows = []
for query_id, result in video_results.items():
    row = {
        "video_id": query_id,
        "total_frames": result["total_frames"],
    }

    # Лучший AR и similarity
    best_ar = None
    best_sim = 0
    all_matches = []

    for ar_id, match_info in result["best_matches"].items():
        sim = match_info["similarity"]
        all_matches.append(f"{ar_id}:{sim:.3f}")
        if sim > best_sim:
            best_sim = sim
            best_ar = ar_id

    row["best_ar"] = best_ar
    row["best_similarity"] = round(best_sim, 4) if best_sim > 0 else None
    row["all_matches"] = "; ".join(all_matches)

    # Для каждого порога - есть ли совпадение
    for threshold in THRESHOLDS:
        matched_ars = [ar for ar, info in result["best_matches"].items() if info["similarity"] >= threshold]
        row[f"match_{threshold}"] = len(matched_ars) > 0
        row[f"ars_{threshold}"] = ",".join(map(str, matched_ars)) if matched_ars else ""

    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)
df_summary = df_summary.sort_values("best_similarity", ascending=False, na_position="last")
df_summary.to_csv(OUTPUT_DIR / "video_results.csv", index=False)

# Сводная таблица по порогам в CSV
threshold_summary_rows = []
for threshold in THRESHOLDS:
    stats = threshold_stats[threshold]
    threshold_summary_rows.append({
        "threshold": threshold,
        "videos_with_match": stats["videos_with_match"],
        "videos_without_match": stats["videos_without_match"],
        "videos_multi_ar": stats["videos_multi_ar"],
        "match_rate_pct": round(stats["videos_with_match"] / total_videos * 100, 2)
    })

df_thresholds = pd.DataFrame(threshold_summary_rows)
df_thresholds.to_csv(OUTPUT_DIR / "threshold_summary.csv", index=False)

# AR статистика по порогам в CSV (matched/total/rate для каждого AR и порога)
ar_stats_rows = []

for ar_id in sorted(query_ar_counts.keys()):
    row = {"ar": ar_id, "total_videos": query_ar_counts[ar_id]}
    for threshold in THRESHOLDS:
        ar_stat = threshold_stats[threshold]["ar_stats"].get(str(ar_id), {})
        row[f"matched_{threshold}"] = ar_stat.get("matched", 0)
        row[f"rate_{threshold}"] = round(ar_stat.get("rate", 0), 2)
        row[f"same_ar_{threshold}"] = ar_stat.get("matched_same_ar", 0)
        row[f"same_rate_{threshold}"] = round(ar_stat.get("same_ar_rate", 0), 2)
    ar_stats_rows.append(row)

df_ar_stats = pd.DataFrame(ar_stats_rows)
df_ar_stats.to_csv(OUTPUT_DIR / "ar_threshold_stats.csv", index=False)

print(f"""
{'='*60}
DONE!
{'='*60}
Results in: {OUTPUT_DIR}

Files:
  - video_results.json        : Full results per video
  - video_results.csv         : CSV with matched ARs per video (columns per threshold)
  - threshold_statistics.json : Detailed stats per threshold
  - threshold_summary.csv     : Summary table across thresholds
  - ar_threshold_stats.csv    : AR detection rates per threshold

Summary:
  - Videos processed: {total_videos}
  - Thresholds evaluated: {THRESHOLDS}
""")

connections.disconnect("default")
