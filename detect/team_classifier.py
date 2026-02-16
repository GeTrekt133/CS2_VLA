import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import json
import cv2
import shutil
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm


def classify_teams_by_spawn(csv_path: str, initial_ticks: int = 5, visualize: bool = True) -> Dict[int, str]:
    """
    Кластеризует игроков на CT/T по начальным позициям спавна используя K-means.

    Args:
        csv_path: путь к spotted_alive_players.csv
        initial_ticks: количество начальных уникальных тиков для анализа позиций спавна
        visualize: визуализировать ли кластеры (scatter plot)

    Returns:
        dict: {steamid: 'CT' или 'T'}
    """
    # 1. Загрузить CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")
    print(f"Unique ticks: {df['tick'].nunique()}")
    print(f"Unique players (steamid): {df['steamid'].nunique()}")

    # 2. Взять первые N уникальных тиков
    unique_ticks = sorted(df['tick'].unique())
    initial_tick_set = unique_ticks[:initial_ticks]
    print(f"\nUsing first {initial_ticks} ticks for spawn position analysis: {initial_tick_set}")

    df_initial = df[df['tick'].isin(initial_tick_set)].copy()
    print(f"Records in initial ticks: {len(df_initial)}")

    # 3. Для каждого steamid вычислить среднюю позицию (X, Y)
    player_positions = df_initial.groupby('steamid')[['X', 'Y']].mean()
    print(f"\nComputed average spawn positions for {len(player_positions)} players")

    if len(player_positions) < 2:
        raise ValueError(f"Not enough players to cluster (found {len(player_positions)}). Need at least 2.")

    # 4. K-means clustering с k=2
    X = player_positions[['X', 'Y']].values
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    player_positions['cluster'] = labels
    cluster_centers = kmeans.cluster_centers_

    print(f"\nK-means clustering complete")
    print(f"Cluster 0 center: X={cluster_centers[0][0]:.1f}, Y={cluster_centers[0][1]:.1f}")
    print(f"Cluster 1 center: X={cluster_centers[1][0]:.1f}, Y={cluster_centers[1][1]:.1f}")

    # 5. Определить какой кластер CT, какой T
    # Эвристика: обычно CT имеют меньшую медиану Y координаты (но это зависит от карты)
    # Более надежный способ: использовать X координату или визуальную проверку
    # Для универсальности используем кластер с меньшим индексом как CT, больший как T
    # НО даём пользователю возможность визуально проверить

    cluster_0_y_median = player_positions[player_positions['cluster'] == 0]['Y'].median()
    cluster_1_y_median = player_positions[player_positions['cluster'] == 1]['Y'].median()

    # Простая эвристика: кластер с меньшей медианой Y = CT (работает на большинстве карт)
    if cluster_0_y_median < cluster_1_y_median:
        ct_cluster = 0
        t_cluster = 1
    else:
        ct_cluster = 1
        t_cluster = 0

    print(f"\nHeuristic assignment:")
    print(f"Cluster {ct_cluster} → CT (median Y={player_positions[player_positions['cluster']==ct_cluster]['Y'].median():.1f})")
    print(f"Cluster {t_cluster} → T (median Y={player_positions[player_positions['cluster']==t_cluster]['Y'].median():.1f})")

    # 6. Создать маппинг steamid -> team
    team_mapping = {}
    for steamid, row in player_positions.iterrows():
        if row['cluster'] == ct_cluster:
            team_mapping[int(steamid)] = 'CT'
        else:
            team_mapping[int(steamid)] = 'T'

    ct_count = sum(1 for v in team_mapping.values() if v == 'CT')
    t_count = sum(1 for v in team_mapping.values() if v == 'T')
    print(f"\nFinal team assignment: {ct_count} CT, {t_count} T")

    # 7. Визуализация
    if visualize:
        visualize_clusters(player_positions, cluster_centers, ct_cluster, t_cluster)

    return team_mapping


def visualize_clusters(player_positions: pd.DataFrame,
                       cluster_centers: np.ndarray,
                       ct_cluster: int,
                       t_cluster: int,
                       save_path: str = None):
    """
    Визуализирует результаты кластеризации.

    Args:
        player_positions: DataFrame с колонками X, Y, cluster
        cluster_centers: координаты центров кластеров
        ct_cluster: индекс кластера CT
        t_cluster: индекс кластера T
        save_path: путь для сохранения графика (опционально)
    """
    plt.figure(figsize=(12, 8))

    # Отображаем игроков
    for cluster_id in [ct_cluster, t_cluster]:
        cluster_data = player_positions[player_positions['cluster'] == cluster_id]
        color = 'blue' if cluster_id == ct_cluster else 'red'
        label = 'CT' if cluster_id == ct_cluster else 'T'
        plt.scatter(cluster_data['X'], cluster_data['Y'],
                   c=color, label=label, alpha=0.6, s=100, edgecolors='black')

    # Отображаем центры кластеров
    plt.scatter(cluster_centers[ct_cluster, 0], cluster_centers[ct_cluster, 1],
               c='darkblue', marker='X', s=500, edgecolors='black', linewidths=2,
               label='CT Center', zorder=10)
    plt.scatter(cluster_centers[t_cluster, 0], cluster_centers[t_cluster, 1],
               c='darkred', marker='X', s=500, edgecolors='black', linewidths=2,
               label='T Center', zorder=10)

    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.title('CS2 Player Team Clustering (Spawn Positions)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Visualization saved to {save_path}")
    else:
        plt.savefig('team_clustering_visualization.png', dpi=150)
        print("Visualization saved to team_clustering_visualization.png")

    plt.show()


def save_team_mapping(team_mapping: Dict[int, str], output_path: str):
    """
    Сохраняет маппинг команд в JSON файл.

    Args:
        team_mapping: словарь {steamid: 'CT' или 'T'}
        output_path: путь к выходному JSON файлу
    """
    # Конвертируем int ключи в строки для JSON
    team_mapping_str = {str(k): v for k, v in team_mapping.items()}

    with open(output_path, 'w') as f:
        json.dump(team_mapping_str, f, indent=2)

    print(f"\nTeam mapping saved to {output_path}")
    print(f"Total players: {len(team_mapping)}")


def load_team_mapping(json_path: str) -> Dict[int, str]:
    """
    Загружает маппинг команд из JSON файла.

    Args:
        json_path: путь к JSON файлу

    Returns:
        dict: {steamid: 'CT' или 'T'}
    """
    with open(json_path, 'r') as f:
        team_mapping_str = json.load(f)

    # Конвертируем строковые ключи обратно в int
    team_mapping = {int(k): v for k, v in team_mapping_str.items()}

    return team_mapping


def extract_color_features(img: np.ndarray) -> np.ndarray:
    """
    Извлекает цветовые признаки из изображения bbox для классификации CT/T.

    CT обычно: зелёный/оливковый цвет одежды, тёмно-синие элементы
    T обычно: коричневый/серый/бежевый цвет одежды + белые элементы (маски, рубашки)

    Args:
        img: BGR изображение (crop bbox)

    Returns:
        np.ndarray: вектор цветовых признаков (20 признаков)
    """
    if img is None or img.size == 0:
        return np.zeros(20)

    # Конвертируем в HSV для лучшего анализа цвета
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Разделяем каналы
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(img)

    # Маска для исключения очень тёмных пикселей (фон)
    valid_mask = (v > 30) & (v < 250)

    if valid_mask.sum() < 10:
        # Если мало валидных пикселей, используем все
        valid_mask = np.ones_like(v, dtype=bool)

    features = []

    # 1. Средние значения HSV (только валидные пиксели)
    features.append(h[valid_mask].mean() if valid_mask.any() else 0)  # 0
    features.append(s[valid_mask].mean() if valid_mask.any() else 0)  # 1
    features.append(v[valid_mask].mean() if valid_mask.any() else 0)  # 2

    # 2. Соотношение зелёного к красному (CT более зелёные)
    g_mean = g[valid_mask].mean() if valid_mask.any() else 1
    r_mean = r[valid_mask].mean() if valid_mask.any() else 1
    b_mean = b[valid_mask].mean() if valid_mask.any() else 1

    features.append(g_mean / (r_mean + 1))  # 3: G/R ratio - CT выше
    features.append(g_mean / (b_mean + 1))  # 4: G/B ratio
    features.append((g_mean - r_mean) / (g_mean + r_mean + 1))  # 5: Normalized G-R - CT выше

    # 3. Доля "зелёных" пикселей (CT форма: Hue 40-75, умеренная насыщенность)
    green_ct_mask = (h >= 40) & (h <= 75) & (s >= 40) & (s <= 180) & valid_mask
    green_ct_ratio = green_ct_mask.sum() / (valid_mask.sum() + 1)
    features.append(green_ct_ratio)  # 6: Green CT ratio - CT выше

    # 4. Доля "коричневых/оранжевых" пикселей (Hue: 10-25)
    brown_mask = (h >= 10) & (h <= 25) & valid_mask
    brown_ratio = brown_mask.sum() / (valid_mask.sum() + 1)
    features.append(brown_ratio)  # 7: Brown ratio - T выше

    # 5. Доля "серых" пикселей (низкая насыщенность, средняя яркость)
    gray_mask = (s < 40) & (v > 50) & (v < 200) & valid_mask
    gray_ratio = gray_mask.sum() / (valid_mask.sum() + 1)
    features.append(gray_ratio)  # 8: Gray ratio - T выше

    # 6. Стандартное отклонение Hue (разнообразие цветов)
    features.append(h[valid_mask].std() if valid_mask.sum() > 1 else 0)  # 9

    # 7. Медианные значения (более устойчивы к выбросам)
    features.append(np.median(h[valid_mask]) if valid_mask.any() else 0)  # 10
    features.append(np.median(s[valid_mask]) if valid_mask.any() else 0)  # 11

    # ==================== НОВЫЕ ПРИЗНАКИ ====================

    # 8. Доля БЕЛЫХ пикселей (высокая яркость + низкая насыщенность)
    # T-модели часто имеют белые элементы (маски, рубашки)
    white_mask = (v > 180) & (s < 50) & valid_mask
    white_ratio = white_mask.sum() / (valid_mask.sum() + 1)
    features.append(white_ratio)  # 12: White ratio - T выше

    # 9. Доля ОЛИВКОВЫХ/ХАКИ пикселей (CT форма)
    # Hue 25-45 (между коричневым и зелёным), средняя насыщенность
    olive_mask = (h >= 25) & (h <= 45) & (s >= 30) & (s <= 120) & valid_mask
    olive_ratio = olive_mask.sum() / (valid_mask.sum() + 1)
    features.append(olive_ratio)  # 13: Olive/Khaki ratio - CT выше

    # 10. Доля ТЕЛЕСНОГО цвета (открытые части тела)
    # T-модели чаще показывают руки/лицо
    skin_mask = (h >= 5) & (h <= 20) & (s >= 30) & (s <= 100) & (v >= 100) & valid_mask
    skin_ratio = skin_mask.sum() / (valid_mask.sum() + 1)
    features.append(skin_ratio)  # 14: Skin ratio - T может быть выше

    # 11. Доля ТЁМНО-СИНИХ пикселей (некоторые CT модели)
    # Hue 100-130, высокая насыщенность, низкая-средняя яркость
    navy_mask = (h >= 100) & (h <= 130) & (s >= 50) & (v >= 30) & (v <= 150) & valid_mask
    navy_ratio = navy_mask.sum() / (valid_mask.sum() + 1)
    features.append(navy_ratio)  # 15: Navy ratio - CT выше

    # 12. EDGE DENSITY (плотность границ/текстура)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 50, 150)
    edge_density = edges.sum() / (255 * edges.size + 1)
    features.append(edge_density)  # 16: Edge density

    # 13. Анализ ВЕРХНЕЙ ТРЕТИ (голова/плечи - более информативно)
    top_h = max(1, img.shape[0] // 3)
    top_region = img[:top_h, :]
    if top_region.size > 0:
        top_hsv = cv2.cvtColor(top_region, cv2.COLOR_BGR2HSV)
        top_h_ch, top_s_ch, top_v_ch = cv2.split(top_hsv)
        top_valid = (top_v_ch > 30) & (top_v_ch < 250)
        if top_valid.sum() > 5:
            # Белый в верхней части (маски T)
            top_white = ((top_v_ch > 180) & (top_s_ch < 50) & top_valid).sum() / (top_valid.sum() + 1)
            # Зелёный в верхней части (шлемы CT)
            top_green = ((top_h_ch >= 40) & (top_h_ch <= 75) & (top_s_ch >= 40) & top_valid).sum() / (top_valid.sum() + 1)
            features.append(top_white)  # 17: Top white ratio - T выше
            features.append(top_green)  # 18: Top green ratio - CT выше
        else:
            features.extend([0, 0])
    else:
        features.extend([0, 0])

    # 14. Доля БЕЖЕВЫХ пикселей (T одежда)
    # Hue 15-30, низкая-средняя насыщенность, высокая яркость
    beige_mask = (h >= 15) & (h <= 30) & (s >= 20) & (s <= 80) & (v >= 120) & valid_mask
    beige_ratio = beige_mask.sum() / (valid_mask.sum() + 1)
    features.append(beige_ratio)  # 19: Beige ratio - T выше

    return np.array(features, dtype=np.float32)


def classify_bboxes_by_color(
    labels_dir: str,
    images_dir: str,
    visualize: bool = True,
    save_path: str = None,
    padding: float = 0.1
) -> pd.DataFrame:
    """
    Кластеризует YOLO bbox на CT/T по цвету одежды игроков.

    CT: зелёная/оливковая форма
    T: коричневая/серая/бежевая форма

    Args:
        labels_dir: директория с YOLO .txt файлами
        images_dir: директория с изображениями
        visualize: визуализировать результаты
        save_path: путь для сохранения визуализации
        padding: отступ вокруг bbox при вырезании

    Returns:
        DataFrame с колонками: filename, filepath, x_center, y_center, w, h, new_class, color_features
    """
    labels_path = Path(labels_dir)
    images_path = Path(images_dir)

    if not labels_path.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")
    if not images_path.exists():
        raise ValueError(f"Images directory not found: {images_dir}")

    # Собираем все bbox с цветовыми признаками
    bbox_data = []

    txt_files = list(labels_path.glob("*.txt"))
    print(f"Found {len(txt_files)} label files")
    print(f"Extracting color features from bbox crops...")

    for txt_file in tqdm(txt_files, desc="Processing labels"):
        img_stem = txt_file.stem

        # Ищем соответствующее изображение
        img_file = None
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = images_path / f"{img_stem}{ext}"
            if candidate.exists():
                img_file = candidate
                break

        if img_file is None:
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        # Читаем аннотации
        with open(txt_file, 'r') as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    # Вырезаем bbox с padding
                    bbox_w = int(w * img_w)
                    bbox_h = int(h * img_h)
                    bbox_cx = int(x_center * img_w)
                    bbox_cy = int(y_center * img_h)

                    pad_w = int(bbox_w * padding)
                    pad_h = int(bbox_h * padding)

                    x1 = max(0, bbox_cx - bbox_w // 2 - pad_w)
                    y1 = max(0, bbox_cy - bbox_h // 2 - pad_h)
                    x2 = min(img_w, bbox_cx + bbox_w // 2 + pad_w)
                    y2 = min(img_h, bbox_cy + bbox_h // 2 + pad_h)

                    crop = img[y1:y2, x1:x2]

                    if crop.size == 0:
                        continue

                    # Извлекаем цветовые признаки
                    color_features = extract_color_features(crop)

                    bbox_data.append({
                        'filename': txt_file.name,
                        'filepath': txt_file,
                        'x_center': x_center,
                        'y_center': y_center,
                        'w': w,
                        'h': h,
                        'original_class': class_id,
                        'color_features': color_features
                    })

    if len(bbox_data) < 2:
        raise ValueError(f"Not enough bboxes to cluster (found {len(bbox_data)})")

    print(f"\nExtracted features from {len(bbox_data)} bounding boxes")

    # Создаём DataFrame
    df = pd.DataFrame(bbox_data)

    # Собираем матрицу признаков для кластеризации
    X = np.vstack(df['color_features'].values)

    # Нормализуем признаки
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-means кластеризация
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    df['cluster'] = labels

    # Определяем какой кластер CT, какой T используя улучшенную эвристику
    # Индексы признаков (20 признаков):
    # 0-2: Mean H, S, V
    # 3: G/R ratio - CT выше
    # 4: G/B ratio
    # 5: Normalized G-R - CT выше
    # 6: Green CT ratio - CT выше
    # 7: Brown ratio - T выше
    # 8: Gray ratio - T выше
    # 9: Hue std
    # 10-11: Median H, S
    # 12: White ratio - T выше (НОВЫЙ)
    # 13: Olive ratio - CT выше (НОВЫЙ)
    # 14: Skin ratio - T выше (НОВЫЙ)
    # 15: Navy ratio - CT выше (НОВЫЙ)
    # 16: Edge density
    # 17: Top white ratio - T выше (НОВЫЙ)
    # 18: Top green ratio - CT выше (НОВЫЙ)
    # 19: Beige ratio - T выше (НОВЫЙ)

    cluster_0_features = X[labels == 0].mean(axis=0)
    cluster_1_features = X[labels == 1].mean(axis=0)

    def compute_ct_score(features):
        """
        Вычисляет score для определения CT кластера.
        Положительные признаки для CT, отрицательные для T.
        """
        score = 0.0
        # CT признаки (положительные)
        score += features[6] * 2.0    # Green CT ratio (важно)
        score += features[13] * 1.5   # Olive/Khaki ratio
        score += features[15] * 1.0   # Navy ratio
        score += features[3] * 0.5    # G/R ratio
        score += features[18] * 1.5   # Top green ratio

        # T признаки (отрицательные для CT)
        score -= features[12] * 2.0   # White ratio (T маски)
        score -= features[7] * 1.0    # Brown ratio
        score -= features[8] * 0.5    # Gray ratio
        score -= features[17] * 1.5   # Top white ratio (T маски)
        score -= features[19] * 1.0   # Beige ratio
        score -= features[14] * 0.5   # Skin ratio

        return score

    score_0 = compute_ct_score(cluster_0_features)
    score_1 = compute_ct_score(cluster_1_features)

    if score_0 > score_1:
        ct_cluster = 0
        t_cluster = 1
    else:
        ct_cluster = 1
        t_cluster = 0

    print(f"\nColor-based heuristic assignment (improved):")
    print(f"  CT Score cluster 0: {score_0:.3f}")
    print(f"  CT Score cluster 1: {score_1:.3f}")
    print(f"\n  Cluster {ct_cluster} → CT (class 0)")
    print(f"    Green CT ratio: {X[labels==ct_cluster, 6].mean():.3f}")
    print(f"    Olive ratio: {X[labels==ct_cluster, 13].mean():.3f}")
    print(f"    Navy ratio: {X[labels==ct_cluster, 15].mean():.3f}")
    print(f"    Top green ratio: {X[labels==ct_cluster, 18].mean():.3f}")
    print(f"    White ratio: {X[labels==ct_cluster, 12].mean():.3f}")
    print(f"  Cluster {t_cluster} → T (class 1)")
    print(f"    White ratio: {X[labels==t_cluster, 12].mean():.3f}")
    print(f"    Brown ratio: {X[labels==t_cluster, 7].mean():.3f}")
    print(f"    Gray ratio: {X[labels==t_cluster, 8].mean():.3f}")
    print(f"    Beige ratio: {X[labels==t_cluster, 19].mean():.3f}")
    print(f"    Top white ratio: {X[labels==t_cluster, 17].mean():.3f}")

    # Присваиваем новые классы
    df['new_class'] = df['cluster'].apply(lambda c: 0 if c == ct_cluster else 1)

    ct_count = (df['new_class'] == 0).sum()
    t_count = (df['new_class'] == 1).sum()
    print(f"\nFinal assignment: {ct_count} CT (class 0), {t_count} T (class 1)")

    # Визуализация
    if visualize:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        ct_mask = df['new_class'] == 0
        t_mask = df['new_class'] == 1

        # 1. Scatter plot: Green CT ratio vs White ratio (ключевые признаки)
        ax1 = axes[0, 0]
        ax1.scatter(X[ct_mask, 6], X[ct_mask, 12], c='blue', label=f'CT: {ct_count}', alpha=0.5, s=30)
        ax1.scatter(X[t_mask, 6], X[t_mask, 12], c='red', label=f'T: {t_count}', alpha=0.5, s=30)
        ax1.set_xlabel('Green CT Ratio', fontsize=12)
        ax1.set_ylabel('White Ratio', fontsize=12)
        ax1.set_title('Green vs White (key features)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Scatter plot: Olive ratio vs Beige ratio
        ax2 = axes[0, 1]
        ax2.scatter(X[ct_mask, 13], X[ct_mask, 19], c='blue', label=f'CT: {ct_count}', alpha=0.5, s=30)
        ax2.scatter(X[t_mask, 13], X[t_mask, 19], c='red', label=f'T: {t_count}', alpha=0.5, s=30)
        ax2.set_xlabel('Olive/Khaki Ratio', fontsize=12)
        ax2.set_ylabel('Beige Ratio', fontsize=12)
        ax2.set_title('Olive vs Beige', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Scatter plot: Top green vs Top white (головная часть)
        ax3 = axes[0, 2]
        ax3.scatter(X[ct_mask, 18], X[ct_mask, 17], c='blue', label=f'CT: {ct_count}', alpha=0.5, s=30)
        ax3.scatter(X[t_mask, 18], X[t_mask, 17], c='red', label=f'T: {t_count}', alpha=0.5, s=30)
        ax3.set_xlabel('Top Green Ratio', fontsize=12)
        ax3.set_ylabel('Top White Ratio', fontsize=12)
        ax3.set_title('Top Region: Green vs White', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Scatter plot: Brown ratio vs Gray ratio
        ax4 = axes[1, 0]
        ax4.scatter(X[ct_mask, 7], X[ct_mask, 8], c='blue', label=f'CT: {ct_count}', alpha=0.5, s=30)
        ax4.scatter(X[t_mask, 7], X[t_mask, 8], c='red', label=f'T: {t_count}', alpha=0.5, s=30)
        ax4.set_xlabel('Brown Ratio', fontsize=12)
        ax4.set_ylabel('Gray Ratio', fontsize=12)
        ax4.set_title('Brown vs Gray', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Scatter plot: Navy ratio vs G/R ratio
        ax5 = axes[1, 1]
        ax5.scatter(X[ct_mask, 15], X[ct_mask, 3], c='blue', label=f'CT: {ct_count}', alpha=0.5, s=30)
        ax5.scatter(X[t_mask, 15], X[t_mask, 3], c='red', label=f'T: {t_count}', alpha=0.5, s=30)
        ax5.set_xlabel('Navy Ratio', fontsize=12)
        ax5.set_ylabel('G/R Ratio', fontsize=12)
        ax5.set_title('Navy vs G/R', fontsize=14)
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Feature importance bar chart
        ax6 = axes[1, 2]
        feature_names = ['Green CT', 'Olive', 'Navy', 'Top Green', 'White', 'Brown', 'Gray', 'Top White', 'Beige']
        feature_indices = [6, 13, 15, 18, 12, 7, 8, 17, 19]
        ct_means = [X[ct_mask, i].mean() for i in feature_indices]
        t_means = [X[t_mask, i].mean() for i in feature_indices]

        x_pos = np.arange(len(feature_names))
        width = 0.35
        ax6.bar(x_pos - width/2, ct_means, width, label='CT', color='blue', alpha=0.7)
        ax6.bar(x_pos + width/2, t_means, width, label='T', color='red', alpha=0.7)
        ax6.set_xlabel('Feature', fontsize=12)
        ax6.set_ylabel('Mean Value', fontsize=12)
        ax6.set_title('Feature Comparison CT vs T', fontsize=14)
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(feature_names, rotation=45, ha='right')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')

        plt.suptitle('CT/T Classification by Color Features (Improved)', fontsize=16, fontweight='bold')
        plt.tight_layout()

        vis_path = save_path or str(labels_path.parent / 'color_clustering.png')
        plt.savefig(vis_path, dpi=150)
        print(f"\nVisualization saved to {vis_path}")
        plt.show()

    return df


def classify_bboxes_by_position(labels_dir: str, visualize: bool = True, save_path: str = None) -> Dict[str, int]:
    """
    Кластеризует все YOLO bbox на два класса (CT=0, T=1) по позиции центра на экране.
    Использует K-means с k=2.

    Args:
        labels_dir: директория с YOLO .txt файлами (class x_center y_center w h)
        visualize: визуализировать результаты
        save_path: путь для сохранения визуализации

    Returns:
        dict: {filename: new_class_id} для каждого файла
    """
    labels_path = Path(labels_dir)

    if not labels_path.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")

    # Собираем все bbox
    bbox_data = []  # [(filename, x_center, y_center, w, h, original_class)]

    txt_files = list(labels_path.glob("*.txt"))
    print(f"Found {len(txt_files)} label files in {labels_dir}")

    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    bbox_data.append({
                        'filename': txt_file.name,
                        'filepath': txt_file,
                        'x_center': x_center,
                        'y_center': y_center,
                        'w': w,
                        'h': h,
                        'original_class': class_id
                    })

    if len(bbox_data) < 2:
        raise ValueError(f"Not enough bboxes to cluster (found {len(bbox_data)})")

    print(f"Loaded {len(bbox_data)} bounding boxes")

    # Создаём DataFrame
    df = pd.DataFrame(bbox_data)

    # K-means по центру bbox (x_center, y_center)
    X = df[['x_center', 'y_center']].values
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    df['cluster'] = labels
    cluster_centers = kmeans.cluster_centers_

    print(f"\nK-means clustering complete")
    print(f"Cluster 0 center: x={cluster_centers[0][0]:.3f}, y={cluster_centers[0][1]:.3f}")
    print(f"Cluster 1 center: x={cluster_centers[1][0]:.3f}, y={cluster_centers[1][1]:.3f}")

    # Определяем какой кластер CT (0), какой T (1)
    # Эвристика: CT обычно слева (меньший x) или выше (меньший y)
    # Используем x_center для определения
    cluster_0_x_mean = df[df['cluster'] == 0]['x_center'].mean()
    cluster_1_x_mean = df[df['cluster'] == 1]['x_center'].mean()

    # Кластер с меньшим средним X = CT (класс 0)
    if cluster_0_x_mean < cluster_1_x_mean:
        ct_cluster = 0
        t_cluster = 1
    else:
        ct_cluster = 1
        t_cluster = 0

    print(f"\nHeuristic assignment (by X position):")
    print(f"Cluster {ct_cluster} → CT (class 0), mean x={df[df['cluster']==ct_cluster]['x_center'].mean():.3f}")
    print(f"Cluster {t_cluster} → T (class 1), mean x={df[df['cluster']==t_cluster]['x_center'].mean():.3f}")

    # Присваиваем новые классы
    df['new_class'] = df['cluster'].apply(lambda c: 0 if c == ct_cluster else 1)

    ct_count = (df['new_class'] == 0).sum()
    t_count = (df['new_class'] == 1).sum()
    print(f"\nFinal assignment: {ct_count} CT (class 0), {t_count} T (class 1)")

    # Визуализация
    if visualize:
        plt.figure(figsize=(12, 8))

        # CT (class 0)
        ct_data = df[df['new_class'] == 0]
        plt.scatter(ct_data['x_center'], ct_data['y_center'],
                   c='blue', label=f'CT (class 0): {len(ct_data)}', alpha=0.5, s=30)

        # T (class 1)
        t_data = df[df['new_class'] == 1]
        plt.scatter(t_data['x_center'], t_data['y_center'],
                   c='red', label=f'T (class 1): {len(t_data)}', alpha=0.5, s=30)

        # Центры кластеров
        plt.scatter(cluster_centers[ct_cluster, 0], cluster_centers[ct_cluster, 1],
                   c='darkblue', marker='X', s=300, edgecolors='white', linewidths=2,
                   label='CT Center', zorder=10)
        plt.scatter(cluster_centers[t_cluster, 0], cluster_centers[t_cluster, 1],
                   c='darkred', marker='X', s=300, edgecolors='white', linewidths=2,
                   label='T Center', zorder=10)

        plt.xlabel('X Center (normalized)', fontsize=12)
        plt.ylabel('Y Center (normalized)', fontsize=12)
        plt.title('BBox Clustering: CT vs T by Screen Position', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().invert_yaxis()  # Y=0 сверху как на экране
        plt.tight_layout()

        vis_path = save_path or str(labels_path.parent / 'bbox_clustering.png')
        plt.savefig(vis_path, dpi=150)
        print(f"\nVisualization saved to {vis_path}")
        plt.show()

    return df


def update_labels_with_clusters(df: pd.DataFrame, backup: bool = True):
    """
    Обновляет YOLO label файлы с новыми классами из кластеризации.

    Args:
        df: DataFrame с колонками filepath, x_center, y_center, w, h, new_class
        backup: создавать бэкап оригинальных файлов
    """
    # Группируем по файлам
    grouped = df.groupby('filepath')

    updated_count = 0
    for filepath, group in grouped:
        filepath = Path(filepath)

        # Бэкап
        if backup:
            backup_path = filepath.with_suffix('.txt.bak')
            if not backup_path.exists():
                shutil.copy(filepath, backup_path)

        # Записываем новые аннотации
        with open(filepath, 'w') as f:
            for _, row in group.iterrows():
                line = f"{row['new_class']} {row['x_center']:.6f} {row['y_center']:.6f} {row['w']:.6f} {row['h']:.6f}\n"
                f.write(line)

        updated_count += 1

    print(f"Updated {updated_count} label files")
    if backup:
        print("Backups saved with .txt.bak extension")


def save_bbox_crops_by_cluster(df: pd.DataFrame, images_dir: str, output_dir: str, padding: float = 0.1):
    """
    Сохраняет вырезанные bbox изображения, разделённые по кластерам (CT/T).

    Args:
        df: DataFrame с колонками filename, x_center, y_center, w, h, new_class
        images_dir: директория с исходными изображениями
        output_dir: директория для сохранения кропов
        padding: добавочный отступ вокруг bbox (в долях от размера bbox)
    """
    images_path = Path(images_dir)
    output_path = Path(output_dir)

    # Создаём папки для CT и T
    ct_dir = output_path / "CT"
    t_dir = output_path / "T"
    ct_dir.mkdir(parents=True, exist_ok=True)
    t_dir.mkdir(parents=True, exist_ok=True)

    ct_count = 0
    t_count = 0
    errors = 0

    # Группируем по файлам (один файл = одно изображение)
    grouped = df.groupby('filename')

    for filename, group in tqdm(grouped, desc="Saving bbox crops"):
        # Имя изображения = имя label файла с другим расширением
        img_stem = Path(filename).stem

        # Пробуем разные расширения
        img_file = None
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = images_path / f"{img_stem}{ext}"
            if candidate.exists():
                img_file = candidate
                break

        if img_file is None:
            errors += 1
            continue

        # Загружаем изображение
        img = cv2.imread(str(img_file))
        if img is None:
            errors += 1
            continue

        img_h, img_w = img.shape[:2]

        # Обрабатываем каждый bbox в этом изображении
        for idx, row in group.iterrows():
            x_center = row['x_center']
            y_center = row['y_center']
            w = row['w']
            h = row['h']
            new_class = int(row['new_class'])

            # Конвертируем нормализованные координаты в пиксели
            bbox_w = int(w * img_w)
            bbox_h = int(h * img_h)
            bbox_cx = int(x_center * img_w)
            bbox_cy = int(y_center * img_h)

            # Добавляем padding
            pad_w = int(bbox_w * padding)
            pad_h = int(bbox_h * padding)

            x1 = max(0, bbox_cx - bbox_w // 2 - pad_w)
            y1 = max(0, bbox_cy - bbox_h // 2 - pad_h)
            x2 = min(img_w, bbox_cx + bbox_w // 2 + pad_w)
            y2 = min(img_h, bbox_cy + bbox_h // 2 + pad_h)

            # Вырезаем bbox
            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Определяем папку и имя файла
            if new_class == 0:
                out_dir = ct_dir
                out_name = f"ct_{ct_count:06d}.jpg"
                ct_count += 1
            else:
                out_dir = t_dir
                out_name = f"t_{t_count:06d}.jpg"
                t_count += 1

            # Сохраняем
            cv2.imwrite(str(out_dir / out_name), crop)

    print(f"\nSaved bbox crops:")
    print(f"  CT (class 0): {ct_count} images -> {ct_dir}")
    print(f"  T (class 1): {t_count} images -> {t_dir}")
    if errors > 0:
        print(f"  Errors (missing images): {errors}")


def main():
    """Основной скрипт для запуска кластеризации bbox."""
    import argparse

    parser = argparse.ArgumentParser(description='Classify YOLO bboxes into CT/T classes')
    parser.add_argument('--labels', type=str, default=r'D:\yolo_dataset\labels',
                        help='Directory with YOLO label files')
    parser.add_argument('--images', type=str, default=r'D:\yolo_dataset\images',
                        help='Directory with images (required for --method color and --save-crops)')
    parser.add_argument('--method', type=str, choices=['position', 'color'], default='color',
                        help='Clustering method: "position" (by bbox screen position) or "color" (by clothing color)')
    parser.add_argument('--update', action='store_true',
                        help='Update label files with new classes')
    parser.add_argument('--save-crops', action='store_true',
                        help='Save cropped bbox images separated by cluster (CT/T)')
    parser.add_argument('--crops-output', type=str, default=r'D:\yolo_dataset\bbox_crops',
                        help='Output directory for bbox crops')
    parser.add_argument('--padding', type=float, default=0.1,
                        help='Padding around bbox crops (fraction of bbox size)')
    parser.add_argument('--no-backup', action='store_true',
                        help='Do not create backup when updating')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Do not show visualization')
    args = parser.parse_args()

    labels_dir = args.labels
    images_dir = args.images

    if not Path(labels_dir).exists():
        print(f"Error: Labels directory not found at {labels_dir}")
        return

    print("="*60)
    print(f"YOLO BBox Classifier - K-means Clustering ({args.method.upper()})")
    print("="*60)

    # Кластеризация
    if args.method == 'color':
        # Для color метода нужны изображения
        if not Path(images_dir).exists():
            print(f"Error: Images directory not found at {images_dir}")
            print("Images are required for color-based clustering!")
            return

        df = classify_bboxes_by_color(
            labels_dir=labels_dir,
            images_dir=images_dir,
            visualize=not args.no_visualize,
            padding=args.padding
        )
    else:
        # Position-based clustering
        df = classify_bboxes_by_position(
            labels_dir=labels_dir,
            visualize=not args.no_visualize
        )

    # Обновляем файлы если указано
    if args.update:
        print("\n" + "="*60)
        print("Updating label files...")
        update_labels_with_clusters(df, backup=not args.no_backup)

    # Сохраняем кропы bbox по кластерам
    if args.save_crops:
        print("\n" + "="*60)
        print("Saving bbox crops by cluster...")
        if not Path(images_dir).exists():
            print(f"Error: Images directory not found at {images_dir}")
        else:
            save_bbox_crops_by_cluster(
                df=df,
                images_dir=images_dir,
                output_dir=args.crops_output,
                padding=args.padding
            )

    print("\n" + "="*60)
    print("Done!")
    if not args.update:
        print("Run with --update to apply new classes to label files")
    if not args.save_crops:
        print("Run with --save-crops to save bbox images by cluster")
    print("="*60)


if __name__ == "__main__":
    main()
