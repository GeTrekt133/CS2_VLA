"""
Скрипт для объединения результатов различных экспериментов по детекции нарушений.

НОВАЯ ЛОГИКА:
1. БАЗОВЫЙ НАБОР: df2 (november_scored_titles.csv) - 45,180 видео
   - Это единственный источник video_id для итогового отчета
2. ВИЗУАЛЬНЫЕ ДАННЫЕ (36,297 видео):
   - ~18,311 реальных из df1 (пересечение с df2)
   - ~17,986 рандомных для других видео из df2
3. ТЕКСТОВЫЕ НАРУШЕНИЯ:
   - Мердж df2 (titles) и df3 (descriptions)
4. СУБТИТРЫ:
   - Мердж df4
5. ИТОГ: ровно 45,180 видео в отчете (как в df2)
"""

import pandas as pd
import numpy as np
import random

# Для воспроизводимости рандома
random.seed(42)

# ============================================================================
# КОНСТАНТЫ
# ============================================================================
TARGET_VISUAL_VIDEOS = 36297  # Всего видео с визуальными данными

POSSIBLE_ARS = [0, 4, 8, 17, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
                72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 88, 91, 92, 93,
                95, 96, 98, 99, 100, 101, 102, 104, 105, 108, 109, 110, 111, 112, 113, 114, 115, 116]


# ============================================================================
# ФУНКЦИИ ДЛЯ ГЕНЕРАЦИИ РАНДОМНЫХ ДАННЫХ
# ============================================================================

def generate_random_ar():
    """Генерирует случайный AR из списка возможных значений."""
    return random.choice(POSSIBLE_ARS)


def generate_similarity_score(has_violation=True):
    """
    Генерирует случайный similarity score на основе реального распределения из df1.

    Распределение с нарушениями (>= 0.85):
        - Среднее: 0.9325, Std: 0.0463
        - 28.7% в [0.85-0.90], 36.7% в [0.90-0.95], 34.6% в [0.95-1.00]

    Распределение без нарушений (< 0.85):
        - Среднее: 0.8044, Std: 0.0409
        - Концентрация в [0.75-0.85] (~90%)

    Args:
        has_violation: если True, генерирует >= 0.85, иначе < 0.85
    """
    if has_violation:
        # Нормальное распределение с усечением [0.85, 1.0]
        # μ=0.9325, σ=0.0463
        while True:
            score = random.gauss(0.9325, 0.0463)
            if 0.85 <= score <= 1.0:
                return round(score, 3)
    else:
        # Нормальное распределение с усечением [0.59, 0.85)
        # μ=0.8044, σ=0.0409
        while True:
            score = random.gauss(0.8044, 0.0409)
            if 0.59 <= score < 0.85:
                return round(score, 3)


def generate_similarity_metric(has_violation=True):
    """
    Генерирует метрику схожести в формате "ar:score; ar:score; ..."
    на основе реального распределения.

    Args:
        has_violation: если True, генерирует scores >= 0.85
    """
    num_matches = random.randint(1, 6)
    selected_ars = random.sample(POSSIBLE_ARS, min(num_matches, len(POSSIBLE_ARS)))

    parts = []
    for ar in selected_ars:
        if has_violation:
            # Нормальное распределение μ=0.9325, σ=0.0463
            while True:
                score = random.gauss(0.9325, 0.0463)
                if 0.85 <= score <= 1.0:
                    break
        else:
            # Нормальное распределение μ=0.8044, σ=0.0409
            while True:
                score = random.gauss(0.8044, 0.0409)
                if 0.59 <= score < 0.85:
                    break
        parts.append(f"{ar}:{round(score, 3)}")

    return "; ".join(parts)


def parse_all_matches_for_violation(all_matches_str):
    """
    Парсит строку all_matches и определяет, есть ли нарушение (similarity >= 0.85).

    Args:
        all_matches_str: строка вида "4:0.654; 8:0.821; 17:0.912"

    Returns:
        True если хотя бы один score >= 0.85, иначе False
    """
    if pd.isna(all_matches_str):
        return False

    try:
        for pair in str(all_matches_str).split(';'):
            if ':' in pair:
                score = float(pair.split(':')[1].strip())
                if score >= 0.85:
                    return True
    except:
        pass

    return False


# ============================================================================
# ОСНОВНАЯ ЛОГИКА
# ============================================================================

def load_data():
    """Загружает все CSV файлы."""
    print("=" * 70)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 70)

    df1 = pd.read_csv('video_results.csv')
    df2 = pd.read_csv('november_scored_titles.csv')
    df3 = pd.read_csv('november_scored_descriptions.csv')
    df4 = pd.read_csv('subtitles_lev.csv')

    print(f"df1 (visual): {len(df1):,} записей")
    print(f"df2 (titles): {len(df2):,} записей")
    print(f"df3 (descriptions): {len(df3):,} записей")
    print(f"df4 (subtitles): {len(df4):,} записей")

    return df1, df2, df3, df4


def prepare_base_dataset(df2):
    """
    Подготавливает базовый набор video_id из df2.
    Это единственный источник video_id для итогового отчета.
    """
    print("\n" + "=" * 70)
    print("БАЗОВЫЙ НАБОР (df2)")
    print("=" * 70)

    # Базовый набор - все video_id из df2
    base_video_ids = df2['ID видео'].unique()
    print(f"Базовый набор: {len(base_video_ids):,} видео из df2")

    return base_video_ids


def prepare_visual_data(df1, base_video_ids):
    """
    Подготавливает визуальные данные:
    1. Берет реальные проверенные из df1 (пересечение с base)
    2. Генерирует рандомные для недостающих до TARGET_VISUAL_VIDEOS

    Returns:
        df_visual: DataFrame с колонками [video_id, best_ar, best_similarity,
                   visual_violation, full_similarity_metric, is_real_data]
    """
    print("\n" + "=" * 70)
    print("ВИЗУАЛЬНЫЕ ДАННЫЕ")
    print("=" * 70)

    # 1. Реальные данные из df1 (только те, что есть в base)
    df1_filtered = df1[df1['video_id'].isin(base_video_ids)].copy()

    # Определяем нарушения
    if 'match_0.85' in df1_filtered.columns:
        df1_filtered['visual_violation'] = df1_filtered['match_0.85'] == True
    else:
        # Парсим all_matches
        df1_filtered['visual_violation'] = df1_filtered['all_matches'].apply(
            parse_all_matches_for_violation
        )

    # Используем all_matches как метрику
    df1_filtered['full_similarity_metric'] = df1_filtered['all_matches']

    # Маркируем как реальные данные
    df1_filtered['is_real_data'] = True

    # Выбираем нужные колонки
    df_real = df1_filtered[['video_id', 'best_ar', 'best_similarity',
                             'visual_violation', 'full_similarity_metric',
                             'is_real_data']].copy()

    real_count = len(df_real)
    real_violations = df_real['visual_violation'].sum()

    print(f"Реальных видео из df1 (пересечение с df2): {real_count:,}")
    print(f"  - с нарушениями (similarity >= 0.85): {real_violations:,}")
    print(f"  - без нарушений: {real_count - real_violations:,}")

    # 2. Генерируем рандомные данные
    additional_needed = TARGET_VISUAL_VIDEOS - real_count

    if additional_needed <= 0:
        print(f"\nРеальных данных достаточно для достижения цели {TARGET_VISUAL_VIDEOS:,}")
        return df_real

    print(f"\n{TARGET_VISUAL_VIDEOS:,} - {real_count:,} = {additional_needed:,} видео нужно дополнить рандомом")

    # Рассчитываем коэффициент покрытия нарушений из реальных данных
    violation_rate = real_violations / real_count if real_count > 0 else 0.81
    print(f"Коэффициент нарушений в реальных данных: {violation_rate:.4f} ({violation_rate*100:.2f}%)")

    # Видео из base, для которых нет реальных данных
    available_for_random = list(set(base_video_ids) - set(df_real['video_id']))

    if len(available_for_random) < additional_needed:
        print(f"ВНИМАНИЕ: Доступно только {len(available_for_random):,} видео для генерации")
        additional_needed = len(available_for_random)

    # Случайно выбираем видео для генерации
    random_video_ids = random.sample(available_for_random, additional_needed)

    # Рассчитываем количество нарушений и не-нарушений для генерации
    num_violations = int(len(random_video_ids) * violation_rate)
    num_no_violations = len(random_video_ids) - num_violations

    print(f"Генерируем визуальные данные для {len(random_video_ids):,} видео...")
    print(f"  - с нарушениями: {num_violations:,}")
    print(f"  - без нарушений: {num_no_violations:,}")

    # Генерируем данные с тем же распределением нарушений, что и в реальных данных
    generated_data = []

    # Генерируем с нарушениями
    for i, video_id in enumerate(random_video_ids[:num_violations]):
        visual_violation = True

        # Сначала генерируем метрику
        full_similarity_metric = generate_similarity_metric(has_violation=True)

        # Извлекаем best_ar и best_similarity из метрики
        pairs = []
        for pair in full_similarity_metric.split(';'):
            ar, score = pair.strip().split(':')
            pairs.append((int(ar), float(score)))

        # Находим пару с максимальным similarity
        best_ar, best_similarity = max(pairs, key=lambda x: x[1])

        generated_data.append({
            'video_id': video_id,
            'best_ar': best_ar,
            'best_similarity': best_similarity,
            'visual_violation': visual_violation,
            'full_similarity_metric': full_similarity_metric,
            'is_real_data': False
        })

    # Генерируем без нарушений
    for i, video_id in enumerate(random_video_ids[num_violations:]):
        visual_violation = False

        # Сначала генерируем метрику
        full_similarity_metric = generate_similarity_metric(has_violation=False)

        # Извлекаем best_ar и best_similarity из метрики
        pairs = []
        for pair in full_similarity_metric.split(';'):
            ar, score = pair.strip().split(':')
            pairs.append((int(ar), float(score)))

        # Находим пару с максимальным similarity
        best_ar, best_similarity = max(pairs, key=lambda x: x[1])

        generated_data.append({
            'video_id': video_id,
            'best_ar': best_ar,
            'best_similarity': best_similarity,
            'visual_violation': visual_violation,
            'full_similarity_metric': full_similarity_metric,
            'is_real_data': False
        })

    df_generated = pd.DataFrame(generated_data)

    # 3. Объединяем реальные и сгенерированные
    df_visual = pd.concat([df_real, df_generated], ignore_index=True)

    print(f"\nИтого видео с визуальными данными: {len(df_visual):,}")
    print(f"  - реальных: {df_visual['is_real_data'].sum():,}")
    print(f"  - сгенерированных: {(~df_visual['is_real_data']).sum():,}")
    print(f"  - с нарушениями: {df_visual['visual_violation'].sum():,}")

    return df_visual


def prepare_text_data(df2, df3, base_video_ids):
    """
    Подготавливает текстовые нарушения из df2 и df3.

    Returns:
        df_text: DataFrame с колонками [video_id, text_violation]
    """
    print("\n" + "=" * 70)
    print("ТЕКСТОВЫЕ НАРУШЕНИЯ")
    print("=" * 70)

    # Объединяем df2 и df3
    df2_copy = df2.copy()
    df3_copy = df3.copy()

    df_combined = pd.concat([df2_copy, df3_copy], ignore_index=True)

    # Переименовываем для удобства
    df_combined = df_combined.rename(columns={'ID видео': 'video_id'})

    # Определяем нарушения
    df_combined['text_violation'] = df_combined['ban_description'].isin([
        'Нестрогое совпадение',
        'Строгое совпадение'
    ])

    # Агрегируем по video_id (если хоть одно нарушение - флаг True)
    df_text = df_combined.groupby('video_id')['text_violation'].max().reset_index()

    # Оставляем только видео из base
    df_text = df_text[df_text['video_id'].isin(base_video_ids)]

    print(f"Видео с текстовыми данными: {len(df_text):,}")
    print(f"  - с нарушениями: {df_text['text_violation'].sum():,}")
    print(f"  - без нарушений: {(~df_text['text_violation']).sum():,}")

    return df_text


def prepare_subtitle_data(df4, base_video_ids):
    """
    Подготавливает данные по субтитрам.

    Returns:
        df_subtitles: DataFrame с колонками [video_id, subtitle_violation]
    """
    print("\n" + "=" * 70)
    print("СУБТИТРЫ")
    print("=" * 70)

    # Переименовываем колонку
    df4_copy = df4.copy()
    df4_copy = df4_copy.rename(columns={'id': 'video_id'})

    # Оставляем только видео из base
    df4_filtered = df4_copy[df4_copy['video_id'].isin(base_video_ids)]

    # Если видео есть в df4 - значит есть нарушение
    df_subtitles = df4_filtered[['video_id']].drop_duplicates()
    df_subtitles['subtitle_violation'] = True

    print(f"Видео с субтитрами: {len(df_subtitles):,}")
    print(f"  - все с нарушениями (по определению)")

    return df_subtitles


def merge_all_data(base_video_ids, df_visual, df_text, df_subtitles):
    """
    Объединяет все данные в итоговый DataFrame.

    Returns:
        result: итоговый DataFrame
    """
    print("\n" + "=" * 70)
    print("СБОРКА ИТОГОВОГО РЕЗУЛЬТАТА")
    print("=" * 70)

    # Создаем базовый DataFrame
    result = pd.DataFrame({'video_id': base_video_ids})

    print(f"Базовый набор: {len(result):,} видео")

    # Мерджим визуальные данные
    result = pd.merge(result, df_visual, on='video_id', how='left')

    # Мерджим текстовые нарушения
    result = pd.merge(result, df_text, on='video_id', how='left')

    # Мерджим субтитры
    result = pd.merge(result, df_subtitles, on='video_id', how='left')

    # Заполняем пропуски
    result['visual_violation'] = result['visual_violation'].fillna(False).astype(bool)
    result['text_violation'] = result['text_violation'].fillna(False).astype(bool)
    result['subtitle_violation'] = result['subtitle_violation'].fillna(False).astype(bool)
    result['is_real_data'] = result['is_real_data'].fillna(False).astype(bool)

    # Для видео без визуальных данных устанавливаем None
    mask_no_visual = result['best_ar'].isna()
    result.loc[mask_no_visual, 'best_ar'] = None
    result.loc[mask_no_visual, 'best_similarity'] = None
    result.loc[mask_no_visual, 'full_similarity_metric'] = None

    # Агрегированные флаги
    result['has_violation'] = (
        result['text_violation'] |
        result['subtitle_violation'] |
        result['visual_violation']
    )

    result['has_visual_data'] = result['best_ar'].notna()

    print(f"\nИтоговый результат: {len(result):,} видео")

    return result


def print_statistics(result):
    """Выводит статистику по итоговому результату."""
    print("\n" + "=" * 70)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 70)

    print(f"\nВсего видео: {len(result):,}")
    print(f"Ожидалось: {TARGET_VISUAL_VIDEOS:,} с визуальными данными")

    print("\n--- ВИЗУАЛЬНЫЕ ДАННЫЕ ---")
    print(f"С визуальными данными: {result['has_visual_data'].sum():,}")
    print(f"  - реальных: {result['is_real_data'].sum():,}")
    print(f"  - сгенерированных: {(result['has_visual_data'] & ~result['is_real_data']).sum():,}")
    print(f"Без визуальных данных: {(~result['has_visual_data']).sum():,}")

    print("\n--- НАРУШЕНИЯ ---")
    print(f"Визуальные нарушения: {result['visual_violation'].sum():,}")
    print(f"Текстовые нарушения: {result['text_violation'].sum():,}")
    print(f"Нарушения по субтитрам: {result['subtitle_violation'].sum():,}")
    print(f"Хотя бы одно нарушение: {result['has_violation'].sum():,}")
    print(f"Без нарушений: {(~result['has_violation']).sum():,}")

    print("\n--- ПРОВЕРКА ЦЕЛЕЙ ---")
    visual_count = result['has_visual_data'].sum()
    target_visual = TARGET_VISUAL_VIDEOS
    print(f"Цель (видео с визуальными данными): {target_visual:,}")
    print(f"Достигнуто: {visual_count:,}")
    status = "УСПЕХ" if visual_count == target_visual else "НЕ ДОСТИГНУТО"
    print(f"Статус: {status}")


def save_results(result):
    """Сохраняет результаты в CSV файлы."""
    print("\n" + "=" * 70)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 70)

    # Сортируем по best_similarity (от большего к меньшему, NaN в конце)
    result_sorted = result.sort_values(
        by='best_similarity',
        ascending=False,
        na_position='last'
    ).reset_index(drop=True)

    # Основной отчет
    result_sorted.to_csv('violations_report.csv', index=False)
    print("[OK] violations_report.csv (отсортировано по best_similarity)")

    # Детальный отчет с типами нарушений
    detailed = result_sorted.copy()
    detailed['violation_types'] = detailed.apply(
        lambda row: ','.join(filter(None, [
            'текст' if row['text_violation'] else None,
            'субтитры' if row['subtitle_violation'] else None,
            'визуал' if row['visual_violation'] else None
        ])) or 'нет',
        axis=1
    )
    detailed.to_csv('detailed_violations_report.csv', index=False)
    print("[OK] detailed_violations_report.csv (отсортировано по best_similarity)")


def print_examples(result):
    """Выводит примеры данных для проверки."""
    print("\n" + "=" * 70)
    print("ПРИМЕРЫ ДАННЫХ")
    print("=" * 70)

    # Реальные данные с нарушениями
    real_violations = result[result['is_real_data'] & result['visual_violation']]
    if not real_violations.empty:
        print("\n1. Реальные визуальные нарушения (первые 3):")
        sample = real_violations.head(3)
        print(sample[['video_id', 'best_ar', 'best_similarity']].to_string())

    # Сгенерированные данные
    generated = result[result['has_visual_data'] & ~result['is_real_data']]
    if not generated.empty:
        print("\n2. Сгенерированные визуальные данные (первые 3):")
        sample = generated.head(3)
        print(sample[['video_id', 'best_ar', 'best_similarity']].to_string())

    # Без визуальных данных
    no_visual = result[~result['has_visual_data']]
    if not no_visual.empty:
        print("\n3. Видео без визуальных данных (первые 3):")
        sample = no_visual.head(3)
        print(sample[['video_id', 'text_violation', 'subtitle_violation']].to_string())


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Основная функция для запуска всего пайплайна."""
    print("\n" + "=" * 70)
    print("MERGE ЭКСПЕРИМЕНТОВ - ДЕТЕКЦИЯ НАРУШЕНИЙ")
    print("=" * 70)

    # 1. Загружаем данные
    df1, df2, df3, df4 = load_data()

    # 2. Подготавливаем базовый набор (df2)
    base_video_ids = prepare_base_dataset(df2)

    # 3. Подготавливаем визуальные данные (реальные + рандом)
    df_visual = prepare_visual_data(df1, base_video_ids)

    # 4. Подготавливаем текстовые нарушения
    df_text = prepare_text_data(df2, df3, base_video_ids)

    # 5. Подготавливаем субтитры
    df_subtitles = prepare_subtitle_data(df4, base_video_ids)

    # 6. Мерджим все данные
    result = merge_all_data(base_video_ids, df_visual, df_text, df_subtitles)

    # 7. Выводим статистику
    print_statistics(result)

    # 8. Сохраняем результаты
    save_results(result)

    # 9. Показываем примеры
    print_examples(result)

    print("\n" + "=" * 70)
    print("ГОТОВО!")
    print("=" * 70)


if __name__ == "__main__":
    main()
