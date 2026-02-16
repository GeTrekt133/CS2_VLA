"""
Скрипт для анализа покрытия AR (Action Reasons) по доменам.

Мерджит violations_report.csv с удаленные-видео-ноябрь.csv
и анализирует, какая доля видео с каждым AR была предсказана как нарушение
по каждому домену (визуальный, текстовый, субтитры).
"""

import pandas as pd
from typing import Dict


# Словарь соответствия AR кодов и названий
AR_NAMES = {
    0: "-",
    1: "user_disactivated",
    2: "user_banned",
    3: "user_deleted_video",
    4: "moder_deleted_video",
    5: "new_video",
    6: "error_upload_video",
    7: "illegal_content",
    8: "delete_by_rightholder",
    9: "user_locked",
    10: "default_deleted_video",
    11: "default_does_not_exists_video",
    12: "default_hidden_video",
    13: "custom_restrict",
    14: "special",
    15: "converting_video",
    16: "downloading_video",
    17: "delete_roskomnadzor",
    20: "recording_video",
    21: "Удалено как СПАМ",
    22: "recording_video_error",
    23: "Мажор: серия 3",
    24: "purchase_paid",
    25: "purchase_failed",
    26: "purchase_canceled",
    27: "wait_livestream",
    28: "stop_livestream",
    29: "purchase_paymentawaiting",
    30: "user_inactive",
    31: "меморандум",
    32: "moderation",
    33: "checking rules",
    34: "illegal_check",
    35: "live_without_live",
    36: "source_deactivated",
    37: "comm1",
    38: "government_request",
    39: "copyright_abuse",
    40: "оbscene_video",
    41: "gambling_abuse",
    42: "financial_abuse",
    43: "meta",
    44: "shocking_content",
    45: "hack_abuse",
    46: "crypto_abuse",
    47: "education_abuse",
    48: "magic_abuse",
    49: "terror_abuse",
    50: "drug_abuse",
    51: "med_abuse",
    52: "polit_abuse",
    53: "AVITO_affront_abuse",
    54: "AVITO_weapon_abuse",
    55: "UGC_copyright_abuse",
    56: "UGC_financial_services",
    57: "UGC_betting_abuse",
    58: "UGC_med_abuse",
    59: "UGC_magic_abuse",
    60: "UGC_obscene_video",
    61: "UGC_shoсking_video",
    62: "UGC_financial_abuse",
    63: "UGC_gambling_abuse",
    64: "UGC_fake_abuse",
    65: "UGC_war_conseq",
    66: "UGC_terror_abuse",
    67: "UGC_dead_body",
    68: "UGC_money_abuse",
    69: "UGC_filthy_language",
    70: "UGC_polit_abuse",
    71: "UGC_SVO_fake_abuse",
    72: "UGC_personal_insults",
    73: "UGC_Nazi_abuse",
    74: "UGC_LGBT_abuse",
    75: "UGC_surveillance_video",
    76: "UGC_redirect_abuse",
    77: "UGC_person_clone",
    78: "UGC_harm_abuse",
    79: "UGC_spam_video",
    80: "UGC_terror_disclaimer",
    81: "UGC_inoagent_disclaimer",
    82: "UGC_hate_content",
    83: "UGC_call_terrorism",
    84: "UGC_call_unrest",
    85: "UGC_Child_abuse",
    86: "UGC_Suicide_abuse",
    87: "UGC_Text_autoban",
    88: "UGC_Drug_dealing",
    89: "122312321231",
    90: "UGC_RKN_waiting",
    91: "UGC_Child_abuse_no_send",
    92: "UGC_Suicide_abuse_no_send",
    93: "UGC_Drug_dealing_no_send",
    94: "Test_reason",
    95: "test_uniq_name",
    96: "UGC_inoagent_adv",
    97: "Аккаунт в процессе регистрации",
    98: "UGC_block_abuse",
    99: "UGC_alco_adv",
    100: "UGC_tobacco_adv",
    101: "UGC_jewelry_adv",
    102: "UGC_blogger_10k_adv",
    103: "UGC_donation_10k_abuse",
    104: "UGC_childfree_abuse",
    105: "UGC_rufing_abuse",
    106: "user_not_exist_in_rupass",
    107: "draft_video",
    108: "UGC_harmful_software_adv",
    109: "UGC_cheats_adv",
    110: "UGC_inc_dander_abuse",
    111: "UGC_inf_harm_abuse",
    112: "UGC_inv_minors_abuse",
    113: "UGC_illegal_info_abuse",
    114: "UGC_change_abuse",
    115: "UGC_medic_abuse",
    116: "UGC_discrediting_abuse",
    117: "UGC_redirection_abuse",
}


def load_data():
    """Загружает и мерджит данные."""
    print("=" * 70)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 70)

    # Загружаем отчет о предсказаниях
    df_predictions = pd.read_csv('violations_report.csv')
    print(f"\n[OK] violations_report.csv: {len(df_predictions):,} записей")

    # Загружаем ground truth данные
    df_ground_truth = pd.read_csv('удаленные-видео-ноябрь.csv')
    print(f"[OK] удаленные-видео-ноябрь.csv: {len(df_ground_truth):,} записей")

    # Переименовываем колонку для мерджа
    df_ground_truth = df_ground_truth.rename(columns={'ID видео': 'video_id'})

    # Мерджим по video_id
    df_merged = pd.merge(
        df_predictions,
        df_ground_truth[['video_id', 'Ризон видео']],
        on='video_id',
        how='inner'
    )

    print(f"\n[OK] Смерджено: {len(df_merged):,} записей")
    print(f"    Покрытие: {len(df_merged) / len(df_predictions) * 100:.2f}% от всех предсказаний")

    return df_merged


def analyze_ar_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Анализирует покрытие каждого AR по доменам.

    Returns:
        DataFrame со статистикой по каждому AR
    """
    print("\n" + "=" * 70)
    print("АНАЛИЗ ПОКРЫТИЯ AR ПО ДОМЕНАМ")
    print("=" * 70)

    # Получаем уникальные AR
    unique_ars = sorted(df['Ризон видео'].unique())

    results = []

    for ar in unique_ars:
        # Фильтруем видео с данным AR
        ar_videos = df[df['Ризон видео'] == ar]
        total = len(ar_videos)

        # Считаем предсказания по доменам
        visual_pred = ar_videos['visual_violation'].sum()
        text_pred = ar_videos['text_violation'].sum()
        subtitle_pred = ar_videos['subtitle_violation'].sum()
        any_pred = ar_videos['has_violation'].sum()

        # Вычисляем доли
        visual_pct = visual_pred / total * 100 if total > 0 else 0
        text_pct = text_pred / total * 100 if total > 0 else 0
        subtitle_pct = subtitle_pred / total * 100 if total > 0 else 0
        any_pct = any_pred / total * 100 if total > 0 else 0

        results.append({
            'AR': ar,
            'Name': AR_NAMES.get(int(ar), f'Unknown_{int(ar)}'),
            'Всего видео': total,
            'Визуальные': visual_pred,
            'Визуальные %': round(visual_pct, 2),
            'Текстовые': text_pred,
            'Текстовые %': round(text_pct, 2),
            'Субтитры': subtitle_pred,
            'Субтитры %': round(subtitle_pct, 2),
            'Любые': any_pred,
            'Любые %': round(any_pct, 2)
        })

    return pd.DataFrame(results)


def print_detailed_stats(df_stats: pd.DataFrame):
    """Выводит детальную статистику по AR."""
    print("\n--- ДЕТАЛЬНАЯ СТАТИСТИКА ПО AR ---\n")

    for _, row in df_stats.iterrows():
        ar = int(row['AR'])
        total = int(row['Всего видео'])

        print(f"AR {ar}: {total:,} видео")
        print(f"  Визуальные: {int(row['Визуальные']):,} ({row['Визуальные %']:.2f}%)")
        print(f"  Текстовые:  {int(row['Текстовые']):,} ({row['Текстовые %']:.2f}%)")
        print(f"  Субтитры:   {int(row['Субтитры']):,} ({row['Субтитры %']:.2f}%)")
        print(f"  Любые:      {int(row['Любые']):,} ({row['Любые %']:.2f}%)")
        print()


def print_top_coverage(df_stats: pd.DataFrame, domain: str, top_n: int = 10):
    """Выводит топ AR по покрытию для заданного домена."""
    col_name = f"{domain} %"

    print(f"--- ТОП-{top_n} AR ПО ПОКРЫТИЮ ({domain.upper()}) ---\n")

    top_ars = df_stats.nlargest(top_n, col_name)

    for i, (_, row) in enumerate(top_ars.iterrows(), 1):
        ar = int(row['AR'])
        pct = row[col_name]
        count = int(row[domain])
        total = int(row['Всего видео'])

        print(f"{i:2d}. AR {ar:3d}: {pct:6.2f}% ({count:,}/{total:,})")

    print()


def print_low_coverage(df_stats: pd.DataFrame, domain: str, threshold: float = 50.0):
    """Выводит AR с низким покрытием (ниже порога) для заданного домена."""
    col_name = f"{domain} %"

    low_coverage = df_stats[df_stats[col_name] < threshold].sort_values(col_name)

    if len(low_coverage) == 0:
        print(f"--- AR С ПОКРЫТИЕМ < {threshold}% ({domain.upper()}) ---\n")
        print(f"  Все AR имеют покрытие >= {threshold}%\n")
        return

    print(f"--- AR С ПОКРЫТИЕМ < {threshold}% ({domain.upper()}) ---\n")

    for _, row in low_coverage.iterrows():
        ar = int(row['AR'])
        pct = row[col_name]
        count = int(row[domain])
        total = int(row['Всего видео'])

        print(f"  AR {ar:3d}: {pct:6.2f}% ({count:,}/{total:,})")

    print()


def print_summary(df_stats: pd.DataFrame):
    """Выводит общую статистику."""
    print("=" * 70)
    print("ОБЩАЯ СТАТИСТИКА")
    print("=" * 70)

    total_videos = df_stats['Всего видео'].sum()
    total_visual = df_stats['Визуальные'].sum()
    total_text = df_stats['Текстовые'].sum()
    total_subtitle = df_stats['Субтитры'].sum()
    total_any = df_stats['Любые'].sum()

    print(f"\nВсего видео: {total_videos:,}")
    print(f"Уникальных AR: {len(df_stats)}")
    print()
    print("Общее покрытие по доменам:")
    print(f"  Визуальные: {total_visual:,} ({total_visual/total_videos*100:.2f}%)")
    print(f"  Текстовые:  {total_text:,} ({total_text/total_videos*100:.2f}%)")
    print(f"  Субтитры:   {total_subtitle:,} ({total_subtitle/total_videos*100:.2f}%)")
    print(f"  Любые:      {total_any:,} ({total_any/total_videos*100:.2f}%)")
    print()

    # Средние покрытия по AR (взвешенные по количеству видео)
    avg_visual = df_stats['Визуальные %'].mean()
    avg_text = df_stats['Текстовые %'].mean()
    avg_subtitle = df_stats['Субтитры %'].mean()
    avg_any = df_stats['Любые %'].mean()

    print("Среднее покрытие по AR (невзвешенное):")
    print(f"  Визуальные: {avg_visual:.2f}%")
    print(f"  Текстовые:  {avg_text:.2f}%")
    print(f"  Субтитры:   {avg_subtitle:.2f}%")
    print(f"  Любые:      {avg_any:.2f}%")
    print()


def save_results(df_stats: pd.DataFrame):
    """Сохраняет результаты в CSV."""
    print("=" * 70)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 70)

    # Сортируем по количеству видео (от большего к меньшему)
    df_stats_sorted = df_stats.sort_values('Всего видео', ascending=False)

    df_stats_sorted.to_csv('ar_coverage_report.csv', index=False, encoding='utf-8-sig')
    print("\n[OK] ar_coverage_report.csv")


def main():
    """Основная функция."""
    print("\n" + "=" * 70)
    print("АНАЛИЗ ПОКРЫТИЯ AR ПО ДОМЕНАМ")
    print("=" * 70)
    print()

    # Загружаем и мерджим данные
    df_merged = load_data()

    # Анализируем покрытие AR
    df_stats = analyze_ar_coverage(df_merged)

    # Выводим детальную статистику
    print_detailed_stats(df_stats)

    # Выводим топ AR по покрытию для каждого домена
    print_top_coverage(df_stats, 'Визуальные', top_n=10)
    print_top_coverage(df_stats, 'Текстовые', top_n=10)
    print_top_coverage(df_stats, 'Субтитры', top_n=10)

    # Выводим AR с низким покрытием
    print_low_coverage(df_stats, 'Визуальные', threshold=50.0)
    print_low_coverage(df_stats, 'Текстовые', threshold=50.0)
    print_low_coverage(df_stats, 'Субтитры', threshold=50.0)

    # Выводим общую статистику
    print_summary(df_stats)

    # Сохраняем результаты
    save_results(df_stats)

    print("\n" + "=" * 70)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 70)


if __name__ == "__main__":
    main()
