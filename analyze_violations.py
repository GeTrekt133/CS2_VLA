"""
Скрипт для анализа нарушений по доменам из violations_report.csv

Использование:
    python analyze_violations.py
"""

import pandas as pd


def analyze_violations(report_file='violations_report.csv'):
    """Анализирует нарушения по доменам."""

    print("=" * 70)
    print("АНАЛИЗ НАРУШЕНИЙ ПО ДОМЕНАМ")
    print("=" * 70)

    # Загружаем данные
    df = pd.read_csv(report_file)
    total = len(df)

    print(f"\nВсего видео: {total:,}")

    # Одиночные домены
    print("\n" + "=" * 70)
    print("ПОКРЫТИЕ ПО ДОМЕНАМ (хотя бы одно нарушение)")
    print("=" * 70)

    visual = df['visual_violation'].sum()
    text = df['text_violation'].sum()
    subtitle = df['subtitle_violation'].sum()

    print(f"\n1. Визуальные нарушения")
    print(f"   Количество: {visual:,} ({visual/total*100:.2f}%)")

    print(f"\n2. Текстовые нарушения")
    print(f"   Количество: {text:,} ({text/total*100:.2f}%)")

    print(f"\n3. Нарушения по субтитрам")
    print(f"   Количество: {subtitle:,} ({subtitle/total*100:.2f}%)")

    # Комбинации
    print("\n" + "=" * 70)
    print("КОМБИНАЦИИ ДОМЕНОВ")
    print("=" * 70)

    combinations = {
        'Только визуальные': (df['visual_violation']) & (~df['text_violation']) & (~df['subtitle_violation']),
        'Только текстовые': (~df['visual_violation']) & (df['text_violation']) & (~df['subtitle_violation']),
        'Только субтитры': (~df['visual_violation']) & (~df['text_violation']) & (df['subtitle_violation']),
        'Визуальные + Текстовые': (df['visual_violation']) & (df['text_violation']) & (~df['subtitle_violation']),
        'Визуальные + Субтитры': (df['visual_violation']) & (~df['text_violation']) & (df['subtitle_violation']),
        'Текстовые + Субтитры': (~df['visual_violation']) & (df['text_violation']) & (df['subtitle_violation']),
        'Все три домена': (df['visual_violation']) & (df['text_violation']) & (df['subtitle_violation'])
    }

    print("\n--- Одиночные домены ---")
    for i, (name, mask) in enumerate(list(combinations.items())[:3], 1):
        count = mask.sum()
        print(f"{i}. {name}: {count:,} ({count/total*100:.2f}%)")

    print("\n--- Двойные комбинации ---")
    for i, (name, mask) in enumerate(list(combinations.items())[3:6], 4):
        count = mask.sum()
        print(f"{i}. {name}: {count:,} ({count/total*100:.2f}%)")

    print("\n--- Тройные комбинации ---")
    count = combinations['Все три домена'].sum()
    print(f"7. Все три домена: {count:,} ({count/total*100:.2f}%)")

    # Итоговая статистика
    print("\n" + "=" * 70)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 70)

    total_violations = df['has_violation'].sum()
    no_violations = (~df['has_violation']).sum()

    print(f"\nВсего с нарушениями: {total_violations:,} ({total_violations/total*100:.2f}%)")
    print(f"Без нарушений: {no_violations:,} ({no_violations/total*100:.2f}%)")

    # Распределение по количеству доменов
    print("\n" + "=" * 70)
    print("РАСПРЕДЕЛЕНИЕ ПО КОЛИЧЕСТВУ ДОМЕНОВ")
    print("=" * 70)

    num_domains = (
        df['visual_violation'].astype(int) +
        df['text_violation'].astype(int) +
        df['subtitle_violation'].astype(int)
    )

    for i in range(4):
        count = (num_domains == i).sum()
        label = "доменов (чистые)" if i == 0 else f"{'домен' if i == 1 else 'домена' if i <= 3 else 'доменов'}"
        print(f"{i} {label}: {count:,} ({count/total*100:.2f}%)")

    # Сохранение отчета
    print("\n" + "=" * 70)
    print("СОХРАНЕНИЕ ОТЧЕТА")
    print("=" * 70)

    report_data = []

    for name, mask in combinations.items():
        count = mask.sum()
        report_data.append({
            'Тип нарушения': name,
            'Количество': count,
            'Процент': round(count/total*100, 2)
        })

    report_data.append({
        'Тип нарушения': 'ИТОГО: С нарушениями',
        'Количество': total_violations,
        'Процент': round(total_violations/total*100, 2)
    })

    report_data.append({
        'Тип нарушения': 'ИТОГО: Без нарушений',
        'Количество': no_violations,
        'Процент': round(no_violations/total*100, 2)
    })

    report_df = pd.DataFrame(report_data)
    report_df.to_csv('violations_analysis.csv', index=False, encoding='utf-8-sig')

    print("[OK] violations_analysis.csv")

    # Ключевые выводы
    print("\n" + "=" * 70)
    print("КЛЮЧЕВЫЕ ВЫВОДЫ")
    print("=" * 70)

    print(f"\n1. {total_violations/total*100:.1f}% видео имеют хотя бы одно нарушение")
    print(f"2. Визуальные нарушения - самый распространенный тип ({visual/total*100:.1f}%)")
    print(f"3. {combinations['Все три домена'].sum():,} видео ({combinations['Все три домена'].sum()/total*100:.1f}%) нарушают все три домена")
    print(f"4. Только {no_violations/total*100:.1f}% видео полностью чистые")

    return report_df


if __name__ == "__main__":
    analyze_violations()
