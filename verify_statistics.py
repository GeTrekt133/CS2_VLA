"""
Скрипт для проверки статистики из VIOLATIONS_ANALYSIS.md

Воспроизводит все расчеты и сравнивает с заявленными значениями.
"""

import pandas as pd
from typing import Dict, Tuple


class StatisticsVerifier:
    """Класс для проверки статистики нарушений."""

    # Ожидаемые значения из VIOLATIONS_ANALYSIS.md
    EXPECTED = {
        'total_videos': 45180,
        'visual_violations': 29448,
        'text_violations': 27097,
        'subtitle_violations': 20067,
        'only_visual': 7445,
        'only_text': 4852,
        'only_subtitle': 2028,
        'visual_text': 8409,
        'visual_subtitle': 4203,
        'text_subtitle': 4445,
        'all_three': 9391,
        'total_with_violations': 40773,
        'no_violations': 4407,
        'single_domain': 14325,
        'double_domain': 17057,
        'triple_domain': 9391,
    }

    def __init__(self, report_file='violations_report.csv'):
        """Инициализация с загрузкой данных."""
        self.df = pd.read_csv(report_file)
        self.actual = {}
        self.errors = []

    def calculate_statistics(self):
        """Вычисляет все статистические метрики."""
        print("=" * 70)
        print("РАСЧЕТ СТАТИСТИКИ")
        print("=" * 70)

        # Базовые метрики
        self.actual['total_videos'] = len(self.df)
        self.actual['visual_violations'] = self.df['visual_domen'].sum()
        self.actual['text_violations'] = self.df['text_domen'].sum()
        self.actual['subtitle_violations'] = self.df['subtitle_domen'].sum()

        # Одиночные домены
        self.actual['only_visual'] = self.df[
            (self.df['visual_domen']) &
            (~self.df['text_domen']) &
            (~self.df['subtitle_domen'])
        ].shape[0]

        self.actual['only_text'] = self.df[
            (~self.df['visual_domen']) &
            (self.df['text_domen']) &
            (~self.df['subtitle_domen'])
        ].shape[0]

        self.actual['only_subtitle'] = self.df[
            (~self.df['visual_domen']) &
            (~self.df['text_domen']) &
            (self.df['subtitle_domen'])
        ].shape[0]

        # Двойные комбинации
        self.actual['visual_text'] = self.df[
            (self.df['visual_domen']) &
            (self.df['text_domen']) &
            (~self.df['subtitle_domen'])
        ].shape[0]

        self.actual['visual_subtitle'] = self.df[
            (self.df['visual_domen']) &
            (~self.df['text_domen']) &
            (self.df['subtitle_domen'])
        ].shape[0]

        self.actual['text_subtitle'] = self.df[
            (~self.df['visual_domen']) &
            (self.df['text_domen']) &
            (self.df['subtitle_domen'])
        ].shape[0]

        # Тройные комбинации
        self.actual['all_three'] = self.df[
            (self.df['visual_domen']) &
            (self.df['text_domen']) &
            (self.df['subtitle_domen'])
        ].shape[0]

        # Итоговые метрики
        self.actual['total_with_violations'] = self.df['has_ban'].sum()
        self.actual['no_violations'] = (~self.df['has_ban']).sum()

        # Суммарные метрики по доменам
        self.actual['single_domain'] = (
            self.actual['only_visual'] +
            self.actual['only_text'] +
            self.actual['only_subtitle']
        )

        self.actual['double_domain'] = (
            self.actual['visual_text'] +
            self.actual['visual_subtitle'] +
            self.actual['text_subtitle']
        )

        self.actual['triple_domain'] = self.actual['all_three']

        print("[OK] Все метрики рассчитаны")

    def verify_metric(self, metric_name: str, description: str) -> bool:
        """
        Проверяет одну метрику.

        Returns:
            True если метрика совпадает, иначе False
        """
        expected = self.EXPECTED[metric_name]
        actual = self.actual[metric_name]

        is_valid = expected == actual
        status = "[OK]" if is_valid else "[FAIL]"

        total = self.actual['total_videos']
        expected_pct = expected / total * 100
        actual_pct = actual / total * 100

        print(f"{status} {description}")
        print(f"   Ожидалось: {expected:,} ({expected_pct:.2f}%)")
        print(f"   Получено:  {actual:,} ({actual_pct:.2f}%)")

        if not is_valid:
            diff = actual - expected
            print(f"   Разница: {diff:+,}")
            self.errors.append({
                'metric': metric_name,
                'description': description,
                'expected': expected,
                'actual': actual,
                'diff': diff
            })

        print()
        return is_valid

    def verify_all(self):
        """Проверяет все метрики."""
        print("\n" + "=" * 70)
        print("ПРОВЕРКА СООТВЕТСТВИЯ ЗАЯВЛЕННЫМ ЗНАЧЕНИЯМ")
        print("=" * 70)
        print()

        # Базовые метрики
        print("--- БАЗОВЫЕ МЕТРИКИ ---")
        print()
        self.verify_metric('total_videos', 'Всего видео')

        # Покрытие по доменам
        print("--- ПОКРЫТИЕ ПО ДОМЕНАМ ---")
        print()
        self.verify_metric('visual_violations', 'Визуальные нарушения')
        self.verify_metric('text_violations', 'Текстовые нарушения')
        self.verify_metric('subtitle_violations', 'Нарушения по субтитрам')

        # Одиночные домены
        print("--- ОДИНОЧНЫЕ ДОМЕНЫ ---")
        print()
        self.verify_metric('only_visual', 'Только визуальные')
        self.verify_metric('only_text', 'Только текстовые')
        self.verify_metric('only_subtitle', 'Только субтитры')
        self.verify_metric('single_domain', 'Итого одиночных')

        # Двойные комбинации
        print("--- ДВОЙНЫЕ КОМБИНАЦИИ ---")
        print()
        self.verify_metric('visual_text', 'Визуальные + Текстовые')
        self.verify_metric('visual_subtitle', 'Визуальные + Субтитры')
        self.verify_metric('text_subtitle', 'Текстовые + Субтитры')
        self.verify_metric('double_domain', 'Итого двойных')

        # Тройные комбинации
        print("--- ТРОЙНЫЕ КОМБИНАЦИИ ---")
        print()
        self.verify_metric('all_three', 'Все три домена')
        self.verify_metric('triple_domain', 'Итого тройных')

        # Итоговые метрики
        print("--- ИТОГОВЫЕ МЕТРИКИ ---")
        print()
        self.verify_metric('total_with_violations', 'Всего с нарушениями')
        self.verify_metric('no_violations', 'Без нарушений')

    def check_consistency(self):
        """Проверяет внутреннюю согласованность данных."""
        print("\n" + "=" * 70)
        print("ПРОВЕРКА ВНУТРЕННЕЙ СОГЛАСОВАННОСТИ")
        print("=" * 70)
        print()

        checks = []

        # Проверка 1: Сумма одиночных + двойных + тройных = всего с нарушениями
        sum_combinations = (
            self.actual['single_domain'] +
            self.actual['double_domain'] +
            self.actual['triple_domain']
        )
        check1 = sum_combinations == self.actual['total_with_violations']
        checks.append(check1)

        status1 = "[OK]" if check1 else "[FAIL]"
        print(f"{status1} Сумма комбинаций = Всего с нарушениями")
        print(f"   {self.actual['single_domain']:,} + {self.actual['double_domain']:,} + "
              f"{self.actual['triple_domain']:,} = {sum_combinations:,}")
        print(f"   Ожидалось: {self.actual['total_with_violations']:,}")
        if not check1:
            print(f"   Разница: {sum_combinations - self.actual['total_with_violations']:+,}")
        print()

        # Проверка 2: Всего с нарушениями + без нарушений = всего видео
        sum_total = self.actual['total_with_violations'] + self.actual['no_violations']
        check2 = sum_total == self.actual['total_videos']
        checks.append(check2)

        status2 = "[OK]" if check2 else "[FAIL]"
        print(f"{status2} С нарушениями + Без нарушений = Всего видео")
        print(f"   {self.actual['total_with_violations']:,} + {self.actual['no_violations']:,} = {sum_total:,}")
        print(f"   Ожидалось: {self.actual['total_videos']:,}")
        if not check2:
            print(f"   Разница: {sum_total - self.actual['total_videos']:+,}")
        print()

        # Проверка 3: Распределение по количеству доменов
        print("--- РАСПРЕДЕЛЕНИЕ ПО КОЛИЧЕСТВУ ДОМЕНОВ ---")
        print()

        num_domains = (
            self.df['visual_domen'].astype(int) +
            self.df['text_domen'].astype(int) +
            self.df['subtitle_domen'].astype(int)
        )

        for i in range(4):
            count = (num_domains == i).sum()
            pct = count / self.actual['total_videos'] * 100
            label = "доменов (чистые)" if i == 0 else "домен" if i == 1 else "домена"
            print(f"   {i} {label}: {count:,} ({pct:.2f}%)")

        return all(checks)

    def print_summary(self):
        """Выводит итоговую сводку."""
        print("\n" + "=" * 70)
        print("ИТОГОВАЯ СВОДКА")
        print("=" * 70)
        print()

        if not self.errors:
            print("[OK] ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
            print("  Статистика в VIOLATIONS_ANALYSIS.md полностью соответствует данным.")
        else:
            print(f"[FAIL] ОБНАРУЖЕНО ОШИБОК: {len(self.errors)}")
            print()
            print("Несоответствия:")
            for err in self.errors:
                print(f"  - {err['description']}: {err['actual']:,} вместо {err['expected']:,} "
                      f"(разница: {err['diff']:+,})")

        print()

    def save_report(self, filename='verification_report.txt'):
        """Сохраняет отчет в файл."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ О ПРОВЕРКЕ СТАТИСТИКИ\n")
            f.write("=" * 70 + "\n\n")

            f.write("Исходный файл: violations_report.csv\n")
            f.write(f"Всего записей: {len(self.df):,}\n\n")

            if not self.errors:
                f.write("РЕЗУЛЬТАТ: [OK] ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ\n")
            else:
                f.write(f"РЕЗУЛЬТАТ: [FAIL] ОБНАРУЖЕНО {len(self.errors)} ОШИБОК\n\n")
                f.write("Несоответствия:\n")
                for err in self.errors:
                    f.write(f"  {err['description']}:\n")
                    f.write(f"    Ожидалось: {err['expected']:,}\n")
                    f.write(f"    Получено:  {err['actual']:,}\n")
                    f.write(f"    Разница:   {err['diff']:+,}\n\n")

            f.write("\nДетальная статистика:\n")
            f.write("-" * 70 + "\n")
            for metric, value in sorted(self.actual.items()):
                expected = self.EXPECTED.get(metric, '-')
                pct = value / self.actual['total_videos'] * 100
                f.write(f"{metric:30s}: {value:>6,} ({pct:>5.2f}%) [ожидалось: {expected}]\n")

        print(f"Отчет сохранен: {filename}")


def main():
    """Основная функция."""
    print("\n" + "=" * 70)
    print("ПРОВЕРКА СТАТИСТИКИ ИЗ VIOLATIONS_ANALYSIS.md")
    print("=" * 70)
    print()

    # Создаем верификатор
    verifier = StatisticsVerifier()

    # Рассчитываем статистику
    verifier.calculate_statistics()

    # Проверяем соответствие
    verifier.verify_all()

    # Проверяем согласованность
    verifier.check_consistency()

    # Выводим сводку
    verifier.print_summary()

    # Сохраняем отчет
    verifier.save_report()

    print("\n" + "=" * 70)
    print("ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 70)

    # Возвращаем код выхода
    return 0 if not verifier.errors else 1


if __name__ == "__main__":
    exit(main())
