"""
Скрипт для запуска полного пайплайна разметки CT/T для YOLO датасета.

Этапы:
1. Кластеризация команд (team_classifier.py)
2. Обработка скриншотов с разметкой классов (gt_find.py)
"""

import sys
from pathlib import Path

# Добавляем текущую директорию в путь для импортов
sys.path.insert(0, str(Path(__file__).parent))

from team_classifier import classify_teams_by_spawn, save_team_mapping
from gt_find import process_batch


def main():
    """Основной пайплайн обработки."""

    # ========== НАСТРОЙКИ ==========
    # Путь к CSV с данными из демо
    CSV_PATH = r"C:\Users\misas\CS2_NN\spotted_alive_players.csv"

    # Директория со скриншотами
    SCREENSHOTS_DIR = r"C:\Users\misas\CS2_NN\detect\screenshots"

    # Выходная директория для результатов
    OUTPUT_DIR = r"C:\Users\misas\CS2_NN\detect\screenshots_output"

    # Путь для сохранения team_mapping.json
    TEAM_MAPPING_PATH = r"C:\Users\misas\CS2_NN\team_mapping.json"

    # Параметры кластеризации
    INITIAL_TICKS = 5  # Сколько начальных тиков использовать для определения команд

    # Параметры обработки
    CROP_SIZE = 322
    SAVE_ANNOTATIONS = True  # Сохранять ли .txt файлы с YOLO аннотациями
    # ==============================

    print("="*70)
    print("CS2 YOLO Dataset - CT/T Labeling Pipeline")
    print("="*70)

    # Проверяем существование файлов
    csv_path = Path(CSV_PATH)
    screenshots_dir = Path(SCREENSHOTS_DIR)

    if not csv_path.exists():
        print(f"\n❌ Error: CSV file not found at {CSV_PATH}")
        print("Please run parse_demo.py first to generate spotted_alive_players.csv")
        return

    if not screenshots_dir.exists():
        print(f"\n❌ Error: Screenshots directory not found at {SCREENSHOTS_DIR}")
        print("Please run screen_capture.py first to capture screenshots")
        return

    # Шаг 1: Кластеризация команд
    print("\n" + "="*70)
    print("STEP 1: Team Classification (K-means Clustering)")
    print("="*70)

    team_mapping_path = Path(TEAM_MAPPING_PATH)

    if team_mapping_path.exists():
        print(f"\n⚠️  team_mapping.json already exists at {TEAM_MAPPING_PATH}")
        response = input("Do you want to re-run clustering? (y/n): ").strip().lower()

        if response != 'y':
            print("Skipping clustering, using existing team_mapping.json")
        else:
            print("\nRunning team classification...")
            team_mapping = classify_teams_by_spawn(
                csv_path=str(csv_path),
                initial_ticks=INITIAL_TICKS,
                visualize=True
            )
            save_team_mapping(team_mapping, str(team_mapping_path))
    else:
        print("\nRunning team classification...")
        team_mapping = classify_teams_by_spawn(
            csv_path=str(csv_path),
            initial_ticks=INITIAL_TICKS,
            visualize=True
        )
        save_team_mapping(team_mapping, str(team_mapping_path))

    # Шаг 2: Обработка скриншотов
    print("\n" + "="*70)
    print("STEP 2: Bbox Detection & YOLO Annotation Generation")
    print("="*70)

    print(f"\nInput directory: {SCREENSHOTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"CSV path: {CSV_PATH}")
    print(f"Team mapping: {TEAM_MAPPING_PATH}")
    print(f"Save annotations: {SAVE_ANNOTATIONS}")

    response = input("\nProceed with batch processing? (y/n): ").strip().lower()
    if response != 'y':
        print("Aborted by user")
        return

    stats = process_batch(
        input_dir=str(screenshots_dir),
        output_dir=OUTPUT_DIR,
        crop_size=CROP_SIZE,
        save_annotations=SAVE_ANNOTATIONS,
        csv_path=str(csv_path),
        team_mapping_path=str(team_mapping_path)
    )

    # Финальная статистика
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Total images processed: {stats['total']}")
    print(f"Successfully detected: {stats['success']}")
    print(f"No detection: {stats['no_detection']}")
    print(f"Failed: {stats['failed']}")
    print(f"\nResults saved to: {OUTPUT_DIR}")

    if SAVE_ANNOTATIONS:
        print(f"\nYOLO annotations (.txt files) saved with class IDs:")
        print("  0 = CT (Counter-Terrorist)")
        print("  1 = T (Terrorist)")
        print("\nYou can now use these annotations to train YOLO11n!")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
