import pandas as pd
import sys

def merge_results():
    """
    Объединяет video_results.csv и test_day xlsx файл
    """
    print("Загружаем файлы...")

    # Читаем CSV файл
    try:
        video_results = pd.read_csv('video_results.csv')
        print(f"Загружено {len(video_results)} записей из video_results.csv")
        print(f"Колонки: {list(video_results.columns)}")
    except Exception as e:
        print(f"Ошибка при чтении video_results.csv: {e}")
        return

    # Читаем XLSX файл
    try:
        test_day = pd.read_excel('test_day_2026-02-05 (2).xlsx')
        print(f"Загружено {len(test_day)} записей из test_day_2026-02-05 (2).xlsx")
        print(f"Колонки: {list(test_day.columns)}")
    except Exception as e:
        print(f"Ошибка при чтении xlsx файла: {e}")
        return

    # Проверяем наличие video_id в обоих файлах
    if 'video_id' not in video_results.columns:
        print("ОШИБКА: В video_results.csv нет колонки 'video_id'")
        return

    # В xlsx файле колонка называется 'id', переименуем её в 'video_id'
    if 'id' in test_day.columns:
        test_day = test_day.rename(columns={'id': 'video_id'})
        print("Колонка 'id' переименована в 'video_id'")
    elif 'video_id' not in test_day.columns:
        print("ОШИБКА: В xlsx файле нет колонки 'video_id' или 'id'")
        print(f"Доступные колонки: {list(test_day.columns)}")
        return

    # Создаем словарь для быстрого поиска
    video_dict = {}
    for _, row in video_results.iterrows():
        video_id = row['video_id']
        video_dict[video_id] = {
            'best_similarity': row.get('best_similarity', None),
            'match_0.85': row.get('match_0.85', None),
            'match_0.9': row.get('match_0.9', None)
        }

    # Добавляем новые колонки в test_day
    best_similarity_list = []
    match_085_list = []
    match_09_list = []

    for _, row in test_day.iterrows():
        video_id = row['video_id']

        if video_id in video_dict:
            best_similarity_list.append(video_dict[video_id]['best_similarity'])
            match_085_list.append(video_dict[video_id]['match_0.85'])
            match_09_list.append(video_dict[video_id]['match_0.9'])
        else:
            best_similarity_list.append('нет файла')
            match_085_list.append('нет файла')
            match_09_list.append('нет файла')

    # Добавляем колонки
    test_day['best_similarity'] = best_similarity_list
    test_day['match_0.85'] = match_085_list
    test_day['match_0.9'] = match_09_list

    # Конвертируем True/False в 1/0 для match_0.85 и match_0.9
    test_day['match_0.85'] = test_day['match_0.85'].replace({True: 1, False: 0})
    test_day['match_0.9'] = test_day['match_0.9'].replace({True: 1, False: 0})

    # Сохраняем результат
    output_file = 'merged_results.csv'
    test_day.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nРезультат сохранен в {output_file}")
    print(f"Всего записей: {len(test_day)}")

    # Статистика
    no_file_count = sum(1 for x in best_similarity_list if x == 'нет файла')
    print(f"Записей без файла: {no_file_count}")
    print(f"Записей с данными: {len(test_day) - no_file_count}")

    # Колонки detect_*
    detect_cols = [col for col in test_day.columns if col.startswith('detect_')]

    # Создаем копию для подсчета, где "нет файла" = 0
    df_calc = test_day.copy()
    df_calc['match_0.85'] = df_calc['match_0.85'].replace('нет файла', 0)
    df_calc['match_0.9'] = df_calc['match_0.9'].replace('нет файла', 0)

    # Хотя бы одна 1 в detect_* ИЛИ match = 1 (с "нет файла" = 0)
    at_least_one_085 = ((df_calc[detect_cols] == 1).any(axis=1) | (df_calc['match_0.85'] == 1)).sum()
    at_least_one_09 = ((df_calc[detect_cols] == 1).any(axis=1) | (df_calc['match_0.9'] == 1)).sum()

    total_all = len(test_day)
    total_with_files = len(test_day) - no_file_count

    print("\n" + "="*50)
    print("СТАТИСТИКА: Хотя бы одна 1 в detect_* ИЛИ match = 1")
    print("="*50)
    print(f"От ВСЕХ записей (включая 'нет файла' как 0):")
    print(f"  match_0.85: {at_least_one_085}/{total_all} = {at_least_one_085/total_all*100:.2f}%")
    print(f"  match_0.9:  {at_least_one_09}/{total_all} = {at_least_one_09/total_all*100:.2f}%")
    print(f"\nОт записей С ФАЙЛАМИ (исключая 'нет файла'):")
    print(f"  match_0.85: {at_least_one_085}/{total_with_files} = {at_least_one_085/total_with_files*100:.2f}%")
    print(f"  match_0.9:  {at_least_one_09}/{total_with_files} = {at_least_one_09/total_with_files*100:.2f}%")
    print("="*50)

if __name__ == '__main__':
    merge_results()
