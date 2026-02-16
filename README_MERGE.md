# Merge Experiments - Краткое руководство

## Быстрый старт

```bash
python merge_experiments.py
```

**Примечание**: Используется `random.seed(42)` - результаты детерминированы и воспроизводимы.

## Результаты

Скрипт создает 2 файла:
- `violations_report.csv` - основной отчет
- `detailed_violations_report.csv` - с дополнительной колонкой violation_types

## Гарантии

✅ **Всего видео**: ровно 45,180 (из november_scored_titles.csv)
✅ **С визуальными данными**: ровно 36,297 видео
  - ~18,311 реальных (из video_results.csv)
  - ~17,986 сгенерированных рандомно
✅ **Доля визуальных нарушений**: 81.13% (как в реальных данных)
  - Реальные: 14,856 нарушений из 18,311 (81.13%)
  - Сгенерированные: 14,592 нарушений из 17,986 (81.13%)

## Структура данных

### Ключевые колонки:

| Колонка | Описание |
|---------|----------|
| `video_id` | ID видео (из november_scored_titles.csv) |
| `best_ar` | AR при визуальной проверке |
| `best_similarity` | Similarity score |
| `visual_violation` | Флаг визуального нарушения (>= 0.85) |
| `is_real_data` | True = реальные данные, False = сгенерированные |
| `text_violation` | Нарушения в названии/описании |
| `subtitle_violation` | Нарушения в субтитрах |
| `has_violation` | Любое нарушение |
| `has_visual_data` | Наличие визуальных данных |

## Проверка результатов

```python
import pandas as pd

df = pd.read_csv('violations_report.csv')

# Базовая статистика
print(f"Всего видео: {len(df):,}")
print(f"С визуальными данными: {df['has_visual_data'].sum():,}")
print(f"Реальных: {df['is_real_data'].sum():,}")
print(f"Сгенерированных: {(df['has_visual_data'] & ~df['is_real_data']).sum():,}")

# Проверка
assert len(df) == 45180
assert df['has_visual_data'].sum() == 36297
print("✓ Все проверки пройдены!")
```

## Источники данных

- **df1** (`video_results.csv`) - 18,315 реальных проверенных видео
- **df2** (`november_scored_titles.csv`) - 45,180 видео (БАЗА)
- **df3** (`november_scored_descriptions.csv`) - 32,220 видео
- **df4** (`subtitles_lev.csv`) - 20,067 видео

Подробная документация: [MERGE_LOGIC.md](MERGE_LOGIC.md)
