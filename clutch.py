import pandas as pd
from demoparser2 import DemoParser

# Загружаем демку
parser = DemoParser(r"C:\Users\misas\Downloads\1-469477d6-c6da-4c50-bd56-bd7b54426192-1-1.dem\1-469477d6-c6da-4c50-bd56-bd7b54426192-1-1.dem")

# Парсим тики с нужными полями
df = parser.parse_ticks(['is_alive', 'team_num'])

# Фильтруем только живых игроков
alive = df[df['is_alive'] == True]

# Считаем количество живых по командам для каждого тика
alive_counts = (
    alive.groupby(['tick', 'team_num'])
    .size()
    .unstack(fill_value=0)
)

# Переименовываем столбцы — обычно team_num == 2 → T, 3 → CT
alive_counts.columns = [
    'team1_alive' if col == 2 else 'team2_alive' if col == 3 else f'team_{col}_alive'
    for col in alive_counts.columns
]

# Добавляем колонку с форматом "5v4"
if 'team1_alive' not in alive_counts.columns:
    alive_counts['team1_alive'] = 0
if 'team2_alive' not in alive_counts.columns:
    alive_counts['team2_alive'] = 0

alive_counts['situation'] = alive_counts.apply(lambda x: f"{x['team1_alive']}v{x['team2_alive']}", axis=1)

# Сохраняем
alive_counts = alive_counts.drop(columns=['team1_alive', 'team2_alive'])
alive_counts.reset_index(inplace=True)
alive_counts.to_csv("round_situations.csv", index=False)

print(alive_counts.head(15))
