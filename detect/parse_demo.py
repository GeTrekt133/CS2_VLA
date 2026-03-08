from demoparser2 import DemoParser
import pandas as pd


def parse_spotted_alive_players(demo_path, output_csv=None):
    """
    Парсит демку CS2 и фильтрует тики где игроки видимы (spotted=True) и живы (is_alive=True)
    
    Args:
        demo_path: путь к файлу демки (.dem)
        output_csv: путь для сохранения результата в CSV (опционально)
    
    Returns:
        DataFrame с колонками: tick, steamid, nick, X, Y, Z, is_alive, spotted
    """
    parser = DemoParser(demo_path)
    
    # Парсим тики с необходимыми параметрами
    ticks_df = parser.parse_ticks([
        'tick', 
        'steamid', 
        'user_id',
        'name',  # ник игрока
        'X', 
        'Y', 
        'Z', 
        'is_alive', 
        'spotted',
        'yaw',
        'pitch'
    ])
    
    # Фильтруем по условиям: игрок жив И видим
    filtered_df = ticks_df[
        (ticks_df['is_alive'] == True) & 
        (ticks_df['spotted'] == True)
    ].copy()
    
    # Сортируем по tick
    filtered_df = filtered_df.sort_values('tick').reset_index(drop=True)
    
    # Сохраняем тики с шагом в 16
    unique_ticks = filtered_df['tick'].unique()
    ticks_with_step = unique_ticks[::16]
    filtered_df = filtered_df[filtered_df['tick'].isin(ticks_with_step)].reset_index(drop=True)
    
    # Сохраняем в CSV если путь указан
    if output_csv:
        filtered_df.to_csv(output_csv, index=False)
        print(f"Результаты сохранены в {output_csv}")
        print(f"Всего найдено записей: {len(filtered_df)}")
    
    return filtered_df


if __name__ == "__main__":
    # Пример использования
    demo_file = r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\1-03f67162-abf3-437e-b575-86538acdb399-1-1.dem"  # Укажи путь к своей демке
    
    # Парсим и сохраняем результат
    spotted_alive_df = parse_spotted_alive_players(
        demo_path=demo_file,
        output_csv='spotted_alive_players.csv'
    )
    
    print("\nПервые 10 строк результата:")
    print(spotted_alive_df.head(10))
    print(f"\nСтруктура данных:")
    print(spotted_alive_df.info())
