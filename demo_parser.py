from demoparser2 import DemoParser
import pandas as pd
from tqdm import tqdm


parser = DemoParser(r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\1-03f67162-abf3-437e-b575-86538acdb399-1-1.dem")
# event_df = parser.parse_event("player_death", player=["X", "Y"])
# ticks_df = parser.parse_ticks(['active_weapon_name', 'active_weapon_ammo', 'total_ammo_left'])
# ticks_df = parser.parse_ticks(['FORWARD', 'LEFT', 'RIGHT', 'BACK'])
# ticks_df = parser.parse_ticks([], players=[76561198386265483])
# ticks_df = parser.parse_ticks(['inventory', 'has_helmet', 'has_defuser', 'armor_value', 'balance']) # для консоли в кс2 - spec_player {user_id + 1}
start_df = parser.parse_event('round_prestart')
# start_df['start'] = [True] * len(start_df)
# freeze_end_df = parser.parse_event('round_freeze_end')
# freeze_end_df['start'] = [False] * len(freeze_end_df)
# buy_ticks = list(pd.concat([start_df, freeze_end_df])['tick'])
# ticks_df = ticks_df[ticks_df['tick'].isin(buy_ticks)]
# ticks_df['start'] = ticks_df['tick'].isin(start_df['tick'])
# ticks_df = parser.parse_ticks(['active_weapon_name', 'active_weapon_ammo'])
# ticks_df = parser.parse_ticks(['X', 'Y', 'is_alive', 'spotted', 'user_id', 'team_num'])
# ticks_df = ticks_df.drop_duplicates(subset=['tick'])
# event_df = event_df[['assister_spotted', 'attacker_spotted', 'user_spotted', 'assister_name', 'attacker_name', 'user_name']]
# bomb_plant = parser.parse_event('bomb_planted', player=["X", "Y"])
# bomb_drop['bomb_dropped'] = [True] * len(bomb_drop)
# bomb_drop = bomb_drop.drop(columns=['entindex'])
# bomd_pickup = parser.parse_event('bomb_pickup', player=["X", "Y"])
# bomd_pickup['bomb_dropped'] = [False] * len(bomd_pickup)
# bomb = pd.concat([bomd_pickup, bomb_drop]).sort_values('tick')
# list_bomb_dropped, bombX, bombY, index = [], [], [], 0
# for i in tqdm(range(len(bomb))):
#     bomb_dropped = bomb['bomb_dropped'].iloc[i]
#     tick = bomb['tick'].iloc[i]
#     steamid = bomb['user_steamid'].iloc[i]
#     if bomb_dropped == False: flag = True
#     else: flag = False
#     while index < len(ticks_df) and ticks_df['tick'].iloc[index] < tick:
#         list_bomb_dropped.append(flag)
#         if flag == True:
#             bombX.append(bomb['user_X'].iloc[i])
#             bombY.append(bomb['user_Y'].iloc[i])
#         else:
#             tick_val = ticks_df['tick'].iloc[index]
#             # print(tick_val)
#             player_tick = ticks_df[(ticks_df['steamid'] == int(steamid)) & (ticks_df['tick'] == tick_val)]
#             # print(player_tick, steamid, type(steamid))
#             bombX.append(player_tick.iloc[0]['X'])
#             bombY.append(player_tick.iloc[0]['Y'])
#         index += 1
# ticks_df['bomb_dropped'] = list_bomb_dropped
# ticks_df['bobmX'] = bombX
# ticks_df['bobmY'] = bombY
start_df.to_csv('start.csv', index=False)
event_names = parser.list_game_events()
print(event_names)
# print(bomb_plant)