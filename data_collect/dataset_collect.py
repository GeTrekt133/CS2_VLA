from demoparser2 import DemoParser
from tqdm import tqdm
import json
import math
import random as r
import pandas as pd
import os


TICKRATE = 64
ROUND_TIME = 115
BOMB_TIME = 40
KEY_MAPPING = {
    "MOUSE_LEFT": 1 << 0,
    "SPACE": 1 << 1,
    "CTRL": 1 << 2,
    "W": 1 << 3,
    "S": 1 << 4,
    "E": 1 << 5, 
    # "IN_CANCEL": 1 << 6,
    # "IN_TURNLEFT": 1 << 7,
    # "IN_TURNRIGHT": 1 << 8,
    "A": 1 << 9,
    "D": 1 << 10,
    "MOUSE_RIGHT": 1 << 11,
    "R": 1 << 13,
    # "IN_ALT1": 1 << 14,
    # "IN_ALT2": 1 << 15,
    # "IN_SPEED": 1 << 16,
    # "IN_WALK": 1 << 17,
    # "IN_ZOOM": 1 << 18,
    # "IN_WEAPON1": 1 << 19,
    # "IN_WEAPON2": 1 << 20,
    # "IN_BULLRUSH": 1 << 21,
    # "IN_GRENADE1": 1 << 22,
    # "IN_GRENADE2": 1 << 23,
    # "IN_ATTACK3": 1 << 24,
    # "UNKNOWN_25": 1 << 25,
    # "UNKNOWN_26": 1 << 26,
    # "UNKNOWN_27": 1 << 27,
    # "UNKNOWN_28": 1 << 28,
    # "UNKNOWN_29": 1 << 29,
    # "UNKNOWN_30": 1 << 30,
    # "UNKNOWN_31": 1 << 31,
    "TAB": 1 << 33,
    "F": 1 << 35,
}
PISTOLS = [
    'p2000',
    'p250',
    'five-seven',
    'glock-18',
    'tec-9',
    'cz75-auto',
    'dual berettas',
    'desert eagle',
    'r8 revolver'
]
RIFLES = [
    'mp9',
    'mac-10',
    'pp-bizon',
    'mp7',
    'ump-45',
    'p90',
    'mp5-sd',
    'famas',
    'galil ar',
    'm4a4',
    'm4a1-s',
    'ak-47',
    'aug',
    'sg 553',
    'ssg 08',
    'awp',
    'scar-20',
    'g3sg1',
    'nova',
    'xm1014',
    'mag-7',
    'sawed-off',
    'm249',
    'negev',
]

def extract_buttons(buttons: int):
    all_keys_as_str = []
    if pd.isna(buttons):
        # print(buttons)
        return all_keys_as_str
    for (button_name, bit_slot) in KEY_MAPPING.items():
        if (int(buttons) & bit_slot) != 0:    # Check if nth bit is set
            all_keys_as_str.append(button_name)

    return all_keys_as_str


if __name__ == "__main__":
    target_steam = 76561198386265483
    events = [
        'tick', 'inventory', 'has_helmet', 'has_defuser', 'armor_value', 'balance', 'is_alive', 
        'team_num', 'buttons', 'active_weapon_name', 'yaw', 'pitch', 'active_weapon_ammo', 'health', 
        'steamid', 'user_id', 'is_walking', 'round_win_status' #TODO is_walking -> shift
    ]
    frames_dir = r"D:\FramesDataset"
    cs_dir = r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo"
    res_json = {"demos": []}
    for demo_idx, demo_name in enumerate(os.listdir(frames_dir)):
        if demo_idx < 5:
            continue
        demo_idx = 0
        res_json['demos'].append({})
        demo_path = os.path.join(frames_dir, demo_name)
        parser = DemoParser(f"{os.path.join(cs_dir, demo_name)}.dem")
        ticks_df = parser.parse_ticks(events)
        ticks_df_target = ticks_df[ticks_df['steamid'] == target_steam]
        # ticks_df_target['round_win_status'].to_csv('ticks_target.csv', index=False)
        bomb_planted = list(parser.parse_event('bomb_planted')['tick'])
        starts_round = list(parser.parse_event('round_prestart')['tick'])[3:]
        round_freeze_end = list(parser.parse_event('round_freeze_end')['tick'])[2:]
        frames = sorted([int(t[5:-4]) for t in os.listdir(demo_path)])
        buttons, aw_t, rows, teams, scores = [], '', [], [], []
        max_tick = ticks_df['tick'].max()

        for i in range(len(round_freeze_end)):
            freeze_start_tick = int(starts_round[i])
            freeze_end_tick = int(round_freeze_end[i])
            team = ticks_df_target[ticks_df_target['tick'] == freeze_start_tick]['team_num'].item()
            team = 'T' if team == 2 else 'CT'
            teams.append(team)

            freeze_duration = (freeze_end_tick - freeze_start_tick) / TICKRATE
            round_end_tick = starts_round[i + 1] - 1 if i + 1 < len(starts_round) else max_tick

            bomb_tick = next((bt for bt in bomb_planted if freeze_end_tick < bt < round_end_tick), None)
            # print(i, freeze_start_tick, freeze_end_tick, round_end_tick, bomb_tick)

            for tick in range(freeze_start_tick, freeze_end_tick):
                sec_elapsed = (tick - freeze_start_tick) / TICKRATE
                time_left = max(freeze_duration - sec_elapsed, 0)
                rows.append([tick, math.ceil(time_left), True, False])

            for tick in range(freeze_end_tick, round_end_tick):
                if bomb_tick and tick >= bomb_tick:
                    sec_elapsed = (tick - bomb_tick) / TICKRATE
                    time_left = max(BOMB_TIME - sec_elapsed, 0)
                    rows.append([tick, math.ceil(time_left), False, True])
                    if time_left == 0:
                        break
                else:
                    sec_elapsed = (tick - freeze_end_tick) / TICKRATE
                    time_left = max(ROUND_TIME - sec_elapsed, 0)
                    rows.append([tick, math.ceil(time_left), False, False])
                    if time_left == 0:
                        break
        time_left_df = pd.DataFrame(rows, columns=['tick', 'time_left', 'freeze', 'bomb'])
        for i, button in tqdm(enumerate(ticks_df_target['buttons'])):
            new_button = extract_buttons(button)
            if ticks_df_target.iloc[i]['is_walking'] == True:
                new_button.append('SHIFT')
            if ticks_df_target.iloc[i]['active_weapon_name']:
                aw = ticks_df_target.iloc[i]['active_weapon_name'].lower()
            # with open('logs.txt', 'a+') as f:
            #     f.write(f"{i},{aw},{aw_t}\n")
            if aw != aw_t:
                aw_t = aw
                cnt = r.randint(3, 7)
            if cnt != 0:
                if 'knife' in aw or 'bayonet' in aw or 'karambit' in aw or 'daggers' in aw:
                    new_button.append('WEAPON3')
                elif aw in PISTOLS:
                    new_button.append('WEAPON2')
                elif aw in RIFLES:
                    new_button.append('WEAPON1')
                else:
                    match(aw):
                        case 'high explosive grenade': new_button.append('HE')
                        case 'flashbang': new_button.append('FLASH')
                        case 'smoke grenade': new_button.append('SMOKE')
                        case 'decoy grenade': new_button.append('DECOY')
                        case 'molotov': new_button.append('MOLOTOV')
                        case 'incendiary grenade': new_button.append('MOLOTOV')
                        case '': new_button.append('C4')
                cnt -= 1
            result = '[' + ', '.join(f'"{x}"' for x in new_button) + ']'
            buttons.append(result)
        ticks_df_target['buttons'] = buttons
        res_json['demos'][demo_idx]['rounds'] = []
        res_json['demos'][demo_idx]['demo_path'] = demo_path
        round_win, score = 0, [0, 0]
        round_win_status_df = ticks_df_target[['tick', 'round_win_status']].loc[ticks_df_target['round_win_status'].shift() != ticks_df_target['round_win_status']]
        round_win_status_df = round_win_status_df[round_win_status_df['round_win_status'] != 0].reset_index(drop=True)
        my_side = teams[0]
        for i, start in tqdm(enumerate(starts_round)):
            max_frame = frames[-1] - 10
            if i + 1 != len(starts_round):
                max_frame = starts_round[i+1] - 10
            frames_round = []
            for frame in frames:
                # print(frame)
                # print(ticks_df_target[(ticks_df_target['tick'] == frame)]['is_alive'])
                try:
                    if frame in range(start - 10, max_frame) and ticks_df_target[(ticks_df_target['tick'] == frame)]['is_alive'].item() == True:
                        frames_round.append(frame)
                except:
                    pass
            side = teams[i]
            res_json['demos'][demo_idx]['rounds'].append({'round_id': i + 1, 
                                       'start_tick': frames_round[0],
                                       'end_tick': frames_round[-1],
                                       'states': []})
            alive_players_df = ticks_df[['tick', 'is_alive', 'team_num']]
            cnt = 0
            new_score = score.copy()
            if i != 0:
                win_side = int(round_win_status_df['round_win_status'].iloc[i])  # 2 (T win) или 3 (CT win)

                swap_sides = ((i - 1) // 12) % 2 == 1 or (i > 25 and (i - 1) % 3 == 0)
                current_my_side = (
                    'CT' if (my_side == 'T' and swap_sides)
                    else 'T' if (my_side == 'CT' and swap_sides)
                    else my_side
                )

                if current_my_side == 'T':
                    if win_side == 2:
                        score[0] += 1
                    elif win_side == 3:
                        score[1] += 1
                else:
                    if win_side == 3:
                        score[0] += 1
                    elif win_side == 2:
                        score[1] += 1
                new_score = score.copy()
            for tick in frames_round:
                tick_row = ticks_df_target[ticks_df_target['tick'] == tick]
                for j in range(4):
                    tick_tmp = ticks_df_target[ticks_df_target['tick'] == tick + j]
                    idx = 1
                    while tick_tmp.empty:
                        tick_tmp = ticks_df_target[ticks_df_target['tick'] == tick + j - idx]
                        idx += 1
                    time_row = time_left_df[time_left_df['tick'] == tick + j]
                    alive_players = alive_players_df[alive_players_df['tick'] == tick + j]
                    alive_t = alive_players[(alive_players['team_num'] == 2) & (alive_players['is_alive'])].shape[0]
                    alive_ct = alive_players[(alive_players['team_num'] == 3) & (alive_players['is_alive'])].shape[0]
                    # round_win_t = round_win
                    # print(tick + j)
                    # round_win = int(tick_tmp['round_win_status'].item())
                    # if round_win_t != round_win and round_win != 0:
                    #     score[round_win - 2] += 1
                    keys = tick_tmp['buttons'].item()
                    pitch = tick_tmp['pitch'].item()
                    yaw = tick_tmp['yaw'].item()
                    time_left, bomb, freeze = None, None, None
                    if not(time_row.empty):
                        time_left = time_row['time_left'].item()
                        bomb = time_row['bomb'].item()
                        freeze = time_row['freeze'].item()
                    res_json['demos'][demo_idx]['rounds'][i]['states'].append({
                        "tick": tick + j, 
                        "keys": json.loads(keys),
                        "mouse": [yaw, pitch],
                        "detects": [],
                        "hp": tick_tmp['health'].item(),
                        "armor": tick_tmp['armor_value'].item(),
                        "helmet": tick_tmp['has_helmet'].item(),
                        "defuser": tick_tmp['has_defuser'].item(),
                        "weapon": tick_tmp['active_weapon_name'].item(),
                        "ammo": tick_tmp['active_weapon_ammo'].item(),
                        "round_time_left": time_left,
                        "bomb_planted": bomb,
                        "freeze_time": freeze,
                        "score": new_score,
                        "side": side,
                        "ct_alive": alive_ct,
                        "t_alive": alive_t,
                        "weapon_list": tick_tmp['inventory'].item()
                    })
            with open("val_dataset.json", "w", encoding="utf-8") as f:
                json.dump(res_json, f, indent=4, ensure_ascii=False)
        # break
