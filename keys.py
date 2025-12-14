from demoparser2 import DemoParser
from tqdm import tqdm
import pandas as pd
import numpy as np
import random as r


KEY_MAPPING = {
    "IN_ATTACK": 1 << 0,
    "IN_JUMP": 1 << 1,
    "IN_DUCK": 1 << 2,
    "IN_FORWARD": 1 << 3,
    "IN_BACK": 1 << 4,
    "IN_USE": 1 << 5, 
    "IN_CANCEL": 1 << 6,
    "IN_TURNLEFT": 1 << 7,
    "IN_TURNRIGHT": 1 << 8,
    "IN_MOVELEFT": 1 << 9,
    "IN_MOVERIGHT": 1 << 10,
    "IN_ATTACK2": 1 << 11,
    "IN_RELOAD": 1 << 13,
    "IN_ALT1": 1 << 14,
    "IN_ALT2": 1 << 15,
    "IN_SPEED": 1 << 16,
    "IN_WALK": 1 << 17,
    "IN_ZOOM": 1 << 18,
    "IN_WEAPON1": 1 << 19,
    "IN_WEAPON2": 1 << 20,
    "IN_BULLRUSH": 1 << 21,
    "IN_GRENADE1": 1 << 22,
    "IN_GRENADE2": 1 << 23,
    "IN_ATTACK3": 1 << 24,
    "UNKNOWN_25": 1 << 25,
    "UNKNOWN_26": 1 << 26,
    "UNKNOWN_27": 1 << 27,
    "UNKNOWN_28": 1 << 28,
    "UNKNOWN_29": 1 << 29,
    "UNKNOWN_30": 1 << 30,
    "UNKNOWN_31": 1 << 31,
    "IN_SCORE": 1 << 33,
    "IN_INSPECT": 1 << 35,
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

    for (button_name, bit_slot) in KEY_MAPPING.items():
        if (buttons & bit_slot) != 0:    # Check if nth bit is set
            all_keys_as_str.append(button_name)

    return all_keys_as_str

parser = DemoParser("C:\\Users\\misas\\Downloads\\1-469477d6-c6da-4c50-bd56-bd7b54426192-1-1.dem\\1-469477d6-c6da-4c50-bd56-bd7b54426192-1-1.dem")
ticks_df = parser.parse_ticks(['buttons', 'active_weapon_name'])
ticks_df.to_csv('buttons2.csv', index=False)
ticks_df = ticks_df[ticks_df['steamid'] == 76561198386265483]
buttons, aw_t, cnt = [], '', 0
for i, button in tqdm(enumerate(ticks_df['buttons'])):
    new_button = extract_buttons(button)
    if ticks_df.iloc[i]['active_weapon_name']:
        aw = ticks_df.iloc[i]['active_weapon_name'].lower()
    if aw != aw_t:
        aw_t = aw
        cnt = r.randint(3, 7)
    if cnt != 0:
        if 'knife' in aw or 'bayonet' in aw or 'karambit' in aw or 'daggers':
            new_button.append('IN_WEAPON3')
        elif aw in PISTOLS:
            new_button.append('IN_WEAPON2')
        elif aw in RIFLES:
            new_button.append('IN_WEAPON1')
        else:
            match(aw):
                case 'high explosive grenade': new_button.append('IN_HE')
                case 'flashbang': new_button.append('IN_FLASH')
                case 'smoke grenade': new_button.append('IN_SMOKE')
                case 'decoy grenade': new_button.append('IN_DECOY')
                case 'molotov': new_button.append('IN_MOLOTOV')
                case 'incendiary grenade': new_button.append('IN_MOLOTOV')
                case 'c4 explosive': new_button.append('IN_C4')
        cnt -= 1
    result = '[' + ', '.join(f"'{x}'" for x in new_button) + ']'
    buttons.append(result)

res_df = pd.DataFrame({
    'buttons': buttons,
    'tick': ticks_df['tick'],
    'steamid': ticks_df['steamid'],
    'name': ticks_df['name']
})

res_df.to_csv('buttons.csv', index=False)
