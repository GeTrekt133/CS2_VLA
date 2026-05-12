"""
Setup CS2 for spray pattern testing — runs all cvars + gives weapon.

Run ONCE before iterating with spray_test.py. After this:
  - Infinite ammo (no reload ever needed)
  - God mode (you can't die)
  - Round never ends
  - All bots kicked (no distractions)
  - sv_showimpacts 1 — bullets paint RED BOXES where they hit (perfect for tuning)
  - Player gets AK-47 (or specified weapon) in primary slot

Usage:
    python spray_setup.py
    python spray_setup.py --weapon m4a1
    python spray_setup.py --weapon ak47 --no-impacts
    python spray_setup.py --no-bots-kick   (keep current bots)

Then aim at a wall and run:
    python spray_test.py --recoil-scale 1.0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_collect_v2'))
from cs2_cmd import CS2Commander


WEAPON_SLOT = {
    # primary (slot1)
    'ak47': 1, 'm4a1': 1, 'm4a1_silencer': 1, 'awp': 1, 'famas': 1, 'galilar': 1,
    'aug': 1, 'sg556': 1, 'mp9': 1, 'mac10': 1, 'ump45': 1, 'mp7': 1, 'p90': 1,
    'bizon': 1, 'm249': 1, 'negev': 1, 'ssg08': 1, 'scar20': 1, 'g3sg1': 1,
    'nova': 1, 'mag7': 1, 'sawedoff': 1, 'xm1014': 1,
    # secondary (slot2)
    'glock': 2, 'usp_silencer': 2, 'hkp2000': 2, 'p250': 2, 'fiveseven': 2,
    'tec9': 2, 'cz75a': 2, 'deagle': 2, 'revolver': 2, 'elite': 2,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weapon', default='ak47',
                    help='Weapon to give (e.g. ak47, m4a1, awp, deagle)')
    ap.add_argument('--map', default=None,
                    help='Load map first (e.g. de_mirage, de_dust2). Skip if already loaded.')
    ap.add_argument('--no-impacts', action='store_true',
                    help='Disable sv_showimpacts (bullet hit visualization)')
    ap.add_argument('--no-bots-kick', action='store_true',
                    help='Do NOT kick existing bots')
    args = ap.parse_args()

    cmd = CS2Commander()
    if not cmd.connect():
        print('FAILED: CS2 window not found')
        return

    slot = WEAPON_SLOT.get(args.weapon, 1)

    # Phase 0: load map if requested
    if args.map:
        print(f'[0/4] Loading map {args.map}...')
        cmd.send(f'map {args.map}')
        time.sleep(15)  # wait for map load

    # Phase 1: cvars (work even during warmup)
    cvars = [
        'sv_cheats 1',
        'mp_autoteambalance 0',
        'mp_limitteams 0',
        'mp_freezetime 0',
        'mp_warmuptime 0',
        'mp_warmup_end',
        'mp_roundtime 60',
        'mp_roundtime_defuse 60',
        'mp_round_restart_delay 0',
        'mp_match_end_restart 0',
        'mp_ignore_round_win_conditions 1',  # round never ends
        'mp_buy_anywhere 1',
        'sv_infinite_ammo 1',                # infinite mag — no reload ever needed
        'mp_maxmoney 65535',
        'mp_startmoney 65535',
        'mp_buytime 9999',
        # Buffer settings
        'cl_drawhud 1',
        'r_drawviewmodel 1',
    ]
    if not args.no_impacts:
        cvars += [
            'sv_showimpacts 1',           # red boxes where bullets hit
            'sv_showimpacts_time 8',      # show for 8 seconds
        ]
    # Visibility-check / dataset cleanliness:
    cvars += [
        'r_csgo_render_decals false',     # NO bullet hole decals on walls
        'weapon_accuracy_nospread 1',     # bullets fly EXACTLY where aimed (deterministic)
    ]
    print('[1/4] Applying cvars...')
    cmd.send_batch(cvars)
    time.sleep(1.5)

    if not args.no_bots_kick:
        print('[2/4] Kicking bots...')
        cmd.send('bot_quota 0')
        cmd.send('bot_kick')
        time.sleep(1.5)

    print('[3/4] Restarting round...')
    cmd.send('mp_restartgame 1')
    time.sleep(3.0)

    # Re-apply post-restart settings (some cvars get reset)
    cmd.send_batch([
        'god',
        'sv_infinite_ammo 1',                # infinite mag — no reload ever needed
        'noclip 0',
    ])
    time.sleep(0.5)

    print(f'[4/4] Giving weapon: {args.weapon}')
    cmd.send(f'give weapon_{args.weapon}')
    time.sleep(0.3)
    cmd.send(f'slot{slot}')
    time.sleep(0.3)

    print()
    print('=== READY ===')
    print(f'  Weapon:     {args.weapon} (slot {slot})')
    print(f'  Infinite:   ammo + god mode')
    print(f'  Impacts:    {"OFF" if args.no_impacts else "ON (red boxes for 8 sec)"}')
    print(f'  Round:      never ends')
    print()
    print('Now in CS2:')
    print('  1. Aim at a wall')
    print('  2. Run: python spray_test.py --recoil-scale 1.0')
    print('  3. Watch red impact boxes appear on the wall')
    print('  4. Tune --recoil-scale until pattern looks tight')


if __name__ == '__main__':
    main()
