"""One-time setup for simple_pipeline:
    - load map (de_mirage by default)
    - spawn N bots via bot_quota_mode fill
    - apply gameplay cvars (god, freeze off, infinite ammo)

Run ONCE after starting CS2. Then iterate with simple_pipeline.py.

Usage:
    python simple_setup.py                    # de_mirage, 8 bots
    python simple_setup.py --map de_dust2 --bots 6
    python simple_setup.py --no-load-map      # skip map load (already loaded)
"""

from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_collect_v2'))
from cs2_cmd import CS2Commander


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map',         default='de_mirage')
    ap.add_argument('--bots',        type=int, default=8)
    ap.add_argument('--no-load-map', action='store_true')
    ap.add_argument('--map-wait',    type=float, default=15.0)
    args = ap.parse_args()

    cmd = CS2Commander()
    cmd.connect()
    print(f'[CS2Cmd] Connected.')

    if not args.no_load_map:
        print(f'[1/3] Loading map {args.map}...')
        cmd.send(f'map {args.map}')
        time.sleep(args.map_wait)
    else:
        print('[1/3] Skipping map load.')

    print('[2/3] Applying cvars (gameplay + dataset cleanliness)...')
    cvars_list = [
        # --- core cheat / match control (from plugin's prepare_match) ---
        'sv_cheats 1',
        'mp_autoteambalance 0',
        'mp_limitteams 0',
        'mp_freezetime 0',
        'mp_warmuptime 0',
        'mp_warmup_end',
        'mp_team_intro_time 0',                 # skip team intro cinematic
        'mp_roundtime 60',
        'mp_roundtime_defuse 60',
        'mp_roundtime_hostage 60',
        'mp_round_restart_delay 0',
        'mp_match_end_restart 0',
        'mp_ignore_round_win_conditions 1',     # round never ends on its own
        'mp_buy_anywhere 1',
        'sv_infinite_ammo 1',                   # infinite mag, no reload
        'mp_maxmoney 65535',
        'mp_startmoney 65535',
        'mp_buytime 9999',
        # --- player invincibility (so accidental hits don't kill viewer) ---
        'god',
        # --- bots inert: stand still, do not shoot back ---
        'bot_stop 1',
        'bot_dont_shoot 1',
        # --- dataset cleanliness (decals/blood off, HUD stays on) ---
        'r_csgo_render_decals false',           # no bullet hole / blood decals
        'sv_showimpacts 0',                     # no red boxes from impacts
        'weapon_accuracy_nospread 1',           # bullets fly exactly where aimed
        # --- explicit force HUD/crosshair/hands ON (counteract any previous "hidden" run) ---
        'cl_drawhud 1',                         # HUD: radar, ammo, health, money
        'r_drawviewmodel 1',                    # viewmodel (gun in hand)
        'crosshair 1',                          # crosshair on
        'cl_draw_only_deathnotices 0',          # full HUD overlay
        'cl_team_id_overhead_always 1',         # team-id over heads
        'cl_drawhud_force_radar 1',             # ensure radar
        'cl_drawhud_force_deathnotices 1',
        'safezonex 1', 'safezoney 1',           # full screen HUD
    ]
    # Send one-by-one with explicit sleep — send_batch types too fast and the
    # CS2 console glues cvars together ("r_csgo_render_decals falsesv_showimpacts 0").
    for c in cvars_list:
        cmd.send(c)
        time.sleep(0.15)
    time.sleep(1.0)

    print(f'[3/3] Spawning {args.bots} bots via bot_quota...')
    cmd.send('bot_quota_mode fill'); time.sleep(0.2)
    cmd.send(f'bot_quota {args.bots}'); time.sleep(3.0)

    print()
    print('=== READY ===')
    print(f'  Map:      {args.map}')
    print(f'  Bots:     {args.bots}')
    print(f'  HUD:      ON (radar, ammo, hands, crosshair)')
    print()
    print('Now run:')
    print('  python simple_pipeline.py --snapshots snapshots.json --output D:/Detection --max 50')

    cmd.close()


if __name__ == '__main__':
    main()
