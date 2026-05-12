"""
Extract kill scenarios from CS2 .dem files into kill_scenarios.json.

Output format consumed by scenario_generator.generate_from_kills():
[
  {
    "attacker_pos":   [x, y, z],   # eye-level position of killer
    "victim_pos":     [x, y, z],   # eye-level position of victim
    "attacker_yaw":   float,       # killer's yaw at kill moment
    "attacker_pitch": float,
    "victim_yaw":     float,
    "victim_pitch":   float,
    "victim_ducking": bool,
    "weapon":         str,
    "headshot":       bool,
    "tick":           int,
    "demo":           str,
    "map":            str
  },
  ...
]

Usage:
    pip install demoparser2
    python extract_kills.py \
        --demos "C:/Program Files (x86)/Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/*.dem" \
        --output kill_scenarios.json \
        --map de_mirage \
        --max-demos 5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

try:
    from demoparser2 import DemoParser
except ImportError:
    print("ERROR: pip install demoparser2", file=sys.stderr)
    sys.exit(1)


# Player tick props we need to look up positions/angles at kill ticks
PLAYER_PROPS = ['X', 'Y', 'Z', 'pitch', 'yaw', 'is_alive', 'team_num',
                'flags', 'ducking', 'name']


def extract_from_demo(demo_path: str, target_map: str | None = None) -> list:
    """Parse one demo and return list of kill events with attacker/victim state."""
    print(f'  parsing {os.path.basename(demo_path)}...')
    parser = DemoParser(demo_path)

    header = parser.parse_header()
    map_name = header.get('map_name', '')
    if target_map and map_name != target_map:
        print(f'    skip: map={map_name} != {target_map}')
        return []

    kills_df = parser.parse_event('player_death')
    if kills_df is None or len(kills_df) == 0:
        return []

    # Filter only meaningful kills:
    #   - real attacker (not suicide/world)
    #   - no wallbang (penetrated == 0) — attacker had direct line of sight to victim
    #   - no thrusmoke — visibility through smoke is unreliable to reproduce
    #   - no flashed attacker (rare but breaks aim)
    kills_df = kills_df[
        kills_df['attacker_steamid'].notna()
        & kills_df['user_steamid'].notna()
        & (kills_df['attacker_steamid'] != kills_df['user_steamid'])
        & (kills_df['weapon'] != 'world')
        & (kills_df['penetrated'] == 0)
        & (kills_df['thrusmoke'] == False)
        & (kills_df['attackerblind'] == False)
    ]

    if len(kills_df) == 0:
        return []

    # Get player state for each unique kill tick
    kill_ticks = sorted(set(int(t) for t in kills_df['tick'].tolist()))
    ticks_df   = parser.parse_ticks(
        ['X', 'Y', 'Z', 'pitch', 'yaw', 'flags'],
        ticks=kill_ticks,
    )
    # Index for fast lookup: (tick, steamid) → row
    ticks_df['steamid'] = ticks_df['steamid'].astype(str)
    ticks_df = ticks_df.set_index(['tick', 'steamid'])

    out = []
    for _, kill in kills_df.iterrows():
        tick     = int(kill['tick'])
        attacker = str(kill['attacker_steamid'])
        victim   = str(kill['user_steamid'])

        try:
            atk = ticks_df.loc[(tick, attacker)]
            vic = ticks_df.loc[(tick, victim)]
        except KeyError:
            continue
        # When .loc returns multiple rows (rare), take first
        if hasattr(atk, 'iloc'): atk = atk.iloc[0] if atk.ndim > 1 else atk
        if hasattr(vic, 'iloc'): vic = vic.iloc[0] if vic.ndim > 1 else vic

        vflags = int(vic.get('flags', 0) or 0)
        victim_ducking = bool(vflags & 2)  # FL_DUCKING = 2

        out.append({
            'demo':           os.path.basename(demo_path),
            'map':            map_name,
            'tick':           tick,
            'weapon':         str(kill.get('weapon', '')),
            'headshot':       bool(kill.get('headshot', False)),
            'penetrated':     int(kill.get('penetrated', 0)),
            'distance':       float(kill.get('distance', 0)),
            'attacker_pos':   [float(atk['X']), float(atk['Y']), float(atk['Z'])],
            'attacker_yaw':   float(atk['yaw']),
            'attacker_pitch': float(atk['pitch']),
            'victim_pos':     [float(vic['X']), float(vic['Y']), float(vic['Z'])],
            'victim_yaw':     float(vic['yaw']),
            'victim_pitch':   float(vic['pitch']),
            'victim_ducking': victim_ducking,
        })

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demos',  required=True,
                    help='Glob pattern for .dem files (use quotes for paths with spaces)')
    ap.add_argument('--output', required=True, help='Output JSON path')
    ap.add_argument('--map',    default=None,
                    help='Filter to specific map (e.g. de_mirage). None = all.')
    ap.add_argument('--max-demos', type=int, default=None)
    args = ap.parse_args()

    demos = sorted(glob.glob(args.demos))
    if args.max_demos:
        demos = demos[:args.max_demos]
    print(f'Found {len(demos)} demos')

    all_kills = []
    for d in demos:
        try:
            kills = extract_from_demo(d, target_map=args.map)
            print(f'    +{len(kills)} kills')
            all_kills.extend(kills)
        except Exception as ex:
            print(f'    ERROR: {ex}')

    with open(args.output, 'w') as f:
        json.dump(all_kills, f, separators=(',', ':'))
    print(f'\nSaved {len(all_kills)} kill scenarios → {args.output}')


if __name__ == '__main__':
    main()
