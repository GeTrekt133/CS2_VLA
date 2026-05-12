"""
Extract REAL anti-recoil patterns from pro/skilled players' demos.

For each kill where:
  - weapon = ak47 (or specified)
  - attacker fired >=12 bullets in the lead-up (sustained spray)

Records the per-bullet view angle deltas (player's mouse movements). Aggregates
across many such kills → averaged "skilled player" anti-recoil pattern.

Output JSON compatible with spray_test.py --pattern-file.

Usage:
    pip install demoparser2
    python extract_pro_recoil.py \
        --demos "C:/Program Files (x86)/Steam/steamapps/common/Counter-Strike Global Offensive/game/csgo/*.dem" \
        --weapon ak47 \
        --min-bullets 12 \
        --output pro_recoil_ak47.json \
        --max-demos 10
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import List

try:
    from demoparser2 import DemoParser
except ImportError:
    print("ERROR: pip install demoparser2", file=sys.stderr)
    sys.exit(1)


def normalize_yaw_delta(d: float) -> float:
    while d >  180: d -= 360
    while d < -180: d += 360
    return d


def extract_sprays_from_demo(demo_path: str, weapon: str, min_bullets: int,
                              max_lookback_ticks: int = 256) -> List[List[dict]]:
    """For each AK kill in the demo, extract the killer's pre-kill spray as
    a list of {tick, yaw, pitch}. Returns list of sprays (each spray is a list)."""
    print(f'  parsing {os.path.basename(demo_path)}...')
    parser = DemoParser(demo_path)

    # weapon_fire events with attacker tick state
    fires = parser.parse_event('weapon_fire')
    if fires is None or len(fires) == 0:
        return []

    # Filter to specific weapon
    fires = fires[fires['weapon'] == f'weapon_{weapon}']
    if len(fires) == 0:
        return []

    # Get kills for context: find death events to anchor sprays
    kills = parser.parse_event('player_death')
    if kills is None or len(kills) == 0:
        return []
    kills = kills[
        (kills['weapon'] == weapon)
        & kills['attacker_steamid'].notna()
        & kills['user_steamid'].notna()
        & (kills['attacker_steamid'] != kills['user_steamid'])
        & (kills.get('penetrated', 0) == 0)
    ]
    if len(kills) == 0:
        return []

    # Need yaw/pitch at fire ticks — use parse_ticks on the fire event ticks
    fire_ticks = sorted(set(int(t) for t in fires['tick'].tolist()))
    ticks_df = parser.parse_ticks(['yaw', 'pitch'], ticks=fire_ticks)
    ticks_df['steamid'] = ticks_df['steamid'].astype(str)
    ticks_df = ticks_df.set_index(['tick', 'steamid'])

    sprays = []
    for _, kill in kills.iterrows():
        death_tick = int(kill['tick'])
        attacker = str(kill['attacker_steamid'])

        # Find this attacker's fire events in [death_tick - max_lookback, death_tick]
        atk_fires = fires[
            (fires['tick'] >= death_tick - max_lookback_ticks)
            & (fires['tick'] <= death_tick)
            & (fires['attacker_steamid'].astype(str) == attacker)
        ].sort_values('tick')

        if len(atk_fires) < min_bullets:
            continue

        # Build spray: for each fire tick, look up attacker's yaw/pitch
        spray = []
        for _, fire in atk_fires.iterrows():
            fire_tick = int(fire['tick'])
            try:
                row = ticks_df.loc[(fire_tick, attacker)]
                if hasattr(row, 'iloc') and row.ndim > 1:
                    row = row.iloc[0]
                spray.append({
                    'tick':  fire_tick,
                    'yaw':   float(row['yaw']),
                    'pitch': float(row['pitch']),
                })
            except KeyError:
                continue

        if len(spray) >= min_bullets:
            sprays.append(spray)

    return sprays


def aggregate_pattern(sprays: List[List[dict]], max_bullets: int = 30):
    """Average per-bullet (yaw_delta, pitch_delta) across all sprays."""
    deltas_by_bullet = defaultdict(list)  # bullet_index → list of (dy, dp)

    for spray in sprays:
        for i in range(1, len(spray)):
            dy = normalize_yaw_delta(spray[i]['yaw'] - spray[i-1]['yaw'])
            dp = spray[i]['pitch'] - spray[i-1]['pitch']
            deltas_by_bullet[i].append((dy, dp))

    avg = []
    for bullet_idx in range(1, max_bullets + 1):
        if bullet_idx not in deltas_by_bullet:
            break
        deltas = deltas_by_bullet[bullet_idx]
        avg_y = sum(d[0] for d in deltas) / len(deltas)
        avg_p = sum(d[1] for d in deltas) / len(deltas)
        avg.append({
            'bullet':   bullet_idx,
            'anti_yaw':   avg_y,
            'anti_pitch': avg_p,
            'samples':    len(deltas),
        })
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demos', required=True,
                    help='Glob pattern for .dem files')
    ap.add_argument('--weapon', default='ak47')
    ap.add_argument('--min-bullets', type=int, default=12,
                    help='Minimum bullets fired in spray to count')
    ap.add_argument('--max-bullets', type=int, default=30,
                    help='Output pattern length (clamp)')
    ap.add_argument('--output', default=None)
    ap.add_argument('--max-demos', type=int, default=None)
    args = ap.parse_args()

    out_path = args.output or f'pro_recoil_{args.weapon}.json'

    demos = sorted(glob.glob(args.demos))
    if args.max_demos:
        demos = demos[:args.max_demos]
    print(f'Found {len(demos)} demos')

    all_sprays = []
    for d in demos:
        try:
            sprays = extract_sprays_from_demo(d, args.weapon, args.min_bullets)
            print(f'    +{len(sprays)} qualifying sprays')
            all_sprays.extend(sprays)
        except Exception as ex:
            print(f'    ERROR: {ex}')

    print(f'\nTotal sprays: {len(all_sprays)}')
    if not all_sprays:
        print('No sprays found. Try --min-bullets 8 or different weapon.')
        return

    # Aggregate
    pattern = aggregate_pattern(all_sprays, args.max_bullets)
    print(f'Aggregated pattern: {len(pattern)} bullets')

    # Print first 10
    print(f'\n{"#":>3} | {"avg_yaw_delta":>14} | {"avg_pitch_delta":>16} | samples')
    for p in pattern[:10]:
        print(f'  {p["bullet"]:>2} | {p["anti_yaw"]:>+14.3f} | {p["anti_pitch"]:>+16.3f} | {p["samples"]}')

    # Save in spray_test.py-compatible format
    output = {
        'weapon': args.weapon,
        'source': 'pro_demos',
        'n_kicks':     len(pattern),
        'anti_recoil': pattern,                        # what player DID
        'kicks':       [{                              # negated → for backward-compat
            'bullet':     p['bullet'],
            'kick_yaw':   -p['anti_yaw'],
            'kick_pitch': -p['anti_pitch'],
        } for p in pattern],
        'summary': {
            'n_sprays_aggregated': len(all_sprays),
            'min_bullets_per_spray': args.min_bullets,
            'demos_processed': len(demos),
        },
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved to {out_path}')
    print(f'Test with: python spray_test.py --pattern-file {out_path} --recoil-scale 1.0')


if __name__ == '__main__':
    main()
