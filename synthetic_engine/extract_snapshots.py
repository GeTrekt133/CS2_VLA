"""
Extract MULTI-BOT snapshots from CS2 demos.

For each kill event we capture:
  - Attacker's view (kill scenario)
  - Victim's view (death scenario, same tick + small offset)
  - All ALIVE players at that tick + simple FOV filter from viewer

Output JSON consumed by multibot_pipeline.py:
[
  {
    "demo": str, "map": str, "tick": int, "scenario_type": "kill"|"death",
    "weapon": str, "headshot": bool,
    "viewer": {                      # the player whose perspective we recreate
      "pos":  [x,y,z],  "yaw": float, "pitch": float,
      "team": "ct"|"t", "ducking": bool,
    },
    "kill_target": {                 # only present for "kill" type
      "pos":  [x,y,z],  "yaw": float, "pitch": float,
      "team": "ct"|"t", "ducking": bool, "alive": bool,
    },
    "other_players": [               # all alive players visible from viewer's FOV
      {"pos": [...], "yaw": ..., "pitch": ..., "team": ..., "ducking": ...,
       "is_killer_ally": bool, "in_fov": bool},
      ...
    ]
  },
  ...
]

Usage:
    python extract_snapshots.py \
        --demos "C:/.../game/csgo/*.dem" \
        --output snapshots.json \
        --map de_mirage \
        --max-demos 5
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

try:
    from demoparser2 import DemoParser
except ImportError:
    print("ERROR: pip install demoparser2", file=sys.stderr)
    sys.exit(1)


# Field of view for "visible from attacker" filter (CS2 default ≈ 90° horizontal).
# We use 100° to allow some peripheral players that might be visible just at edge.
FOV_CONE_DEG = 100.0
DEATH_TICK_OFFSET = 0  # 0 = same tick as kill (victim was alive at kill tick)


def angle_between(forward, target_dir):
    """Angle in degrees between two unit-ish 3D vectors."""
    dot = sum(a*b for a, b in zip(forward, target_dir))
    mag = (math.sqrt(sum(a*a for a in forward)) *
           math.sqrt(sum(a*a for a in target_dir)))
    if mag < 1e-6:
        return 180.0
    cos_t = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cos_t))


def yaw_pitch_to_forward(yaw_deg, pitch_deg):
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return (cp * cy, cp * sy, -sp)


def is_in_fov(viewer_pos, viewer_yaw, viewer_pitch, target_pos, fov_deg=FOV_CONE_DEG):
    """True if target is within viewer's FOV cone (NO line-of-sight check)."""
    forward = yaw_pitch_to_forward(viewer_yaw, viewer_pitch)
    direction_to = (target_pos[0] - viewer_pos[0],
                    target_pos[1] - viewer_pos[1],
                    target_pos[2] - viewer_pos[2])
    return angle_between(forward, direction_to) < fov_deg / 2.0


def player_state(row):
    """Build a player state dict from a parse_ticks row."""
    flags = int(row.get('flags', 0) or 0)
    return {
        'pos':     [float(row['X']), float(row['Y']), float(row['Z'])],
        'yaw':     float(row['yaw']),
        'pitch':   float(row['pitch']),
        'team':    'ct' if int(row['team_num']) == 3 else 't',
        'ducking': bool(flags & 2),
        'alive':   int(row.get('health', 0) or 0) > 0,
    }


def build_bomb_state_lookup(parser, kill_ticks):
    """For each kill tick, returns planted_bomb info (pos, yaw) or None.
    Parses bomb_planted/defused/exploded events, finds most recent plant
    that's still alive at the given tick, fetches planter position at plant tick
    (planted bomb sits ~where the planter was standing).
    """
    plants_df  = _safe_event(parser, 'bomb_planted')
    defuses_df = _safe_event(parser, 'bomb_defused')
    explodes_df = _safe_event(parser, 'bomb_exploded')

    if plants_df is None or len(plants_df) == 0:
        return {t: None for t in kill_ticks}

    plants = sorted([(int(r['tick']), str(r['user_steamid']))
                     for _, r in plants_df.iterrows()
                     if r.get('user_steamid') is not None])
    defuses  = sorted([int(r['tick']) for _, r in defuses_df.iterrows()])  if defuses_df is not None else []
    explodes = sorted([int(r['tick']) for _, r in explodes_df.iterrows()]) if explodes_df is not None else []

    # For each kill_tick, find latest plant before it that wasn't defused/exploded
    needed_plant_ticks = set()
    plant_for_tick = {}  # kill_tick -> (plant_tick, planter_id) or None
    for kt in kill_ticks:
        recent = None
        for pt, planter in plants:
            if pt > kt: break
            recent = (pt, planter)
        if recent is None:
            plant_for_tick[kt] = None; continue
        plant_tick, planter_id = recent
        if any(plant_tick < dt <= kt for dt in defuses):
            plant_for_tick[kt] = None; continue
        if any(plant_tick < et <= kt for et in explodes):
            plant_for_tick[kt] = None; continue
        plant_for_tick[kt] = recent
        needed_plant_ticks.add(plant_tick)

    if not needed_plant_ticks:
        return {t: None for t in kill_ticks}

    # Batch fetch planter positions at all relevant plant ticks
    plant_ticks_df = parser.parse_ticks(
        ['X', 'Y', 'Z', 'yaw'],
        ticks=sorted(needed_plant_ticks),
    )
    plant_ticks_df['steamid'] = plant_ticks_df['steamid'].astype(str)
    pos_lookup = {}  # (plant_tick, planter_id) -> (pos, yaw)
    for _, row in plant_ticks_df.iterrows():
        key = (int(row['tick']), str(row['steamid']))
        pos_lookup[key] = (
            [float(row['X']), float(row['Y']), float(row['Z'])],
            float(row['yaw']),
        )

    result = {}
    for kt, plant_info in plant_for_tick.items():
        if plant_info is None:
            result[kt] = None; continue
        pos_yaw = pos_lookup.get(plant_info)
        if pos_yaw is None:
            result[kt] = None; continue
        pos, yaw = pos_yaw
        result[kt] = {'pos': pos, 'yaw': yaw}
    return result


def _safe_event(parser, name):
    try:
        ev = parser.parse_event(name)
        return ev if ev is not None and len(ev) > 0 else None
    except Exception:
        return None


def extract_from_demo(demo_path, target_map=None):
    print(f'  parsing {os.path.basename(demo_path)}...')
    parser = DemoParser(demo_path)
    header = parser.parse_header()
    map_name = header.get('map_name', '')
    if target_map and map_name != target_map:
        print(f'    skip: map={map_name}')
        return []

    kills = parser.parse_event('player_death')
    if kills is None or len(kills) == 0:
        return []
    kills = kills[
        kills['attacker_steamid'].notna()
        & kills['user_steamid'].notna()
        & (kills['attacker_steamid'] != kills['user_steamid'])
        & (kills['weapon'] != 'world')
        & (kills.get('penetrated', 0) == 0)
        & (kills.get('thrusmoke', False) == False)
    ]
    if len(kills) == 0:
        return []

    # Pre-fetch player tick state at all kill ticks
    kill_ticks = sorted(set(int(t) for t in kills['tick'].tolist()))
    ticks_df = parser.parse_ticks(
        ['X', 'Y', 'Z', 'pitch', 'yaw', 'flags', 'health', 'team_num'],
        ticks=kill_ticks,
    )
    ticks_df['steamid'] = ticks_df['steamid'].astype(str)

    # Bomb state per kill tick (planted_bomb pos+yaw or None)
    bomb_state = build_bomb_state_lookup(parser, kill_ticks)

    # Index by tick for fast lookup
    ticks_by_tick = {}
    for tick, group in ticks_df.groupby('tick'):
        ticks_by_tick[int(tick)] = group

    out = []
    for _, kill in kills.iterrows():
        tick = int(kill['tick'])
        attacker_id = str(kill['attacker_steamid'])
        victim_id   = str(kill['user_steamid'])

        if tick not in ticks_by_tick:
            continue
        tick_rows = ticks_by_tick[tick]
        # Build lookup steamid -> row
        rows_by_id = {row['steamid']: row for _, row in tick_rows.iterrows()}

        if attacker_id not in rows_by_id or victim_id not in rows_by_id:
            continue

        atk = player_state(rows_by_id[attacker_id])
        vic = player_state(rows_by_id[victim_id])

        # Build "other players" list (excludes attacker + victim)
        others = []
        for sid, row in rows_by_id.items():
            if sid in (attacker_id, victim_id):
                continue
            ps = player_state(row)
            if not ps['alive']:
                continue
            others.append((sid, ps))

        # ── Kill scenario (attacker's POV) ──────────────────────────────
        kill_others = []
        for sid, ps in others:
            in_fov = is_in_fov(atk['pos'], atk['yaw'], atk['pitch'], ps['pos'])
            if not in_fov:
                continue
            kill_others.append({**ps, 'in_fov': True,
                                'is_killer_ally': ps['team'] == atk['team']})

        out.append({
            'demo':            os.path.basename(demo_path),
            'map':             map_name,
            'tick':            tick,
            'scenario_type':   'kill',
            'weapon':          str(kill.get('weapon', '')),
            'headshot':        bool(kill.get('headshot', False)),
            'viewer':          atk,
            'kill_target':     vic,
            'other_players':   kill_others,
            'planted_bomb':    bomb_state.get(tick),
        })

        # ── Death scenario (victim's POV at kill moment) ───────────────
        # Victim's view, attacker is now their "kill_target" if visible.
        # Other players also filtered by victim's FOV.
        death_others = []
        # Include attacker as one of other_players (visible enemy from victim POV)
        if is_in_fov(vic['pos'], vic['yaw'], vic['pitch'], atk['pos']):
            death_others.append({**atk, 'in_fov': True,
                                 'is_killer_ally': atk['team'] == vic['team']})
        for sid, ps in others:
            if not is_in_fov(vic['pos'], vic['yaw'], vic['pitch'], ps['pos']):
                continue
            death_others.append({**ps, 'in_fov': True,
                                 'is_killer_ally': ps['team'] == vic['team']})

        out.append({
            'demo':            os.path.basename(demo_path),
            'map':             map_name,
            'tick':            tick + DEATH_TICK_OFFSET,
            'scenario_type':   'death',
            'weapon':          str(kill.get('weapon', '')),
            'headshot':        bool(kill.get('headshot', False)),
            'viewer':          vic,
            'kill_target':     None,             # victim doesn't have a kill target
            'other_players':   death_others,
            'planted_bomb':    bomb_state.get(tick),
        })

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demos',  required=True, help='Glob pattern for .dem files')
    ap.add_argument('--output', required=True, help='Output JSON path')
    ap.add_argument('--map',    default=None)
    ap.add_argument('--max-demos', type=int, default=None)
    args = ap.parse_args()

    demos = sorted(glob.glob(args.demos))
    if args.max_demos:
        demos = demos[:args.max_demos]
    print(f'Found {len(demos)} demos')

    all_snapshots = []
    for d in demos:
        try:
            snaps = extract_from_demo(d, target_map=args.map)
            kill_count  = sum(1 for s in snaps if s['scenario_type'] == 'kill')
            death_count = sum(1 for s in snaps if s['scenario_type'] == 'death')
            print(f'    +{kill_count} kill +{death_count} death')
            all_snapshots.extend(snaps)
        except Exception as ex:
            print(f'    ERROR: {ex}')

    with open(args.output, 'w') as f:
        json.dump(all_snapshots, f, separators=(',', ':'))
    print(f'\nSaved {len(all_snapshots)} snapshots → {args.output}')

    # Stats
    avg_others = (sum(len(s['other_players']) for s in all_snapshots) /
                  max(1, len(all_snapshots)))
    print(f'Average other_players per snapshot: {avg_others:.2f}')
    by_n = {}
    for s in all_snapshots:
        n = len(s['other_players'])
        by_n[n] = by_n.get(n, 0) + 1
    print('Distribution by num others in FOV:')
    for n in sorted(by_n):
        print(f'  {n} others: {by_n[n]}')


if __name__ == '__main__':
    main()
