"""Quick extraction of ONE round from a demo for replay testing.

Pulls per-tick positions for all players + fire events + death events.
No full demo parser yet — just enough to verify replay engine works on one round.

Usage:
    python extract_one_round.py \
        --demo "C:/.../game/csgo/match.dem" \
        --round 3 \
        --output round.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

try:
    from demoparser2 import DemoParser
except ImportError:
    print("ERROR: pip install demoparser2", file=sys.stderr)
    sys.exit(1)


# demoparser2's `active_weapon_name` returns display names like "AK-47", "Glock-18".
# But CS2 GiveNamedItem expects class names "weapon_ak47", "weapon_glock". Map them here.
WEAPON_DISPLAY_TO_CLASS = {
    # Rifles
    'AK-47':         'weapon_ak47',
    'M4A4':          'weapon_m4a1',
    'M4A1-S':        'weapon_m4a1_silencer',
    'AUG':           'weapon_aug',
    'SG 553':        'weapon_sg556',
    'FAMAS':         'weapon_famas',
    'Galil AR':      'weapon_galilar',
    # Snipers
    'AWP':           'weapon_awp',
    'SSG 08':        'weapon_ssg08',
    'SCAR-20':       'weapon_scar20',
    'G3SG1':         'weapon_g3sg1',
    # Pistols
    'Glock-18':      'weapon_glock',
    'USP-S':         'weapon_usp_silencer',
    'P2000':         'weapon_hkp2000',
    'P250':          'weapon_p250',
    'Five-SeveN':    'weapon_fiveseven',
    'Tec-9':         'weapon_tec9',
    'Desert Eagle':  'weapon_deagle',
    'CZ75-Auto':     'weapon_cz75a',
    'Dual Berettas': 'weapon_elite',
    'R8 Revolver':   'weapon_revolver',
    # SMGs
    'MAC-10':        'weapon_mac10',
    'MP9':           'weapon_mp9',
    'MP7':           'weapon_mp7',
    'MP5-SD':        'weapon_mp5sd',
    'UMP-45':        'weapon_ump45',
    'P90':           'weapon_p90',
    'PP-Bizon':      'weapon_bizon',
    # Shotguns / heavy
    'Nova':          'weapon_nova',
    'XM1014':        'weapon_xm1014',
    'Sawed-Off':     'weapon_sawedoff',
    'MAG-7':         'weapon_mag7',
    'Negev':         'weapon_negev',
    'M249':          'weapon_m249',
    'Zeus x27':      'weapon_taser',
    # Grenades
    'High Explosive Grenade': 'weapon_hegrenade',
    'Flashbang':     'weapon_flashbang',
    'Smoke Grenade': 'weapon_smokegrenade',
    'Decoy Grenade': 'weapon_decoy',
    'Molotov':       'weapon_molotov',
    'Incendiary Grenade': 'weapon_incgrenade',
    # C4
    'C4 Explosive':  'weapon_c4',
    # Knives — all map to weapon_knife (skin doesn't matter for replay)
    'Knife':         'weapon_knife',
    'Bayonet':       'weapon_knife',
    'Karambit':      'weapon_knife',
    'Butterfly Knife':       'weapon_knife',
    'Flip Knife':            'weapon_knife',
    'Gut Knife':             'weapon_knife',
    'M9 Bayonet':            'weapon_knife',
    'Huntsman Knife':        'weapon_knife',
    'Falchion Knife':        'weapon_knife',
    'Bowie Knife':           'weapon_knife',
    'Shadow Daggers':        'weapon_knife',
    'Stiletto Knife':        'weapon_knife',
    'Ursus Knife':           'weapon_knife',
    'Talon Knife':           'weapon_knife',
    'Navaja Knife':          'weapon_knife',
    'Paracord Knife':        'weapon_knife',
    'Survival Knife':        'weapon_knife',
    'Skeleton Knife':        'weapon_knife',
    'Nomad Knife':           'weapon_knife',
    'Classic Knife':         'weapon_knife',
    'Kukri Knife':           'weapon_knife',
}


def map_weapon(raw: str) -> str:
    """Convert demo display name → CS2 weapon class name. Returns '' if unknown."""
    if not raw: return ''
    if raw in WEAPON_DISPLAY_TO_CLASS:
        return WEAPON_DISPLAY_TO_CLASS[raw]
    # Already a class name? (e.g. "weapon_ak47")
    if raw.startswith('weapon_'): return raw
    # Try lowercase fallback (sometimes works)
    lower = raw.lower().replace('-', '').replace(' ', '')
    return f'weapon_{lower}'


def find_round_tick_range(parser: DemoParser, round_idx: int):
    """Return (tick_start, tick_end) for the requested round (0-indexed).

    Tries round_start/round_end events first; falls back to round_freeze_end +
    next round_prestart for FACEIT/SourceTV demos that don't have round_start.
    """
    def _safe(name):
        try:
            df = parser.parse_event(name)
            return df if df is not None and len(df) > 0 else None
        except Exception:
            return None

    starts_evt = _safe('round_start')
    if starts_evt is None: starts_evt = _safe('round_freeze_end')
    if starts_evt is None:
        raise RuntimeError('Demo has no round_start nor round_freeze_end')

    ends_evt = _safe('round_end')
    if ends_evt is None: ends_evt = _safe('round_prestart')
    if ends_evt is None: ends_evt = _safe('round_announce_match_point')

    starts_ticks = sorted(int(t) for t in starts_evt['tick'].tolist())
    if ends_evt is not None and len(ends_evt) > 0:
        ends_ticks = sorted(int(t) for t in ends_evt['tick'].tolist())
    else:
        # Fallback: end of round = start of next round (or last tick of demo)
        ends_ticks = starts_ticks[1:] + [starts_ticks[-1] + 10000]

    pairs = []
    for s in starts_ticks:
        next_ends = [e for e in ends_ticks if e > s]
        if not next_ends: continue
        pairs.append((s, next_ends[0]))

    print(f'  found {len(pairs)} rounds in demo')
    if round_idx >= len(pairs):
        raise IndexError(f'Round {round_idx} out of range (demo has {len(pairs)} rounds)')
    return pairs[round_idx]


def extract_round(demo_path: str, round_idx: int):
    print(f'Parsing {os.path.basename(demo_path)}...')
    parser = DemoParser(demo_path)
    header = parser.parse_header()
    map_name = header.get('map_name', '')

    tick_start, tick_end = find_round_tick_range(parser, round_idx)
    print(f'Round {round_idx}: ticks {tick_start} -> {tick_end} ({tick_end - tick_start} ticks)')

    # Per-tick player state. is_walking: try to extract; some demoparser2 versions don't have it.
    print('Extracting per-tick player state...')
    fields = ['X', 'Y', 'Z', 'yaw', 'pitch', 'flags', 'health', 'team_num', 'active_weapon_name']
    try:
        ticks_df_walk = parser.parse_ticks(['is_walking'], ticks=[tick_start])
        if 'is_walking' in ticks_df_walk.columns:
            fields.append('is_walking')
            print('  is_walking field available')
    except Exception:
        print('  is_walking not available — walk state will be derived from velocity')
    ticks_df = parser.parse_ticks(fields, ticks=list(range(tick_start, tick_end + 1)))
    ticks_df['steamid'] = ticks_df['steamid'].astype(str)
    print(f'  got {len(ticks_df)} player-tick rows')

    # Identify players in this round (those alive at any point)
    players_seen = {}
    for _, row in ticks_df.iterrows():
        sid = row['steamid']
        if sid in ('nan', 'None', ''): continue
        if sid not in players_seen:
            players_seen[sid] = {
                'steamid': sid,
                'name':    str(row.get('name', f'player_{len(players_seen)}')),
                'team':    'ct' if int(row['team_num']) == 3 else 't',
            }
    players = list(players_seen.values())
    print(f'  {len(players)} unique players: '
          f'{sum(1 for p in players if p["team"]=="ct")} CT + '
          f'{sum(1 for p in players if p["team"]=="t")} T')

    # Group rows by tick → compact per-tick player list
    print('Building tick-indexed structure...')
    ticks = []
    for tick, group in ticks_df.groupby('tick'):
        tick = int(tick)
        rows = []
        for _, r in group.iterrows():
            sid = r['steamid']
            if sid in ('nan', 'None', ''): continue
            if int(r.get('health', 0) or 0) <= 0: continue   # dead — skip
            rows.append({
                'sid': sid,
                'pos': [float(r['X']), float(r['Y']), float(r['Z'])],
                'yaw':   float(r['yaw']),
                'pitch': float(r['pitch']),
                'flags': int(r.get('flags', 0) or 0),
                'hp':    int(r.get('health', 100) or 100),
                'wep':   map_weapon(str(r.get('active_weapon_name', '') or '')),
                'walk':  bool(r.get('is_walking', False)) if 'is_walking' in r else False,
            })
        ticks.append({'t': tick, 'p': rows})

    # Fire events (weapon_fire)
    fire_df = parser.parse_event('weapon_fire')
    fire_events = []
    if fire_df is not None and len(fire_df) > 0:
        fire_df = fire_df[
            (fire_df['tick'] >= tick_start) & (fire_df['tick'] <= tick_end)
        ]
        # weapon_fire has user_steamid, weapon, tick. View angles taken from per-tick data.
        for _, r in fire_df.iterrows():
            fire_events.append({
                'tick': int(r['tick']),
                'sid':  str(r['user_steamid']),
                'wep':  str(r.get('weapon', '')),
            })
    print(f'  {len(fire_events)} fire events')

    # Initial inventory per player — union of unique weapons seen in the round
    # (covers pickups too; first 200 ticks would also work but this is more permissive).
    print('Computing per-player inventory...')
    inv_per_player = {}
    for t_entry in ticks:
        for p in t_entry['p']:
            sid = p['sid']
            wep = (p.get('wep') or '').strip()
            if not wep: continue
            inv_per_player.setdefault(sid, set()).add(wep)
    inventory = {sid: sorted(weps) for sid, weps in inv_per_player.items()}
    for sid, weps in inventory.items():
        print(f'  {sid}: {weps}')

    # Death events
    death_df = parser.parse_event('player_death')
    death_events = []
    if death_df is not None and len(death_df) > 0:
        death_df = death_df[
            (death_df['tick'] >= tick_start) & (death_df['tick'] <= tick_end)
        ]
        for _, r in death_df.iterrows():
            death_events.append({
                'tick':     int(r['tick']),
                'victim':   str(r['user_steamid']),
                'attacker': str(r.get('attacker_steamid', '')),
                'weapon':   str(r.get('weapon', '')),
                'headshot': bool(r.get('headshot', False)),
            })
    print(f'  {len(death_events)} death events')

    return {
        'demo':       os.path.basename(demo_path),
        'map':        map_name,
        'round_idx':  round_idx,
        'tick_start': tick_start,
        'tick_end':   tick_end,
        'players':    players,
        'ticks':      ticks,
        'fire':       fire_events,
        'deaths':     death_events,
        'inventory':  inventory,    # sid → [weapon_X, weapon_Y, ...]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo',   required=True)
    ap.add_argument('--round',  type=int, default=2,
                    help='0-indexed round (default 2 — skip warmup + pistol)')
    ap.add_argument('--output', default='round.json')
    args = ap.parse_args()

    data = extract_round(args.demo, args.round)
    with open(args.output, 'w') as f:
        json.dump(data, f, separators=(',', ':'))
    sz_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f'\nSaved {args.output} ({sz_mb:.1f} MB)')
    print(f'  {len(data["ticks"])} ticks, {len(data["fire"])} fires, {len(data["deaths"])} deaths')


if __name__ == '__main__':
    main()
