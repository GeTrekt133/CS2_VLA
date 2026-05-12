"""Round replay orchestrator.

Loads JSON from extract_one_round.py and replays it via plugin.

V1 scope (this file):
  - Position replay for all live players, every game tick (64 tps)
  - Death events at exact tick (CommitSuicide → bot disappears)
  - NO fire events yet (plugin TriggerBotAttack is V2 TODO)
  - NO grenades (also V2)

Usage:
    python replay_round.py --round round.json --map de_mirage
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from bot_pose_client import BotPose, BotPoseClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_collect_v2'))
try:
    from cs2_cmd import CS2Commander
except Exception:
    CS2Commander = None


CHUNK_TICKS = 500   # how many ticks per replay_chunk message (avoid TCP message size issues)


def assign_slots(client, players, viewer_sid=None, max_retries: int = 8):
    """Map demo steamids → bot slots. If viewer_sid given, the human takes that demo player's
    slot (one fewer bot is allocated). Returns dict steamid → slot (incl. human slot).
    """
    # Bot counts exclude viewer (human takes that slot)
    n_ct = sum(1 for p in players if p['team'] == 'ct' and p['steamid'] != viewer_sid)
    n_t  = sum(1 for p in players if p['team'] == 't'  and p['steamid'] != viewer_sid)
    print(f'  preparing match: {n_ct} CT + {n_t} T bots'
          + (f' (+ human as viewer of {viewer_sid})' if viewer_sid else ''))
    client.prepare_match(n_ct=n_ct, n_t=n_t)
    time.sleep(3.5)
    client.ensure_alive()
    time.sleep(0.6)

    bots_by_team = {'ct': [], 't': []}
    bot_names = {}   # slot -> name (for kicking extras)
    for attempt in range(max_retries):
        r = client.list_bots()
        bots_by_team = {'ct': [], 't': []}
        bot_names = {}
        # list_bots returns "slot=N name=BOT XYZ team=K" entries separated by ", ".
        # We split on "slot=" instead because names may contain spaces.
        msg = r.message or ''
        # Split entries by "slot=" prefix
        entries = ['slot=' + e for e in msg.split('slot=')[1:]]
        for entry in entries:
            tokens = entry.strip().rstrip(',').strip()
            slot = team_num = None; name = ''
            # Parse "slot=N name=Free-form Name team=K"
            try:
                slot_part, rest = tokens.split(' name=', 1)
                slot = int(slot_part.split('=', 1)[1])
                name_part, team_part = rest.rsplit(' team=', 1)
                name = name_part.strip()
                team_num = int(team_part.split(',', 1)[0].strip())
            except Exception:
                continue
            if slot is not None and team_num is not None:
                team = 'ct' if team_num == 3 else 't'
                bots_by_team[team].append(slot)
                bot_names[slot] = name

        ct_ok = len(bots_by_team['ct']) >= n_ct
        t_ok  = len(bots_by_team['t'])  >= n_t
        if ct_ok and t_ok: break

        print(f'    attempt {attempt+1}/{max_retries}: ct={len(bots_by_team["ct"])}/{n_ct} '
              f't={len(bots_by_team["t"])}/{n_t} — waiting (no re-add)...')
        time.sleep(2.0)

    # Trim excess (auto-balance might over-spawn) — kick by actual bot name
    def _trim(team_name, want):
        team_slots = bots_by_team[team_name]
        if len(team_slots) <= want: return
        extras = team_slots[want:]
        for s in extras:
            name = bot_names.get(s, '')
            if name:
                # bot_kick supports name argument
                client._send({'action': 'exec_cmd', 'cmd': f'bot_kick "{name}"'})
                print(f'  kicking extra {team_name.upper()} bot slot={s} name={name!r}')
        bots_by_team[team_name] = team_slots[:want]
    _trim('ct', n_ct)
    _trim('t', n_t)
    time.sleep(0.5)

    print(f'  using slots: ct={bots_by_team["ct"]}, t={bots_by_team["t"]}')

    sid_to_slot = {}
    ct_slots = list(bots_by_team['ct'])
    t_slots  = list(bots_by_team['t'])
    for p in players:
        if p['steamid'] == viewer_sid:
            continue   # human takes this demo player's place — handled separately
        pool = ct_slots if p['team'] == 'ct' else t_slots
        if not pool:
            print(f'  WARN: no slot left for {p["steamid"]} (team {p["team"]})')
            continue
        sid_to_slot[p['steamid']] = pool.pop(0)
    return sid_to_slot


def initial_set_poses(client, ticks, sid_to_slot):
    """Pre-position bots at first tick before replay starts."""
    if not ticks: return
    first_tick = ticks[0]
    poses = []
    for p in first_tick['p']:
        slot = sid_to_slot.get(p['sid'])
        if slot is None: continue
        poses.append(BotPose(
            slot=slot, pos=p['pos'], yaw=p['yaw'], pitch=p['pitch'],
            ducking=(p['flags'] & 2) != 0, hp=9999, freeze=False,
        ))
    if poses:
        print(f'  set initial poses for {len(poses)} bots at tick {first_tick["t"]}')
        client.set_poses(poses)
        time.sleep(0.8)


def extract_weapon_changes(ticks, sid_to_slot, tick_start):
    """Detect per-player weapon changes across the round → list of (relTick, sid, weapon).
    Includes initial weapon at first appearance for each player.
    """
    last_weapon = {}     # sid → last seen weapon
    changes = []
    for t_entry in ticks:
        rel = t_entry['t'] - tick_start
        for p in t_entry['p']:
            sid = p['sid']
            if sid not in sid_to_slot: continue
            wep = p.get('wep', '') or ''
            if wep == '': continue          # skip unknown
            if last_weapon.get(sid) != wep:
                changes.append((rel, sid, wep))
                last_weapon[sid] = wep
    return changes


def chunk_round_for_plugin(round_data, sid_to_slot):
    """Convert round JSON into compact chunks the plugin can ingest.

    Each chunk has:
      ticks:   list of [relTick, sid, x, y, z, yaw, pitch, flags]
      fires:   list of [relTick, sid]
      deaths:  list of [relTick, victim_sid]
      weapons: list of [relTick, sid, weapon_name]   ← only on weapon CHANGE
    """
    tick_start = round_data['tick_start']

    all_fires    = sorted([(f['tick'] - tick_start, f['sid']) for f in round_data['fire']])
    all_deaths   = sorted([(d['tick'] - tick_start, d['victim']) for d in round_data['deaths']])
    all_weapons  = extract_weapon_changes(round_data['ticks'], sid_to_slot, tick_start)
    print(f'  weapon changes: {len(all_weapons)} events')

    ticks = round_data['ticks']
    if not ticks: return []

    chunks = []
    i = 0
    while i < len(ticks):
        end = min(i + CHUNK_TICKS, len(ticks))
        rel_min = ticks[i]['t'] - tick_start
        rel_max = ticks[end - 1]['t'] - tick_start

        chunk_ticks = []
        for t_entry in ticks[i:end]:
            rel = t_entry['t'] - tick_start
            for p in t_entry['p']:
                if p['sid'] not in sid_to_slot: continue
                chunk_ticks.append([
                    rel, p['sid'],
                    p['pos'][0], p['pos'][1], p['pos'][2],
                    p['yaw'], p['pitch'],
                    p['flags'],
                    1 if p.get('walk', False) else 0,   # walk bit (shift held in demo)
                ])

        chunk_fires   = [[rt, sid]      for rt, sid       in all_fires   if rel_min <= rt <= rel_max]
        chunk_deaths  = [[rt, sid]      for rt, sid       in all_deaths  if rel_min <= rt <= rel_max]
        chunk_weapons = [[rt, sid, wep] for rt, sid, wep  in all_weapons if rel_min <= rt <= rel_max]

        chunks.append({
            'ticks':   chunk_ticks,
            'fires':   chunk_fires,
            'deaths':  chunk_deaths,
            'weapons': chunk_weapons,
        })
        i = end
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', required=True, help='round.json from extract_one_round.py')
    ap.add_argument('--map',   default=None, help='Override map (else taken from JSON)')
    ap.add_argument('--setup-delay', type=float, default=20.0,
                    help='Wait after map load (sec). Set to 0 to skip map command if map already loaded.')
    ap.add_argument('--no-load-map', action='store_true',
                    help='Skip map loading — assume correct map is already loaded in CS2')
    ap.add_argument('--viewer-sid', default=None,
                    help='Demo steamid of player whose POV to use. Default: first player from demo.')
    ap.add_argument('--viewer-mode', choices=['player', 'spectator', 'none'], default='player',
                    help='How to view the round:\n'
                         '  player    = human IS that demo player (takes their slot, full player HUD)\n'
                         '  spectator = human spectates a bot in first-person mode (spec FPV)\n'
                         '  none      = free observer, no camera lock')
    ap.add_argument('--lock-human-view', action='store_true',
                    help='Force-override human player view-angle every tick (rigid demo lock). '
                         'Default OFF: user controls view with mouse, weapon animations play correctly. '
                         'Turn ON for ML data collection where exact demo angle is critical.')
    args = ap.parse_args()

    print(f'Loading {args.round}...')
    with open(args.round) as f:
        rd = json.load(f)
    map_name = args.map or rd['map']
    print(f'  map={map_name}, round_idx={rd["round_idx"]}, '
          f'ticks={len(rd["ticks"])}, players={len(rd["players"])}, '
          f'fires={len(rd["fire"])}, deaths={len(rd["deaths"])}')

    cmd = CS2Commander() if CS2Commander else None

    with BotPoseClient(timeout=60.0) as client:
        # Load map (or skip if already loaded)
        if cmd and not args.no_load_map:
            cmd.connect()
            cmd.send(f'map {map_name}')
            print(f'  loading map {map_name} (waiting {args.setup_delay}s)...')
            time.sleep(args.setup_delay)
        else:
            print(f'  skipping map load (assume {map_name} already loaded)')

        # ---- Determine viewer ----
        viewer_sid = args.viewer_sid
        if viewer_sid is None and rd['players']:
            viewer_sid = rd['players'][0]['steamid']
        viewer_demo_player = next((p for p in rd['players'] if p['steamid'] == viewer_sid), None)
        viewer_team = viewer_demo_player['team'] if viewer_demo_player else None
        print(f'  viewer = {viewer_sid} (mode={args.viewer_mode}, team={viewer_team})')

        # PLAYER mode: human takes viewer's slot. Plugin handles team-move + extras in Phase 2.
        if args.viewer_mode == 'player':
            if viewer_demo_player is None:
                print(f'FATAL: viewer-sid {viewer_sid} not found in demo players'); return
            sid_to_slot = assign_slots(client, rd['players'], viewer_sid=viewer_sid)
            r = client._send({'action': 'get_human_slot'})
            if not r.ok:
                print(f'FATAL: get_human_slot failed: {r.message}'); return
            human_info = json.loads(r.message)
            sid_to_slot[viewer_sid] = human_info['slot']
            print(f'  human ({human_info["name"]}) takes slot {human_info["slot"]} for sid {viewer_sid}')
        # SPECTATOR / NONE mode: viewer is allocated as a bot
        else:
            sid_to_slot = assign_slots(client, rd['players'])

        if not sid_to_slot:
            print('FATAL: no slots assigned'); return
        print(f'  total {len(sid_to_slot)} sid→slot mappings')

        # Position bots at tick_start
        initial_set_poses(client, rd['ticks'], sid_to_slot)

        # Init replay
        print('  replay_init...')
        inventory = rd.get('inventory', {})  # sid → list of weapons
        # Determine human's team + expected bot counts for plugin's Phase 2 corrections
        if args.viewer_mode == 'player':
            human_team = viewer_team       # 'ct' or 't'
            n_ct_bots = sum(1 for p in rd['players'] if p['team'] == 'ct' and p['steamid'] != viewer_sid)
            n_t_bots  = sum(1 for p in rd['players'] if p['team'] == 't'  and p['steamid'] != viewer_sid)
        else:
            human_team = 'spec'
            n_ct_bots = sum(1 for p in rd['players'] if p['team'] == 'ct')
            n_t_bots  = sum(1 for p in rd['players'] if p['team'] == 't')

        r = client._send({
            'action': 'replay_init',
            'replay_init': {
                'tick_start':   rd['tick_start'],
                'tick_end':     rd['tick_end'],
                'sid_to_slot':  sid_to_slot,
                'inventory':    {sid: weps for sid, weps in inventory.items() if sid in sid_to_slot},
                'human_team':   human_team,
                'expect_ct':    n_ct_bots,
                'expect_t':     n_t_bots,
                'lock_human_view': bool(args.lock_human_view),
            },
        })
        print(f'    → {r.message}')
        print(f'    human will be on team {human_team!r}, expect bots: {n_ct_bots} CT + {n_t_bots} T')

        # Stream chunks
        chunks = chunk_round_for_plugin(rd, sid_to_slot)
        print(f'  uploading {len(chunks)} chunks...')
        for i, chunk in enumerate(chunks):
            r = client._send({
                'action': 'replay_chunk',
                'replay_chunk': chunk,
            })
            if not r.ok:
                print(f'    chunk {i} failed: {r.message}'); return
            if i % 5 == 0 or i == len(chunks) - 1:
                print(f'    chunk {i+1}/{len(chunks)}: {r.message}')

        # Plugin handles sequence: mp_restartgame + cvars → 1.5s settle → start replay tick=0.
        # Replay starts during HUD freezetime (20s) so demo's freeze actions sync with HUD.
        print('  replay_start! (plugin: 1.5s settle → tick=0)')
        r = client._send({'action': 'replay_start'})
        print(f'    → {r.message}')
        time.sleep(2.5)

        # ---- Viewer mode-specific lock ----
        if args.viewer_mode == 'spectator' and viewer_sid in sid_to_slot:
            view_slot = sid_to_slot[viewer_sid]
            print(f'  spectator FPV: locking to slot {view_slot} (sid {viewer_sid})...')
            r = client._send({'action': 'replay_set_viewer', 'poses': [{'slot': view_slot}]})
            print(f'    → {r.message}')
        elif args.viewer_mode == 'player':
            print(f'  player mode: human IS slot {sid_to_slot[viewer_sid]} '
                  f'with full player HUD (radar, ammo, money)')
        else:
            print(f'  free observer mode (no lock)')

        # Poll status
        max_rel = rd['tick_end'] - rd['tick_start']
        while True:
            time.sleep(2.0)
            r = client._send({'action': 'replay_status'})
            if not r.ok:
                print(f'  status failed: {r.message}'); break
            st = json.loads(r.message)
            print(f'    tick {st["current_tick"]}/{st["max_tick"]} '
                  f'({st["progress"]*100:.1f}%) dead={len(st["dead_slots"])}')
            if not st['active'] or st['current_tick'] >= max_rel:
                break

        print('Done.')


if __name__ == '__main__':
    main()
