"""Replay round + capture screenshots with bbox annotations every N ms.

Builds on replay_round.py infrastructure (replay_init/chunk/start), then while
the replay is running:
  - poll status to know current tick
  - get_geometry → trace_visibility (skips all 'player' entities)
  - project visible bones → body+head bboxes
  - screenshot the CS2 window
  - write COCO-format annotations.json on completion

Usage:
    python replay_capture.py --round round.json --output D:/RoundCapture --interval 0.5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

# Reuse replay_round helpers + CS2Commander
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_collect_v2'))
try:
    from cs2_cmd import CS2Commander
except ImportError:
    CS2Commander = None

from bot_pose_client import BotPoseClient
from replay_round import (
    assign_slots, initial_set_poses, chunk_round_for_plugin,
)
from simple_pipeline import (
    get_window_rect, grab_screenshot, list_bots_robust,
    project_body_bbox, project_head_bbox, occlude_by_closer_bots,
    world_to_screen,
    SKIP_BONES, TOLERANCE, MIN_VISIBLE, HEAD_BONE, NECK_BONE,
    CLASS_IDS,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round',    required=True)
    ap.add_argument('--output',   required=True)
    ap.add_argument('--map',      default=None)
    ap.add_argument('--setup-delay', type=float, default=20.0)
    ap.add_argument('--no-load-map', action='store_true')
    ap.add_argument('--viewer-sid', default=None)
    ap.add_argument('--viewer-mode', choices=['player', 'spectator'], default='player')
    ap.add_argument('--interval', type=float, default=0.5,
                    help='Seconds between frame captures during replay (default 0.5)')
    ap.add_argument('--max-frames', type=int, default=200,
                    help='Hard cap on frames captured (safety)')
    ap.add_argument('--warmup-skip', type=float, default=3.0,
                    help='Extra wait after replay_start before first capture (sec). '
                         'Skips spawn-overlap ticks where all players cluster.')
    args = ap.parse_args()

    print(f'Loading {args.round}...')
    rd = json.load(open(args.round))
    map_name = args.map or rd['map']
    print(f'  map={map_name}, ticks={len(rd["ticks"])}, players={len(rd["players"])}, '
          f'fires={len(rd["fire"])}, deaths={len(rd["deaths"])}')

    os.makedirs(os.path.join(args.output, 'frames'), exist_ok=True)
    coco = {
        'images': [], 'annotations': [],
        'categories': [{'id': v, 'name': k} for k, v in CLASS_IDS.items()],
    }
    img_id = 0; ann_id = 0

    cmd = CS2Commander() if CS2Commander else None

    with BotPoseClient(timeout=60.0) as client:
        # --- Map load ---
        if cmd and not args.no_load_map:
            cmd.connect()
            cmd.send(f'map {map_name}')
            print(f'  loading map (waiting {args.setup_delay}s)...')
            time.sleep(args.setup_delay)
        else:
            print('  skip map load (already loaded)')
            if cmd: cmd.connect()

        # --- Determine viewer ---
        viewer_sid = args.viewer_sid or rd['players'][0]['steamid']
        viewer_demo = next((p for p in rd['players'] if p['steamid'] == viewer_sid), None)
        viewer_team = viewer_demo['team'] if viewer_demo else None
        print(f'  viewer = {viewer_sid} (mode={args.viewer_mode}, team={viewer_team})')

        # --- Slot assignment ---
        if args.viewer_mode == 'player':
            sid_to_slot = assign_slots(client, rd['players'], viewer_sid=viewer_sid)
            r = client._send({'action': 'get_human_slot'})
            if r.ok:
                sid_to_slot[viewer_sid] = json.loads(r.message)['slot']
                print(f'  human takes slot {sid_to_slot[viewer_sid]} for sid {viewer_sid}')
        else:
            sid_to_slot = assign_slots(client, rd['players'])

        if not sid_to_slot:
            print('FATAL: no slots'); return

        initial_set_poses(client, rd['ticks'], sid_to_slot)

        # --- Init replay ---
        inventory = rd.get('inventory', {})
        if args.viewer_mode == 'player':
            human_team = viewer_team
            n_ct_bots = sum(1 for p in rd['players'] if p['team'] == 'ct' and p['steamid'] != viewer_sid)
            n_t_bots  = sum(1 for p in rd['players'] if p['team'] == 't'  and p['steamid'] != viewer_sid)
        else:
            human_team = 'spec'
            n_ct_bots = sum(1 for p in rd['players'] if p['team'] == 'ct')
            n_t_bots  = sum(1 for p in rd['players'] if p['team'] == 't')

        client._send({
            'action': 'replay_init',
            'replay_init': {
                'tick_start': rd['tick_start'],
                'tick_end':   rd['tick_end'],
                'sid_to_slot': sid_to_slot,
                'inventory':  {sid: w for sid, w in inventory.items() if sid in sid_to_slot},
                'human_team': human_team,
                'expect_ct':  n_ct_bots,
                'expect_t':   n_t_bots,
                'lock_human_view': args.viewer_mode == 'player',
            },
        })

        chunks = chunk_round_for_plugin(rd, sid_to_slot)
        print(f'  uploading {len(chunks)} chunks...')
        for i, chunk in enumerate(chunks):
            r = client._send({'action': 'replay_chunk', 'replay_chunk': chunk})
            if not r.ok: print(f'  chunk {i} failed: {r.message}'); return

        # --- Start replay ---
        print('  replay_start...')
        client._send({'action': 'replay_start'})

        # Wait for plugin to actually flip active=True (its timer fires at +1.5s
        # after restart settles). Without this we'd race the old replay's stale state.
        print('  waiting for plugin to enter active replay state...')
        for attempt in range(30):                       # up to 9s
            time.sleep(0.3)
            r_st = client._send({'action': 'replay_status'})
            if not r_st.ok: continue
            st = json.loads(r_st.message)
            if st['active'] and st['current_tick'] < st['max_tick']:
                print(f'  active @ tick {st["current_tick"]}/{st["max_tick"]} (after {(attempt+1)*0.3:.1f}s)')
                break
        else:
            print('  WARNING: replay never became active'); return

        if args.warmup_skip > 0:
            print(f'  warmup wait {args.warmup_skip}s (skipping spawn overlap)...')
            time.sleep(args.warmup_skip)

        if args.viewer_mode == 'spectator' and viewer_sid in sid_to_slot:
            view_slot = sid_to_slot[viewer_sid]
            client._send({'action': 'replay_set_viewer', 'poses': [{'slot': view_slot}]})
            print(f'  spectator FPV → slot {view_slot}')

        # --- Capture loop while replay runs ---
        _, l, t, r_x, b = get_window_rect()
        win_w, win_h = r_x - l, b - t
        viewer_slot = sid_to_slot.get(viewer_sid)

        print(f'\n=== capturing every {args.interval}s, max {args.max_frames} frames ===')
        max_rel = rd['tick_end'] - rd['tick_start']
        frame_idx = 0
        # Initialize view matrix offset (one-time, lost on plugin reload)
        VIEW_MATRIX_OFFSET = 0x232E9C0
        r_vm = client.set_view_matrix_offset(VIEW_MATRIX_OFFSET)
        print(f'  set_view_matrix_offset: {r_vm.ok} {r_vm.message}')

        # Wall-clock anchor + tick anchor for correlating frames with demo ticks.
        # Demo runs at 64 Hz → frame N captured at clock t maps to demo tick t*64.
        capture_t0 = time.perf_counter()
        next_deadline = capture_t0
        prev_tick = None
        prev_clock = None
        while frame_idx < args.max_frames:
            # Wait until next absolute deadline (drift-free vs sleep(interval) which
            # accumulates per-call processing time).
            next_deadline += args.interval
            wait = next_deadline - time.perf_counter()
            if wait > 0: time.sleep(wait)

            # Status check — exit if replay done
            r_st = client._send({'action': 'replay_status'})
            if not r_st.ok:
                print(f'  status failed: {r_st.message}'); break
            st = json.loads(r_st.message)
            if not st['active'] or st['current_tick'] >= max_rel:
                print(f'  replay done @ tick {st["current_tick"]}/{max_rel}')
                break

            # Wall-clock relative to capture start
            clock_now = time.perf_counter() - capture_t0
            cur_tick = st['current_tick']

            # Geometry: viewer eye + all bot bones
            r_geo = client._send({'action': 'get_geometry'})
            if not r_geo.ok:
                if frame_idx % 10 == 0: print(f'  [{frame_idx}] get_geometry failed: {r_geo.message[:80] if r_geo.message else "?"}')
                continue
            geo = json.loads(r_geo.message)
            eye = list(geo['viewer']['eye'])

            r_mat = client.get_view_matrix()
            if not r_mat.ok:
                if frame_idx % 10 == 0: print(f'  [{frame_idx}] get_view_matrix failed: {r_mat.message[:80] if r_mat.message else "?"}')
                continue
            mat = json.loads(r_mat.message).get('matrix')
            if mat is None:
                if frame_idx % 10 == 0: print(f'  [{frame_idx}] view_matrix is None')
                continue

            # Build per-bot bone targets (skip dead + viewer's own pawn)
            bone_keys = []
            targets   = []
            skip_slots = []
            target_slots = set()
            for bot in geo['bots']:
                if not bot.get('alive'): continue
                if bot['slot'] == viewer_slot: continue
                bones = bot.get('bones', {}) or {}
                if not bones: continue
                target_slots.add(bot['slot'])
                for idx_str, pos in bones.items():
                    idx = int(idx_str)
                    if idx in SKIP_BONES or pos is None: continue
                    bone_keys.append((bot['slot'], idx, tuple(pos)))
                    targets.append(list(pos))
                    skip_slots.append(bot['slot'])

            if not targets: continue

            res = client.trace_visibility(eye, targets, tolerance=TOLERANCE, skip_slots=skip_slots)
            if res is None: continue
            vis_arr   = res.get('visible',   [])
            frac_arr  = res.get('fractions', [])
            cont_arr  = res.get('contents',  [])
            name_arr  = res.get('hit_names', [])

            # Build per_slot with screen-projected bones for NMS
            per_slot = {}
            for bot in geo['bots']:
                slot = bot['slot']
                if slot in target_slots:
                    per_slot[slot] = {'world_origin': tuple(bot['origin']), 'bones': []}

            for k_i, (slot, bone_idx, world_pos) in enumerate(bone_keys):
                proj = world_to_screen(world_pos, mat, win_w, win_h)
                if proj is None or proj[2] <= 0: continue
                sx, sy, _ = proj
                v = vis_arr[k_i] if k_i < len(vis_arr) else False
                per_slot[slot]['bones'].append({
                    'idx': bone_idx, 'vis_geom': bool(v),
                    'sx': float(sx), 'sy': float(sy), 'world': world_pos,
                    'frac': frac_arr[k_i] if k_i < len(frac_arr) else -1.0,
                    'cont': cont_arr[k_i] if k_i < len(cont_arr) else 0,
                    'hit_name': name_arr[k_i] if k_i < len(name_arr) else '',
                })

            # DIAG: first frame, log what blocks each bone with frac < 0.5
            if frame_idx == 0:
                for slot, data in per_slot.items():
                    blocked = [b for b in data['bones'] if not b['vis_geom']]
                    if blocked:
                        sample = blocked[:5]
                        print(f'  diag slot {slot} occluded {len(blocked)}/{len(data["bones"])}:')
                        for b in sample:
                            print(f'    bone {b["idx"]} frac={b["frac"]:.3f} '
                                  f'cont=0x{b["cont"]:X} hit={b["hit_name"]!r}')

            # NMS: nearer bot occludes farther bot's bones
            per_slot = occlude_by_closer_bots(per_slot, eye)

            # Map to set-of-visible-bone-indices per slot
            per_bot = {slot: {b['idx'] for b in data['bones'] if b['vis']}
                       for slot, data in per_slot.items()}

            # Capture frame BEFORE saving annotations.
            # clock_sec   = wall time since first capture started
            # tick        = current demo tick (0..max_rel) plugin reported
            # tick_global = absolute tick (tick_start..tick_end) for cross-ref with demo
            img_id += 1
            fname = f'tick_{cur_tick:05d}_{frame_idx:03d}.jpg'
            t_grab = time.perf_counter()
            grab_screenshot(os.path.join(args.output, 'frames', fname))
            grab_ms = (time.perf_counter() - t_grab) * 1000
            coco['images'].append({
                'id': img_id, 'file_name': fname, 'width': win_w, 'height': win_h,
                'tick':         cur_tick,
                'tick_global':  rd['tick_start'] + cur_tick,
                'clock_sec':    round(clock_now, 4),
                'grab_ms':      round(grab_ms, 1),
            })

            # Print delta-tick / delta-clock to verify timing is consistent
            if prev_tick is not None:
                d_tick = cur_tick - prev_tick
                d_clk  = clock_now - prev_clock
                ratio  = d_tick / d_clk if d_clk > 0 else 0
                print(f'    delta: {d_tick:>3} ticks in {d_clk:.3f}s '
                      f'({ratio:.1f} tick/s, expected ~64)')
            prev_tick  = cur_tick
            prev_clock = clock_now

            # Per-bot bbox (body + head)
            n_anns = 0
            for bot in geo['bots']:
                slot = bot['slot']
                if slot not in target_slots: continue
                visible_set = per_bot.get(slot, set())
                if len(visible_set) < MIN_VISIBLE: continue

                bones = bot.get('bones', {}) or {}
                team = bot.get('team', 'ct')
                vis_world = [tuple(bones[str(i)]) for i in visible_set
                              if str(i) in bones and bones[str(i)] is not None]
                body = project_body_bbox(vis_world, mat, win_w, win_h)
                if body is None: continue

                # bones_debug for viz overlay: all bones with screen pos + vis flag + hit details
                bones_debug = [
                    {'idx': b['idx'], 'sx': b['sx'], 'sy': b['sy'],
                     'vis': b['vis'], 'frac': b.get('frac', -1.0),
                     'hit_name': b.get('hit_name', '')}
                    for b in per_slot[slot]['bones']
                ]

                ann_id += 1
                coco['annotations'].append({
                    'id': ann_id, 'image_id': img_id,
                    'category_id': CLASS_IDS[team],
                    'bbox': [body[0], body[1], body[2]-body[0], body[3]-body[1]],
                    'area': (body[2]-body[0]) * (body[3]-body[1]),
                    'iscrowd': 0, 'slot': slot,
                    'bones_debug': bones_debug,
                })
                n_anns += 1

                if HEAD_BONE in visible_set and str(NECK_BONE) in bones:
                    head_pos = bones.get(str(HEAD_BONE)); neck_pos = bones.get(str(NECK_BONE))
                    if head_pos and neck_pos:
                        hb = project_head_bbox(tuple(head_pos), tuple(neck_pos), mat, win_w, win_h)
                        if hb:
                            ann_id += 1
                            coco['annotations'].append({
                                'id': ann_id, 'image_id': img_id,
                                'category_id': CLASS_IDS[f'{team}_head'],
                                'bbox': [hb[0], hb[1], hb[2]-hb[0], hb[3]-hb[1]],
                                'area': (hb[2]-hb[0]) * (hb[3]-hb[1]),
                                'iscrowd': 0, 'slot': slot,
                            })
                            n_anns += 1

            print(f'  frame {frame_idx} @ tick {cur_tick:>4}/{max_rel} '
                  f'(t={clock_now:.2f}s, grab={grab_ms:.0f}ms): '
                  f'{n_anns} anns, {len(target_slots)} alive bots')
            frame_idx += 1

        # --- Save COCO ---
        json.dump(coco, open(os.path.join(args.output, 'annotations.json'), 'w'), indent=2)
        print(f'\nDone: {len(coco["images"])} images, {len(coco["annotations"])} anns -> '
              f'{args.output}/annotations.json')

    if cmd: cmd.close()


if __name__ == '__main__':
    main()
