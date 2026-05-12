"""
Smoke test for trace_visibility_batch plugin action.

Reads viewer eye + bot bone positions, runs raytrace, prints per-bone results.
Compare with old bullet-based visibility check on the same scenario.

Usage:
    1. Run spray_setup.py first (loads map, sets cvars)
    2. Place a bot via prepare/pose, aim at it
    3. python test_raytrace.py
"""

from __future__ import annotations

import json
import time

from bot_pose_client import BotPoseClient


SKIP_BONES = {0, 24}


def main():
    with BotPoseClient(timeout=15.0) as c:
        # Get viewer eye + bot bones
        r = c._send({'action': 'get_geometry'})
        if not r.ok:
            print('get_geometry failed:', r.message)
            return
        geo = json.loads(r.message)
        viewer_eye = geo['viewer']['eye']

        bot = next((b for b in geo['bots'] if b.get('alive') and b.get('bones')), None)
        if bot is None:
            print('No alive bot with bones')
            return

        print(f'Viewer eye: {viewer_eye}')
        print(f'Bot slot={bot["slot"]} origin={bot["origin"]}')

        # Build target list from bones (skip ones excluded from hitbox)
        bones = bot['bones']
        targets = []
        bone_ids = []
        for idx_str, pos in sorted(bones.items(), key=lambda kv: int(kv[0])):
            idx = int(idx_str)
            if idx in SKIP_BONES or pos is None:
                continue
            targets.append(list(pos))
            bone_ids.append(idx)

        print(f'\nTesting {len(targets)} bones via raytrace...')

        # Time the batch call
        t0 = time.perf_counter()
        r = c._send({
            'action': 'trace_visibility_batch',
            'from': list(viewer_eye),
            'targets': targets,
            'tolerance': 30.0,
            # Skip the target bot's own capsule so trace reaches the bone (fraction=1.0
            # if no real obstruction, vs 0.97-0.99 hitting the bot's own hull).
            'skip_slots': [bot['slot']] * len(targets),
        })
        elapsed = time.perf_counter() - t0

        if not r.ok:
            print('FAILED:', r.message)
            return

        result = json.loads(r.message)
        visible = result.get('visible', [])
        fractions = result.get('fractions', [])
        contents = result.get('contents',  [0]*len(visible))
        hit_ents = result.get('hit_ents',  [0]*len(visible))
        hit_names = result.get('hit_names', ['']*len(visible))
        n_vis = result.get('n_visible', 0)
        n_total = result.get('n_total', 0)

        # Decode top contents bits — see PureLiquid TraceMasks.h
        BIT_NAMES = {
             0: 'SOLID',  1: 'HITBOX',  2: 'TRIGGER',  3: 'SKY',
             4: 'PLR_CLIP', 5: 'NPC_CLIP', 6: 'BLOCK_LOS', 8: 'LADDER',
            11: 'NODRAW', 12: 'WINDOW', 13: 'PASS_BUL', 14: 'WORLD_GEOM',
            17: 'TOUCH_ALL', 18: 'PLAYER', 19: 'NPC', 20: 'DEBRIS',
        }
        def decode_contents(c):
            if not c: return 'none'
            names = [n for bit, n in BIT_NAMES.items() if c & (1 << bit)]
            return f'0x{c:X}({"|".join(names) or "?"})'

        print(f'\nResult: {n_vis}/{n_total} bones visible (raytrace took {elapsed*1000:.1f}ms)')
        print(f'Per-call avg: {elapsed*1000/max(1, n_total):.2f}ms')
        print()
        for idx, vis, frac, cont, ent, name in zip(
                bone_ids, visible, fractions, contents, hit_ents, hit_names):
            tag = 'VIS' if vis else 'OCC'
            print(f'  bone[{idx:>2}]  {tag}  frac={frac:.3f}  '
                  f'cont={decode_contents(cont)}  ent=0x{ent & 0xFFFFFFFFFFFF:X}  '
                  f'name={name!r}')


if __name__ == '__main__':
    main()
