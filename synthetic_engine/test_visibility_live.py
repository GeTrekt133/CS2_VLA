"""End-to-end check: VisibilityChecker against live plugin geometry.

Pulls viewer eye + bot bones from the running CS2 plugin, runs is_visible
for each bone, and prints VIS/OCC. Also runs three sanity rays so misalignment
or accidental "always-visible" regressions are obvious.

Setup: plugin loaded, a bot placed (e.g. spray_setup.py + a pose).

Usage:
    python test_visibility_live.py
"""

from __future__ import annotations

import json
import math
import time

from bot_pose_client import BotPoseClient
from visibility import VisibilityChecker


SKIP_BONES = {0, 24}


def main():
    chk = VisibilityChecker(map_name='de_mirage')
    if chk.mesh is None:
        print('Mesh not loaded — aborting.')
        return

    with BotPoseClient(timeout=10.0) as c:
        r = c._send({'action': 'get_geometry'})
        if not r.ok:
            print('get_geometry failed:', r.message); return
        geo = json.loads(r.message)
        eye = tuple(geo['viewer']['eye'])
        print(f'Viewer eye (X,Y,Z): {eye}\n')

        # --- Sanity rays ---
        ex, ey, ez = eye
        cases = [
            ('above->below thru floor', (ex, ey, ez + 200), (ex, ey, ez - 200), False),
            ('zero-length',             eye,                eye,                  True),
            ('+5000 X (far in map)',    eye, (ex + 5000, ey, ez),                None),
            ('lateral 100 units',       eye, (ex + 100,  ey, ez),                True),
        ]
        for name, a, b, expect in cases:
            t0 = time.perf_counter()
            v = chk.is_visible(a, b)
            dt = (time.perf_counter() - t0) * 1000
            tag = 'VIS' if v else 'OCC'
            verdict = ''
            if expect is True:  verdict = ' [GOOD]' if v       else ' [BAD: should be VIS]'
            if expect is False: verdict = ' [GOOD]' if not v   else ' [BAD: should be OCC]'
            print(f'  {name:30s} -> {tag} ({dt:.1f}ms){verdict}')

        # --- All bones of all alive bots ---
        print('\n--- per-bone checks ---')
        for bot in geo['bots']:
            if not bot.get('alive') or not bot.get('bones'): continue
            visible = 0; total = 0
            for idx_str, pos in bot['bones'].items():
                bone_idx = int(idx_str)
                if bone_idx in SKIP_BONES or pos is None: continue
                total += 1
                if chk.is_visible(eye, tuple(pos)):
                    visible += 1
            origin = bot.get('origin', '?')
            d = math.dist(eye, origin) if origin != '?' else float('nan')
            print(f'  bot slot={bot["slot"]:>2}  origin={origin}  '
                  f'distance={d:6.1f}  VIS={visible}/{total}')


if __name__ == '__main__':
    main()
