"""Smoke test for plugin trace with known-occlusion scenarios.

Pulls viewer eye, casts traces to KNOWN points:
  1. Straight up to sky (should be unobstructed → fraction=1.0)
  2. Straight down to floor (should be hit immediately → fraction~0)
  3. Forward 5000 units (likely hits a wall → fraction<<1.0)
  4. Forward 50 units (likely empty air → fraction=1.0)

If all 4 give expected results, plugin trace is reliable.

Usage:
    python test_trace_known.py
"""

from __future__ import annotations

import json

from bot_pose_client import BotPoseClient


def main():
    with BotPoseClient(timeout=10.0) as c:
        r = c._send({'action': 'get_geometry'})
        if not r.ok:
            print('get_geometry failed:', r.message); return
        geo = json.loads(r.message)
        eye = list(geo['viewer']['eye'])
        ex, ey, ez = eye
        print(f'Eye: ({ex:.1f}, {ey:.1f}, {ez:.1f})\n')

        cases = [
            ('UP   (eye + 5000 Z, sky)',     [ex, ey, ez + 5000], 'expect ~1.0 (unobstructed)'),
            ('DOWN (eye - 200 Z, floor)',    [ex, ey, ez - 200],  'expect <<1.0 (hits floor)'),
            ('+X 5000 (across map)',         [ex + 5000, ey, ez], 'expect <<1.0 (wall somewhere)'),
            ('-X 5000 (across map)',         [ex - 5000, ey, ez], 'expect <<1.0 (wall somewhere)'),
            ('+Y 5000 (across map)',         [ex, ey + 5000, ez], 'expect <<1.0 (wall somewhere)'),
            ('+X 30 (close, empty air)',     [ex + 30, ey, ez],   'expect ~1.0'),
        ]

        targets = [c[1] for c in cases]
        r = c.trace_visibility(eye, targets, tolerance=0.0)
        if not r:
            print('trace_visibility failed'); return

        for (name, _t, hint), v, f in zip(cases, r['visible'], r['fractions']):
            tag = 'VIS' if v else 'OCC'
            print(f'  {name:35s}  fraction={f:.3f}  {tag:3s}  ({hint})')


if __name__ == '__main__':
    main()
