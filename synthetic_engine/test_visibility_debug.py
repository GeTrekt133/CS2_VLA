"""Deep visibility debug.

For each bone of each alive bot:
  - eye position, bone position, ray length
  - all mesh hits along the ray (distance, side flag)
  - which were filtered (too close to eye, too close to target)
  - final visibility decision

Usage (place yourself behind a wall first):
    python test_visibility_debug.py
"""

from __future__ import annotations

import json
import math

import numpy as np

from bot_pose_client import BotPoseClient
from visibility import VisibilityChecker


SKIP_BONES = {0, 24}
EYE_EPS    = 4.0
TARGET_EPS = 12.0


def main():
    chk = VisibilityChecker(map_name='de_mirage')
    if chk.mesh is None:
        print('Mesh missing'); return
    print(f'\nMesh: {len(chk.mesh.faces)} triangles')
    bmin, bmax = chk.mesh.bounds
    print(f'Bounds: X[{bmin[0]:.0f},{bmax[0]:.0f}]  '
          f'Y[{bmin[1]:.0f},{bmax[1]:.0f}]  Z[{bmin[2]:.0f},{bmax[2]:.0f}]\n')

    with BotPoseClient(timeout=10.0) as c:
        r = c._send({'action': 'get_geometry'})
        geo = json.loads(r.message)
        eye = np.asarray(geo['viewer']['eye'], dtype=np.float64)
        print(f'EYE = ({eye[0]:.1f}, {eye[1]:.1f}, {eye[2]:.1f})')

        # Sanity: cast straight down from eye, see floor distance
        locs, _, _ = chk.mesh.ray.intersects_location(
            ray_origins=[eye.tolist()], ray_directions=[[0, 0, -1]],
            multiple_hits=True,
        )
        if len(locs):
            dists = np.linalg.norm(locs - eye, axis=1)
            dists.sort()
            print(f'  down-ray hits: {[f"{d:.1f}" for d in dists[:5]]}')
        else:
            print('  down-ray: no hit (eye outside mesh? or above sky)')

        # Cast forward (eye direction unknown — try +X)
        for dirname, dvec in [('+X', [1,0,0]), ('-X', [-1,0,0]),
                              ('+Y', [0,1,0]), ('-Y', [0,-1,0])]:
            locs, _, _ = chk.mesh.ray.intersects_location(
                ray_origins=[eye.tolist()], ray_directions=[dvec], multiple_hits=True,
            )
            if len(locs):
                dists = sorted(np.linalg.norm(locs - eye, axis=1))
                near = [f'{d:.0f}' for d in dists[:3]]
                print(f'  {dirname}-ray hits (first 3): {near}')

        # Per-bone analysis
        for bot in geo['bots']:
            if not bot.get('alive') or not bot.get('bones'): continue
            origin = bot.get('origin', [0,0,0])
            d_origin = math.dist(eye, origin)
            print(f'\nBot slot={bot["slot"]:>2} origin=({origin[0]:.0f},{origin[1]:.0f},{origin[2]:.0f}) '
                  f'distance={d_origin:.1f}')

            for idx_str in sorted(bot['bones'].keys(), key=int):
                bone_idx = int(idx_str)
                if bone_idx in SKIP_BONES: continue
                pos = bot['bones'][idx_str]
                if pos is None: continue

                target = np.asarray(pos, dtype=np.float64)
                direction = target - eye
                target_dist = float(np.linalg.norm(direction))
                if target_dist < 1e-3:
                    print(f'  bone[{bone_idx:>2}] coincident'); continue
                direction /= target_dist

                locs, _, _ = chk.mesh.ray.intersects_location(
                    ray_origins=[eye.tolist()],
                    ray_directions=[direction.tolist()],
                    multiple_hits=True,
                )
                if len(locs) == 0:
                    print(f'  bone[{bone_idx:>2}] dist={target_dist:5.0f}  HITS=0          '
                          f' -> VIS [no obstacle anywhere]'); continue

                dists = np.linalg.norm(locs - eye, axis=1)
                dists.sort()
                kept_mask = (dists > EYE_EPS) & (dists < target_dist - TARGET_EPS)
                blocking = dists[kept_mask]
                show = ','.join(f'{d:.0f}' for d in dists[:5])
                tag = 'OCC' if len(blocking) > 0 else 'VIS'
                why = f'first blocking @ {blocking[0]:.0f}' if len(blocking) else 'all hits filtered'
                print(f'  bone[{bone_idx:>2}] dist={target_dist:5.0f}  hits[{len(dists)}]=({show}) '
                      f'-> {tag} [{why}]')


if __name__ == '__main__':
    main()
