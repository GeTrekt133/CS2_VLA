"""Diagnostic for trimesh GLB axis alignment with CS2 world coords.

Loads world_physics_physics.glb and tries every plausible Y/Z transform.
For each candidate, casts a ray STRAIGHT DOWN from a high point at a known CS2
spawn position (mirage T spawn). The transform whose downward ray hits a floor
near the expected height (≈ -167) is the correct one.

Usage:
    python test_visibility_axes.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


METERS_TO_HAMMER = 39.3700787

# Known CS2 position on de_mirage near T spawn area (from user's earlier diagnostic).
# Bot reported eye_z ~ -103 (origin -167 + 64 view height). Floor under bot ~ -167.
TEST_X = -77.5
TEST_Y = -1680.3
TEST_Z = -167.97   # bot origin (feet on floor)

GLB = Path('maps_extracted/de_mirage/maps/de_mirage/world_physics_physics.glb')


CANDIDATES = {
    'A_identity':           np.eye(4),
    'B_neg_Z_to_Y':         np.array([[1, 0,  0, 0],
                                      [0, 0, -1, 0],
                                      [0, 1,  0, 0],
                                      [0, 0,  0, 1]], dtype=np.float64),
    'C_pos_Z_to_neg_Y':     np.array([[1,  0, 0, 0],
                                      [0,  0, 1, 0],
                                      [0, -1, 0, 0],
                                      [0,  0, 0, 1]], dtype=np.float64),
    'D_swap_YZ':            np.array([[1, 0, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 0, 1]], dtype=np.float64),
    'E_neg_swap_YZ':        np.array([[1,  0,  0, 0],
                                      [0,  0, -1, 0],
                                      [0, -1,  0, 0],
                                      [0,  0,  0, 1]], dtype=np.float64),
}


def axis_diag(name: str, M: np.ndarray):
    print(f'\n=== {name} ===')
    m = trimesh.load(str(GLB), force='mesh')
    m.apply_scale(METERS_TO_HAMMER)
    if not np.allclose(M, np.eye(4)):
        m.apply_transform(M)

    bmin, bmax = m.bounds
    print(f'  bounds X: [{bmin[0]:8.0f}, {bmax[0]:8.0f}]  span={bmax[0]-bmin[0]:6.0f}')
    print(f'  bounds Y: [{bmin[1]:8.0f}, {bmax[1]:8.0f}]  span={bmax[1]-bmin[1]:6.0f}')
    print(f'  bounds Z: [{bmin[2]:8.0f}, {bmax[2]:8.0f}]  span={bmax[2]-bmin[2]:6.0f}')

    # Cast ray STRAIGHT DOWN at viewer position
    # If transform correct, viewer (X,Y,Z) maps to mesh coords directly. We start
    # from Z=+500 and shoot down, expecting to hit floor near Z=TEST_Z.
    origin = np.array([[TEST_X, TEST_Y, 500.0]])
    direction = np.array([[0.0, 0.0, -1.0]])
    locs, _, _ = m.ray.intersects_location(
        ray_origins=origin, ray_directions=direction, multiple_hits=False,
    )
    if len(locs) == 0:
        print(f'  DOWN at ({TEST_X:.0f},{TEST_Y:.0f}): NO HIT (transform misaligned or empty there)')
    else:
        hit = locs[0]
        dz_to_floor = hit[2] - TEST_Z
        print(f'  DOWN at ({TEST_X:.0f},{TEST_Y:.0f}): hit Z={hit[2]:7.1f}  '
              f'(expected ~ {TEST_Z:.1f}, diff={dz_to_floor:+.1f})')

    # Also test cross-check: ray from above to below, must hit floor in between
    eye = np.array([[TEST_X, TEST_Y, TEST_Z + 200.0]])
    target = np.array([[TEST_X, TEST_Y, TEST_Z - 200.0]])
    direction = (target - eye)
    direction /= np.linalg.norm(direction)
    locs, _, _ = m.ray.intersects_location(
        ray_origins=eye, ray_directions=direction, multiple_hits=False,
    )
    if len(locs) == 0:
        print(f'  through-floor: NO HIT — this transform is WRONG (no floor here)')
    else:
        hit_z = locs[0][2]
        ok = (TEST_Z - 200.0) <= hit_z <= (TEST_Z + 200.0)
        tag = 'OK' if ok else 'WRONG'
        print(f'  through-floor: hit Z={hit_z:7.1f}  [{tag}]')


def main():
    if not GLB.exists():
        print(f'GLB missing at {GLB.resolve()}')
        return
    print(f'Loading {GLB}')
    print(f'Test position (CS2): ({TEST_X}, {TEST_Y}, {TEST_Z})\n')

    for name, M in CANDIDATES.items():
        try:
            axis_diag(name, M)
        except Exception as ex:
            print(f'  {name}: FAILED — {ex}')

    print(f'\nPick the transform whose down-ray hit Z is close to {TEST_Z:.0f}.')

    # Bonus: with no transform, also try down-ray in CS2(X,Z) plane to check Y-up convention
    print('\n--- Y-up native query (testing if mesh stays Y-up: shoot Y=down at X=TEST_X, Z=-TEST_Y) ---')
    m = trimesh.load(str(GLB), force='mesh')
    m.apply_scale(METERS_TO_HAMMER)
    # In Y-up glTF: glTF X = CS2 X, glTF Y = CS2 Z (height), glTF Z = -CS2 Y
    gx = TEST_X
    gz = -TEST_Y
    origin = np.array([[gx, 500.0, gz]])
    direction = np.array([[0.0, -1.0, 0.0]])
    locs, _, _ = m.ray.intersects_location(
        ray_origins=origin, ray_directions=direction, multiple_hits=False,
    )
    if len(locs) == 0:
        print(f'  Native Y-up query NO HIT')
    else:
        hit_y = locs[0][1]  # this is mesh Y = CS2 Z
        diff = hit_y - TEST_Z
        print(f'  Native Y-up query at glTF (X={gx:.0f}, Z={gz:.0f}): hit Y={hit_y:.1f} '
              f'(expected ~ {TEST_Z:.1f}, diff={diff:+.1f})')


if __name__ == '__main__':
    main()
