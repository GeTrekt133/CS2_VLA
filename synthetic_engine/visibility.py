"""Map-geometry visibility check via trimesh BVH raycast.

Uses CS2 map's `world_physics_physics.glb` extracted via Source 2 Viewer CLI.
Pure Python — no compilation needed. ~5ms/ray with trimesh's BVH (vs 1ms C++ vischeck).

Workflow:
  1. Source2Viewer-CLI.exe extracts world_physics_physics.glb from the map's .vpk
  2. trimesh loads it, applies meters→hammer-units scale (×39.37)
  3. Build BVH on first use (~500ms cold)
  4. Per-ray check: cast eye→bone, see if first hit is closer than bone

Usage:
    from visibility import VisibilityChecker
    chk = VisibilityChecker(map_name='de_mirage', maps_dir='./maps_extracted')
    visible = chk.is_visible(eye_pos, bone_pos)   # bool
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

# Suppress numpy "divide by zero" warning from trimesh's barycentric code on
# degenerate triangles in worldnode GLBs — they don't affect ray-hit correctness.
np.seterr(divide='ignore', invalid='ignore')


# CS2 GLB exports come in METERS; CS2 game uses HAMMER UNITS (1 unit ≈ 1 inch ≈ 0.0254m)
# Multiply mesh by this to align with our position data (which is in hammer units).
METERS_TO_HAMMER = 39.3700787


class VisibilityChecker:
    """Lazy-loaded per-map visibility checker via trimesh ray-mesh intersection."""

    def __init__(self, map_name: str = 'de_mirage', maps_dir: str = './maps_extracted'):
        self.map_name = map_name
        self.maps_dir = Path(maps_dir)
        self.mesh = None
        self._fallback_warned = False

        if not HAS_TRIMESH:
            print('[visibility] WARNING: trimesh not installed (pip install trimesh rtree).')
            print('             All visibility checks will return True (no occlusion).')
            return

        # Plain Y/Z swap (Source 2 Viewer relabels axes without sign flip).
        # Verified by test_visibility_axes.py.
        T = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],   # CS2_Y = mesh_Z
            [0, 1, 0, 0],   # CS2_Z = mesh_Y
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Visibility = world brushes (sparse) + worldnode aggregates (dense props).
        # The physics mesh alone misses crates / fences / decorations, so peeking
        # behind cover gives all-visible. n0.glb is the merged worldnode root with
        # the full visual world; combined they give realistic occlusion.
        map_dir = self.maps_dir / map_name / 'maps' / map_name
        sources = []
        for name in ('world_physics_physics.glb', 'worldnodes/n0.glb'):
            p = map_dir / name
            if p.exists(): sources.append(p)

        # Fall back to flat layouts if the canonical map_dir is missing
        if not sources:
            for cand in (self.maps_dir / map_name / 'world_physics_physics.glb',
                         self.maps_dir / f'{map_name}.glb'):
                if cand.exists(): sources.append(cand)

        if not sources:
            print(f'[visibility] WARNING: no GLB found for {map_name} under {self.maps_dir}.')
            print(f'             Extract via tools/Source2Viewer-CLI.exe with -d --gltf_export_format glb')
            print(f'             All visibility checks will return True (no occlusion).')
            return

        meshes = []
        for p in sources:
            print(f'[visibility] loading {p.name}...')
            sub = trimesh.load(str(p), force='mesh')
            sub.apply_scale(METERS_TO_HAMMER)
            sub.apply_transform(T)
            print(f'   -> {len(sub.faces):>7d} triangles, '
                  f'bounds {sub.bounds[0].round()} -> {sub.bounds[1].round()}')
            meshes.append(sub)

        m = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        self.mesh = m
        # Warm up BVH (first call builds the tree)
        _ = m.ray.intersects_any(ray_origins=[[0, 0, 0]], ray_directions=[[1, 0, 0]])
        print(f'[visibility] ready: {len(m.faces)} triangles total, '
              f'bounds {m.bounds[0].round()} -> {m.bounds[1].round()}')

    def is_visible(self, eye: Tuple[float, float, float],
                   target: Tuple[float, float, float]) -> bool:
        """True if line from eye → target is unobstructed by map geometry.

        Algorithm: cast ray from eye in direction of target; if first hit is closer
        than target distance, the line of sight is blocked. Adds small tolerance to
        treat hits within a few units of target as "reached" (not blocked).
        """
        if self.mesh is None:
            if not self._fallback_warned:
                print('[visibility] using fallback (always-visible) — no mesh loaded')
                self._fallback_warned = True
            return True

        eye_arr    = np.asarray(eye,    dtype=np.float64)
        target_arr = np.asarray(target, dtype=np.float64)
        direction  = target_arr - eye_arr
        target_dist = float(np.linalg.norm(direction))
        if target_dist < 1e-3:
            return True
        direction /= target_dist

        try:
            # Get ALL hits — first hit alone is unreliable when the eye is touching
            # a wall (returns the back face of the wall the player is hiding behind).
            # We pick the first hit BEYOND a small epsilon from the eye.
            locs, _ray_idx, _tri_idx = self.mesh.ray.intersects_location(
                ray_origins=[eye_arr.tolist()],
                ray_directions=[direction.tolist()],
                multiple_hits=True,
            )
            if len(locs) == 0:
                return True

            dists = np.linalg.norm(locs - eye_arr, axis=1)
            # Skip hits within EYE_EPS of origin (player hull touching a surface).
            # Keep hits between EYE_EPS and (target_dist - TARGET_EPS) — those block.
            EYE_EPS = 4.0
            TARGET_EPS = 12.0
            blocking = dists[(dists > EYE_EPS) & (dists < target_dist - TARGET_EPS)]
            return len(blocking) == 0
        except Exception as ex:
            print(f'[visibility] check failed: {ex}')
            return True
