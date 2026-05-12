"""
MultiBotPipeline — reconstruct multi-player CS2 snapshots and produce labeled
detection dataset (4 classes: ct, t, ct_head, t_head).

Per snapshot:
  1. prepare_match with required CT/T bot count
  2. Place all bots at their snapshot positions (with team, ducking, armor)
  3. Teleport viewer to viewer_pos
  4. ONE-TIME visibility check per bot (camera + bots are static during scenario)
  5. Generate trajectory (aim at kill_target if any, else center bot in view)
  6. Slow time, capture frames
  7. Per frame: project visible bones → body bbox + head bbox per bot
  8. Save COCO-format annotations

Usage:
    python multibot_pipeline.py \
        --snapshots snapshots.json \
        --output D:/DetectionDataset \
        --map de_mirage \
        --view-matrix-offset 0x232E9C0 \
        --max-scenarios 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from typing import Optional, Tuple, List

from PIL import ImageGrab

from bot_pose_client import (
    BotPose, BotPoseClient,
    EYE_OFFSET_STANDING, EYE_OFFSET_CROUCHED, feet_z,
)
from test_bbox_projection import find_cs2_window
from test_bbox_via_matrix import world_to_screen_via_matrix
# visibility is now handled by client.trace_visibility() — engine-level CGameTraceManager
# via CS2TraceRay plugin. The trimesh BVH path (visibility.py) is left in the repo for
# offline analysis but no longer used by the pipeline.

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_collect_v2'))
try:
    from cs2_cmd import CS2Commander
except Exception:
    CS2Commander = None


# Bone indices to skip (not part of hitbox)
SKIP_BONES = {0, 24}
# Visibility tolerance (units): hit point within N units of bone counts as visible.
# Covers player capsule radius (~16) + slack so hits on bot's own collision register.
VISIBILITY_TOLERANCE = 30.0
# Min visible bones for bot to be included in dataset
MIN_VISIBLE_BONES = 2

# Class IDs for detection
CLASS_IDS = {
    'ct':      0,
    't':       1,
    'ct_head': 2,
    't_head':  3,
    'bomb':    4,   # planted_c4 entity
}

# Multi-view per scenario — same bot positions, different camera angles.
# View 0 is always the original demo aim; rest are random perturbations (deterministic by idx).
N_VIEWS_PER_SCENARIO = 3
VIEW_YAW_RANGE      = 25.0   # ± degrees from original
VIEW_PITCH_RANGE    = 8.0    # ± degrees from original

# Per-class anchor-centered bbox adjustments (entities with pos_screen).
# bomb: anchor sits at bomb base — bbox extends upward + slight horizontal pad.
ANCHOR_ADJUST = {
    'bomb': {
        'shift_up_frac':       0.25,
        'horizontal_pad_frac': 0.12,
    },
}


# Realistic weapon distribution per team (weighted random sample)
WEAPONS_BY_TEAM = {
    'ct': [
        ('m4a1',          25),  # M4A4 (game internal name)
        ('m4a1_silencer', 20),
        ('awp',           10),
        ('famas',          5),
        ('aug',            3),
        ('mp9',            5),
        ('usp_silencer',  18),
        ('hkp2000',        2),
        ('deagle',         8),
        ('p250',           4),
    ],
    't': [
        ('ak47',          30),
        ('awp',           10),
        ('galilar',        7),
        ('sg556',          3),
        ('mac10',          5),
        ('glock',         18),
        ('deagle',         8),
        ('tec9',           5),
        ('p250',           4),
        ('cz75a',          3),
        ('sawedoff',       2),
        ('nova',           5),
    ],
}
# Reliable weapon for visibility check — Glock has consistent damage and bullets
VISIBILITY_WEAPON = 'glock'


def sample_weapon(rng, team):
    """Random realistic weapon for given team (weighted)."""
    weapons, weights = zip(*WEAPONS_BY_TEAM[team])
    return rng.choices(weapons, weights=weights, k=1)[0]


def distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def world_to_yaw_pitch(eye, target):
    dx = target[0] - eye[0]
    dy = target[1] - eye[1]
    dz = target[2] - eye[2]
    yaw   = math.degrees(math.atan2(dy, dx))
    dist  = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(-dz, dist))
    return yaw, pitch


def pre_break_around_bot(c, viewer_eye, bot_center, lateral_offsets=(-30.0, -15.0, 15.0, 30.0),
                         settle=0.025, fire=0.025, post=0.04):
    """Fire deterministic bullets at points laterally offset from bot center to break
    any destructible props near the bot's silhouette. Misses the bot itself.

    Same fast tempo as check_bone_visibility (assumes host_timescale ~6).
    Called twice per scenario (before visibility check + after restart) to keep world
    state consistent between visibility measurement and screenshot.
    """
    # Lateral direction perpendicular to camera→bot vector and world up
    dx = bot_center[0] - viewer_eye[0]
    dy = bot_center[1] - viewer_eye[1]
    dz = bot_center[2] - viewer_eye[2]
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dist < 1.0:  return
    fwd = (dx/dist, dy/dist, dz/dist)
    # right = cross(fwd, world_up)
    rx = fwd[1] * 1.0 - fwd[2] * 0.0
    ry = fwd[2] * 0.0 - fwd[0] * 1.0
    rz = fwd[0] * 0.0 - fwd[1] * 0.0
    rmag = math.sqrt(rx*rx + ry*ry + rz*rz) or 1.0
    rx, ry, rz = rx/rmag, ry/rmag, rz/rmag

    for offset in lateral_offsets:
        target = (bot_center[0] + rx*offset,
                  bot_center[1] + ry*offset,
                  bot_center[2] + rz*offset)
        yaw, pitch = world_to_yaw_pitch(viewer_eye, target)
        c.set_player_view(yaw, pitch)
        time.sleep(settle)
        c._send({'action': 'start_attack'})
        time.sleep(fire)
        c._send({'action': 'stop_attack'})
        time.sleep(post)


def sample_armor(rng):
    bucket = rng.choices(['none', 'kevlar', 'kevlar_helmet'],
                          weights=[18, 12, 70], k=1)[0]
    if bucket == 'none':           return 0,   False
    if bucket == 'kevlar':         return 100, False
    return 100, True


def check_bone_visibility(c, viewer_eye, bone_pos, settle=0.025, fire=0.025, post=0.04):
    """LEGACY bullet-based visibility check. Kept as fallback.

    Fire 1 bullet at bone, return True if visible (gap < tolerance).
    Sleeps assume host_timescale ~6 (game runs faster → wall-clock waits shrink).
    """
    target_yaw, target_pitch = world_to_yaw_pitch(viewer_eye, bone_pos)
    c.set_player_view(target_yaw, target_pitch)
    time.sleep(settle)

    c._send({'action': 'clear_impacts'})
    c._send({'action': 'start_impact_record'})
    c._send({'action': 'start_attack'})
    time.sleep(fire)
    c._send({'action': 'stop_attack'})
    time.sleep(post)
    c._send({'action': 'stop_impact_record'})

    r = c._send({'action': 'get_impacts'})
    impacts = json.loads(r.message).get('impacts', [])
    if not impacts:
        return False

    best = min(impacts, key=lambda i: distance((i['x'], i['y'], i['z']), viewer_eye))
    cam_to_imp  = distance(viewer_eye, (best['x'], best['y'], best['z']))
    cam_to_bone = distance(viewer_eye, bone_pos)
    return (cam_to_bone - cam_to_imp) < VISIBILITY_TOLERANCE


def project_entity_aabb(pos, rotation, mins, maxs, mat, w, h):
    """Project entity's local AABB (mins/maxs) into screen-space AABB.

    Transforms 8 local corners by entity rotation (yaw only, ignore pitch/roll for ground items)
    + entity origin, projects each via view matrix, returns clamped bbox.
    """
    yaw_rad = math.radians(rotation[1] if len(rotation) >= 2 else 0.0)
    cyaw, syaw = math.cos(yaw_rad), math.sin(yaw_rad)

    pts = []
    for ex in (mins[0], maxs[0]):
        for ey in (mins[1], maxs[1]):
            for ez in (mins[2], maxs[2]):
                # Rotate around Z (yaw)
                wx = pos[0] + (cyaw * ex - syaw * ey)
                wy = pos[1] + (syaw * ex + cyaw * ey)
                wz = pos[2] + ez
                sx, sy_screen, depth = world_to_screen_via_matrix((wx, wy, wz), mat, w, h)
                if depth > 0:
                    pts.append((sx, sy_screen))
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if x2 < 0 or y2 < 0 or x1 > w or y1 > h: return None
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
    if (x2 - x1) < 4 or (y2 - y1) < 4: return None
    return [float(x1), float(y1), float(x2), float(y2)]


def project_aabb_from_bones(bone_positions, mat, w, h):
    """AABB around projected bone positions, expanded to cover full model surface
    (skeleton bones are inner skeleton; clothes/helmet/weapon extend further)."""
    pts = []
    for pos in bone_positions:
        sx, sy, d = world_to_screen_via_matrix(pos, mat, w, h)
        if d > 0:
            pts.append((sx, sy))
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    bw, bh = x2 - x1, y2 - y1

    # Asymmetric expansion:
    # X: +18% each side (clothes/arms/weapon outside spine)
    # Top: +15% (helmet/hair above head bone)
    # Bottom: +3% (feet bones approximate model bottom OK)
    pad_x   = 0.18 * bw
    pad_top = 0.15 * bh
    pad_bot = 0.03 * bh
    x1 -= pad_x; x2 += pad_x
    y1 -= pad_top; y2 += pad_bot

    # Clamp to screen
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if (x2 - x1) < 4 or (y2 - y1) < 4:
        return None
    return [float(x1), float(y1), float(x2), float(y2)]


def project_head_bbox(head_pos, neck_pos, mat, w, h):
    """Asymmetric head AABB: side=d, down=d, up=1.2d (d = neck→head distance)."""
    dx = head_pos[0] - neck_pos[0]
    dy = head_pos[1] - neck_pos[1]
    dz = head_pos[2] - neck_pos[2]
    d  = math.sqrt(dx*dx + dy*dy + dz*dz)
    if d < 0.5:
        return None
    cx, cy, cz = head_pos
    side = d
    pts = []
    for ex in (-side, side):
        for ey in (-side, side):
            for ez in (-d, 1.2 * d):
                sx, sy, dpt = world_to_screen_via_matrix(
                    (cx + ex, cy + ey, cz + ez), mat, w, h)
                if dpt > 0:
                    pts.append((sx, sy))
    if len(pts) != 8:
        return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if x2 < 0 or y2 < 0 or x1 > w or y1 > h:
        return None
    if (x2 - x1) > w or (y2 - y1) > h:
        return None
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    return [float(x1), float(y1), float(x2), float(y2)]


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

class MultiBotPipeline:
    def __init__(self, output_dir, view_matrix_offset, seed=42):
        self.output_dir = output_dir
        self.frames_dir = os.path.join(output_dir, 'frames')
        os.makedirs(self.frames_dir, exist_ok=True)

        self.view_matrix_offset = view_matrix_offset
        self.client             = BotPoseClient(timeout=20.0)
        self.cmd                = CS2Commander() if CS2Commander else None
        self.rng                = random.Random(seed)

        # COCO dataset accumulator
        self.coco_images = []
        self.coco_anns   = []
        self.coco_img_id = 0
        self.coco_ann_id = 0

    def setup(self, map_name, setup_delay=10.0):
        self.client.connect()
        self.client.set_view_matrix_offset(self.view_matrix_offset)
        self.cs2_bbox = find_cs2_window()
        if self.cs2_bbox is None:
            raise RuntimeError("CS2 window not found")
        L, T, R, B = self.cs2_bbox
        self.win_w, self.win_h = R - L, B - T

        if self.cmd:
            self.cmd.connect()
            self.cmd.send(f"map {map_name}")
            time.sleep(15)

        # Initial prepare with max bot slots — we'll reconfigure per scenario
        self.client.prepare_match(n_ct=5, n_t=5)
        time.sleep(setup_delay)

    def _teleport_player(self, x, y, z):
        if self.cmd is None:
            raise RuntimeError("CS2Commander unavailable")
        self.cmd.send_batch([
            "god 1", "noclip 1",
            f"setpos {x:.2f} {y:.2f} {z + 10:.2f}",
        ])
        time.sleep(0.25)
        self.cmd.send("noclip 0")
        time.sleep(0.25)
        self.cmd.send("god 1")

    def _list_bots_by_team(self):
        r = self.client.list_bots()
        out = {'ct': [], 't': []}
        for chunk in (r.message or '').split(','):
            slot = team_num = None
            for tok in chunk.split():
                if tok.startswith('slot='):  slot     = int(tok.split('=', 1)[1])
                if tok.startswith('team='):  team_num = int(tok.split('=', 1)[1])
            if slot is not None and team_num is not None:
                team = 'ct' if team_num == 3 else 't'
                out[team].append(slot)
        return out

    def run_scenario(self, snapshot, idx):
        viewer = snapshot['viewer']
        others = snapshot['other_players']

        # Per-phase timing
        timings = {}
        t_scenario_start = time.perf_counter()
        def lap(name, since):
            timings[name] = time.perf_counter() - since

        # Count needed bots per team (excluding viewer)
        need_ct = sum(1 for o in others if o['team'] == 'ct')
        need_t  = sum(1 for o in others if o['team'] == 't')
        # Add kill_target if present
        kt = snapshot.get('kill_target')
        if kt:
            if kt['team'] == 'ct': need_ct += 1
            else:                  need_t  += 1

        # Need bots opposite to viewer's team to NOT include viewer
        # (viewer's team gets count + 1 bot less because viewer occupies a slot)
        # Actually viewer is the human; bots are separate entities.
        if need_ct == 0 and need_t == 0:
            print(f'  [{idx}] no bots needed, skip')
            return

        # Re-prepare match if counts differ from current.
        # Plugin's Phase 2 (bot_add) fires +1.5s game-time after prepare_match returns,
        # then bots need additional ticks to spawn + initialize SkeletonInstance.
        t = time.perf_counter()
        self.client.prepare_match(n_ct=need_ct, n_t=need_t)
        time.sleep(2.5)  # Phase 2 (1.5s) + bot spawn settle (~1s); force-respawn handles stragglers
        self.client.ensure_alive()
        time.sleep(0.3)
        lap('prepare_match', t)

        # Get available bot slots
        bots = self._list_bots_by_team()
        ct_slots = list(bots['ct'])
        t_slots  = list(bots['t'])

        # Build pose list — pair each "other" with a slot of correct team
        all_targets = list(others)
        if kt:
            all_targets.append({**kt, 'is_kill_target': True})

        poses = []
        slot_to_team = {}
        for tgt in all_targets:
            slot = (ct_slots if tgt['team'] == 'ct' else t_slots).pop(0) if (
                ct_slots if tgt['team'] == 'ct' else t_slots) else None
            if slot is None: continue
            armor, helmet = sample_armor(self.rng)
            poses.append(BotPose(
                slot=slot,
                pos=tgt['pos'],
                yaw=tgt['yaw'],
                pitch=tgt['pitch'],
                ducking=tgt.get('ducking', False),
                armor=armor, helmet=helmet,
                hp=100,
                freeze=True,
            ))
            slot_to_team[slot] = tgt['team']

        if not poses:
            print(f'  [{idx}] failed to assign bot slots')
            return

        t = time.perf_counter()
        self.client.set_poses(poses)
        time.sleep(0.4)  # OnTick re-applies, only need teleport to register

        # Teleport viewer
        self._teleport_player(*viewer['pos'])
        self.client.set_player_view(yaw=viewer['yaw'], pitch=viewer['pitch'])
        time.sleep(0.4)

        # Re-apply poses once more (sometimes first set_poses fires before pawn fully ready)
        self.client.set_poses(poses)
        time.sleep(0.3)
        lap('place_bots+viewer', t)

        # Get geometry — we need viewer_eye + bot bones.
        # Some bots have empty/zero bones because their animation system never
        # ticks after spawn (SkeletonInstance bones stay at local 0,0,0).
        # Force-respawn those bots → animation re-inits → bones populate.
        expected_slots = set(slot_to_team)

        def fetch_geo():
            r = self.client.get_geometry()
            if not r.ok: return None
            return json.loads(r.message)

        def broken_slots(g):
            return [b['slot'] for b in g['bots']
                    if b['slot'] in expected_slots and not b.get('bones')]

        geo = fetch_geo()
        if geo is None:
            print(f'  [{idx}] get_geometry failed'); return

        broken = broken_slots(geo)
        if broken:
            t = time.perf_counter()
            print(f'  [{idx}] broken bots (bones empty): {broken} → force-respawn')
            self.client._send({
                'action': 'respawn_bots',
                'poses': [{'slot': s, 'yaw': 0, 'pitch': 0} for s in broken],
            })
            time.sleep(1.0)  # let respawn + initial animation tick complete
            self.client.set_poses([p for p in poses if p.slot in broken])
            time.sleep(0.5)
            geo = fetch_geo()
            lap('respawn_broken', t)

        # Final retry loop in case respawn didn't fully fix it
        for attempt in range(4):
            ready = sum(1 for b in geo['bots']
                        if b['slot'] in expected_slots and b.get('bones'))
            if ready == len(expected_slots):
                break
            debugs = [(b['slot'], b.get('bone_debug', '?'), b.get('alive'),
                       tuple(b.get('origin', [0,0,0])))
                      for b in geo['bots'] if b['slot'] in expected_slots]
            print(f'  [{idx}] still not ready ({ready}/{len(expected_slots)}), retry {attempt+1}/4')
            for s, dbg, alive, origin in debugs:
                print(f'        slot={s} alive={alive} origin={origin} → {dbg}')
            time.sleep(0.4)
            geo = fetch_geo()
            if geo is None:
                print(f'  [{idx}] get_geometry failed mid-retry'); return
        if geo is None:
            print(f'  [{idx}] get_geometry failed after retries')
            return

        # ---- Verify bots are at their intended positions before visibility check ----
        # Race condition: bones populate quickly but bot may still be sliding to target
        # position from default spawn. Visibility check on stale position → 0 visible bones
        # → bot skipped from annotation. Wait until origin converges.
        intended_pos_lookup = {p.slot: p.pos for p in poses}
        POS_TOLERANCE = 50.0  # units; bot's AbsOrigin should be within this of intended
        for attempt in range(8):
            misplaced = []
            for b in geo['bots']:
                if b['slot'] not in expected_slots: continue
                intended = intended_pos_lookup.get(b['slot'])
                actual = b.get('origin')
                if not intended or not actual: continue
                d = math.sqrt(sum((a - i) ** 2 for a, i in zip(actual, intended)))
                if d > POS_TOLERANCE:
                    misplaced.append((b['slot'], d))
            if not misplaced:
                break
            if attempt == 0:
                print(f'  [{idx}] bots not at intended pos: {[(s, f"{d:.0f}u") for s, d in misplaced]} → re-set + wait')
                self.client.set_poses(poses)  # force re-apply
            time.sleep(0.4)
            geo = fetch_geo()
            if geo is None:
                print(f'  [{idx}] get_geometry failed during pos verify'); return

        viewer_eye = tuple(geo['viewer']['eye'])

        # ---- Visibility check per bot (ONCE per scenario — bots & camera static) ----
        # Give player a reliable weapon (Glock) for consistent visibility test bullets.
        # Without this, player might be holding bomb/knife → no bullets fire.
        if self.cmd:
            self.cmd.send(f'give weapon_{VISIBILITY_WEAPON}')
            time.sleep(0.15)
            self.cmd.send('slot2')  # secondary slot for pistol
            time.sleep(0.15)

        # Make all bots immortal during visibility test
        self.client._send({
            'action': 'set_bot_health',
            'poses': [{'slot': s, 'hp': 9999, 'yaw': 0, 'pitch': 0} for s in slot_to_team],
        })
        self.client._send({'action': 'exec_cmd', 'cmd': 'sv_showimpacts 0'})
        time.sleep(0.2)

        # vischeck-based visibility — pure Python BVH, no game time speedup needed
        t_vis = time.perf_counter()

        # Collect every (slot, bone_idx, pos) triple, then issue ONE engine-level
        # batch trace via the plugin (CS2TraceRay → CGameTraceManager). Same accuracy
        # as in-game bullet trace, no mesh approximation, no side-effects.
        bot_visibility = {}  # slot → set of visible bone indices
        bot_fractions = {}   # slot → {bone_idx: fraction} for debug overlay
        bot_centers = {}     # slot → 3-tuple chest position (for post-restart pre-break)
        bone_keys: list[tuple[int, int]] = []      # parallel to targets[]
        targets:   list[tuple[float, float, float]] = []
        per_bot_total = {}                         # slot → tested bone count

        for bot_data in geo['bots']:
            slot = bot_data['slot']
            if slot not in slot_to_team:  continue
            bones = bot_data.get('bones', {})

            # Bot center = chest bone (5) if available else mean of all bones
            center = None
            if '5' in bones and bones['5']:
                center = tuple(bones['5'])
            else:
                bvals = [b for b in bones.values() if b]
                if bvals:
                    n = len(bvals)
                    center = (sum(b[0] for b in bvals)/n,
                              sum(b[1] for b in bvals)/n,
                              sum(b[2] for b in bvals)/n)
            if center:
                bot_centers[slot] = center

            n_tested = 0
            for idx_str, pos in bones.items():
                bone_idx = int(idx_str)
                if bone_idx in SKIP_BONES or pos is None: continue
                bone_keys.append((slot, bone_idx))
                targets.append(tuple(pos))
                n_tested += 1
            per_bot_total[slot] = n_tested
            bot_visibility[slot] = set()
            bot_fractions[slot] = {}

        if targets:
            # skip_slots[i] = the bot the bone belongs to — engine ignores its capsule
            # so trace passes through to the actual bone position (otherwise we'd hit
            # the bot's own hull at fraction~0.97).
            skip_slots = [slot for (slot, _bone) in bone_keys]
            res = self.client.trace_visibility(viewer_eye, targets,
                                                tolerance=VISIBILITY_TOLERANCE,
                                                skip_slots=skip_slots)
            if res is None:
                print(f'  [{idx}] WARN: trace_visibility plugin call failed — assuming all visible')
                for slot, _bone in bone_keys:
                    bot_visibility[slot].add(_bone)
                    bot_fractions[slot][_bone] = -1.0  # unknown
            else:
                vis_arr  = res.get('visible',   [])
                frac_arr = res.get('fractions', [])
                for i, (slot, bone_idx) in enumerate(bone_keys):
                    f = frac_arr[i] if i < len(frac_arr) else -1.0
                    bot_fractions[slot][bone_idx] = float(f)
                    if i < len(vis_arr) and vis_arr[i]:
                        bot_visibility[slot].add(bone_idx)

        for slot in slot_to_team:
            n_vis = len(bot_visibility.get(slot, set()))
            print(f'  [{idx}] bot slot={slot} team={slot_to_team[slot]} '
                  f'visible_bones={n_vis}/{per_bot_total.get(slot, 0)}')

        # No restoration needed — vischeck doesn't touch game state
        lap('visibility_check', t_vis)

        # ---- Restart round to wipe blood decals from bot models ----
        # After restart: bots respawn at default spawn points (clean models).
        # We re-apply same poses → same bone positions → cached visibility still valid.
        t = time.perf_counter()
        self.client.restart_round()
        time.sleep(1.8)  # wait for respawn + match settle

        # Re-place bots at exact same positions (OnTick keeps re-applying, but force here)
        self.client.set_poses(poses)
        time.sleep(0.3)

        # Re-teleport viewer (respawned at team spawn after restart)
        self._teleport_player(*viewer['pos'])
        self.client.set_player_view(yaw=viewer['yaw'], pitch=viewer['pitch'])
        time.sleep(0.3)

        # Wait for bot bones to repopulate after respawn (may take a few ticks)
        for attempt in range(6):
            r = self.client.get_geometry()
            if r.ok:
                geo_check = json.loads(r.message)
                ready = sum(1 for b in geo_check['bots']
                            if b['slot'] in slot_to_team and b.get('bones'))
                if ready == len(slot_to_team):
                    break
            print(f'  [{idx}] post-restart bones not ready, retry {attempt+1}/6')
            time.sleep(0.3)
        lap('restart_round+respawn', t)

        # ---- Spawn planted bomb (if scenario has one) ----
        # Done after restart_round (which wipes any previous bomb) and before
        # pre_break + screenshot. Bomb bbox is computed at screenshot time.
        planted_bomb = snapshot.get('planted_bomb')
        if planted_bomb:
            self.client.spawn_planted_bomb(planted_bomb['pos'], planted_bomb.get('yaw', 0.0))
            time.sleep(0.3)

        # ---- Second pre-break: restore destructible state to match visibility measurement ----
        # restart_round restored all breakables to intact. Re-shoot lateral offsets so
        # screenshot's world state matches what visibility check was based on.
        # Bot centers are deterministic (same poses), so same shots → same result.
        if bot_centers:
            t = time.perf_counter()
            # Player needs Glock again — restart may have changed weapon
            if self.cmd:
                self.cmd.send(f'give weapon_{VISIBILITY_WEAPON}'); time.sleep(0.10)
                self.cmd.send('slot2'); time.sleep(0.10)
            # Bots invincible to absorb stray bullets that hit them
            self.client._send({
                'action': 'set_bot_health',
                'poses': [{'slot': s, 'hp': 9999, 'yaw': 0, 'pitch': 0} for s in slot_to_team],
            })
            time.sleep(0.1)
            self.client.host_timescale(6.0)
            for slot, center in bot_centers.items():
                pre_break_around_bot(self.client, viewer_eye, center)
            self.client.host_timescale(1.0)
            self.client._send({'action': 'cleanup_inputs'})
            lap('pre_break_post_restart', t)

        # Give bots realistic random weapons (per-team weighted)
        t = time.perf_counter()
        weapon_assignments = []
        for slot, team in slot_to_team.items():
            wpn = sample_weapon(self.rng, team)
            weapon_assignments.append({'slot': slot, 'weapon': wpn})
        self.client._send({'action': 'give_weapon', 'poses': weapon_assignments})

        # Give player a random scenario weapon based on viewer's team
        if self.cmd:
            viewer_weapon = sample_weapon(self.rng, viewer['team'])
            self.cmd.send(f'give weapon_{viewer_weapon}')
            time.sleep(0.10)
            slot_n = '2' if viewer_weapon in ('glock','usp_silencer','hkp2000','p250',
                                                'fiveseven','tec9','cz75a','deagle',
                                                'revolver') else '1'
            self.cmd.send(f'slot{slot_n}')
            time.sleep(0.10)
        time.sleep(0.15)

        lap('weapons', t)

        # ---- Pre-compute bomb visibility (camera-position dependent, view-direction independent) ----
        # Visibility is line-of-sight from eye → bone. Eye doesn't change between views,
        # so we compute once and reuse for all view angles.
        bomb_world_data = []  # list of bomb dicts (visible only)
        if planted_bomb:
            r = self.client.get_planted_bombs()
            if r.ok:
                bombs = json.loads(r.message).get('bombs', [])
                if bombs:
                    bomb_targets = [tuple(b['pos']) for b in bombs]
                    res = self.client.trace_visibility(viewer_eye, bomb_targets, tolerance=VISIBILITY_TOLERANCE)
                    vis_arr = res.get('visible', []) if res else [True] * len(bombs)
                    for bomb, v in zip(bombs, vis_arr):
                        if v: bomb_world_data.append(bomb)

        # ---- Multi-view capture ----
        # Same bot positions + bomb, different camera angles. Visibility cache is reused
        # (line-of-sight is camera-pos dependent only). Per view: rotate camera, screenshot,
        # re-project bones with new view matrix, write annotations.
        t = time.perf_counter()
        intended_pos = {p.slot: p.pos for p in poses}
        DISPLACEMENT_THRESHOLD = 100.0

        # Generate N view angles: view 0 = original, rest = random perturbations (per-scenario seed)
        view_angles = [(viewer['yaw'], viewer['pitch'])]
        view_rng = random.Random(idx * 9973)  # deterministic per scenario
        for _ in range(N_VIEWS_PER_SCENARIO - 1):
            dyaw   = view_rng.uniform(-VIEW_YAW_RANGE, VIEW_YAW_RANGE)
            dpitch = view_rng.uniform(-VIEW_PITCH_RANGE, VIEW_PITCH_RANGE)
            yaw_v   = viewer['yaw'] + dyaw
            pitch_v = max(-89.0, min(89.0, viewer['pitch'] + dpitch))
            view_angles.append((yaw_v, pitch_v))

        total_anns = 0
        for view_idx, (vyaw, vpitch) in enumerate(view_angles):
            # Set view + brief settle
            self.client.set_player_view(yaw=vyaw, pitch=vpitch)
            time.sleep(0.15)

            img = ImageGrab.grab(bbox=self.cs2_bbox, all_screens=True).convert('RGB')

            mat_r = self.client.get_view_matrix()
            geo_r = self.client.get_geometry()
            if not (mat_r.ok and geo_r.ok):
                print(f'  [{idx}] view {view_idx}: matrix/geo read failed — skip')
                continue
            mat = json.loads(mat_r.message)['matrix']
            geo = json.loads(geo_r.message)

            img_filename = f'scenario_{idx:06d}_view{view_idx}.jpg'
            img.save(os.path.join(self.frames_dir, img_filename), quality=90)

            self.coco_img_id += 1
            self.coco_images.append({
                'id': self.coco_img_id,
                'file_name': img_filename,
                'width': self.win_w,
                'height': self.win_h,
            })

            n_anns_for_image = self._annotate_view(geo, mat, bot_visibility, bot_fractions,
                                                     slot_to_team,
                                                     intended_pos, DISPLACEMENT_THRESHOLD,
                                                     bomb_world_data, idx)
            total_anns += n_anns_for_image

        lap('screenshot+annotate', t)

        # ---- Print per-phase timing summary ----
        total = time.perf_counter() - t_scenario_start
        breakdown = ' | '.join(f'{k}={v:.2f}s' for k, v in timings.items())
        print(f'  [{idx}] saved {N_VIEWS_PER_SCENARIO} views with {total_anns} total annotations '
              f'(total={total:.2f}s)')
        print(f'        timing: {breakdown}')
        # Trailing restart_round dropped — next scenario's prepare_match does mp_restartgame anyway.

    def save_coco(self):
        coco = {
            'images': self.coco_images,
            'annotations': self.coco_anns,
            'categories': [
                {'id': v, 'name': k} for k, v in CLASS_IDS.items()
            ],
        }
        out_path = os.path.join(self.output_dir, 'annotations.json')
        with open(out_path, 'w') as f:
            json.dump(coco, f, separators=(',', ':'))
        print(f'\nSaved COCO annotations: {len(self.coco_images)} images, '
              f'{len(self.coco_anns)} bboxes → {out_path}')

    def _annotate_view(self, geo, mat, bot_visibility, bot_fractions, slot_to_team,
                       intended_pos, displacement_threshold, bomb_world_data, idx):
        """Write COCO annotations for current image_id from given view's geometry.
        Reuses cached bot_visibility (eye-position-dependent only) and bomb_world_data
        (filtered to visible bombs only).
        Returns count of annotations added.
        """
        n = 0
        # Bots
        for bot_data in geo['bots']:
            slot = bot_data['slot']
            if slot not in bot_visibility or not bot_data.get('alive'):
                continue
            visible_bone_set = bot_visibility[slot]
            if len(visible_bone_set) < MIN_VISIBLE_BONES:
                continue

            intended = intended_pos.get(slot)
            actual = bot_data.get('origin')
            if intended and actual:
                dx = actual[0] - intended[0]
                dy = actual[1] - intended[1]
                dz = actual[2] - intended[2]
                disp = math.sqrt(dx*dx + dy*dy + dz*dz)
                if disp > displacement_threshold:
                    continue  # already logged in earlier displacement check

            team = slot_to_team[slot]
            bones = bot_data.get('bones', {})

            # Body bbox
            visible_positions = [tuple(bones[str(i)]) for i in visible_bone_set
                                   if str(i) in bones and bones[str(i)] is not None]
            body_bbox = project_aabb_from_bones(visible_positions, mat, self.win_w, self.win_h)

            # Debug: project EVERY bone to screen + tag visibility + raw fraction
            bones_debug = []
            slot_fracs = bot_fractions.get(slot, {})
            for idx_str, pos in bones.items():
                bone_idx = int(idx_str)
                if bone_idx in SKIP_BONES or pos is None:
                    continue
                sx, sy, depth = world_to_screen_via_matrix(tuple(pos), mat, self.win_w, self.win_h)
                if depth <= 0:
                    continue
                bones_debug.append({
                    'idx':  bone_idx,
                    'sx':   float(sx),
                    'sy':   float(sy),
                    'vis':  bone_idx in visible_bone_set,
                    'frac': slot_fracs.get(bone_idx, -1.0),
                })

            if body_bbox:
                self.coco_ann_id += 1
                x1, y1, x2, y2 = body_bbox
                self.coco_anns.append({
                    'id': self.coco_ann_id,
                    'image_id': self.coco_img_id,
                    'category_id': CLASS_IDS[team],
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'area': (x2 - x1) * (y2 - y1),
                    'iscrowd': 0,
                    'bones_debug': bones_debug,
                    'slot': slot,
                })
                n += 1

            # Head bbox
            if 7 in visible_bone_set:
                head_pos = bones.get('7'); neck_pos = bones.get('6')
                if head_pos and neck_pos:
                    head_bbox = project_head_bbox(tuple(head_pos), tuple(neck_pos),
                                                    mat, self.win_w, self.win_h)
                    if head_bbox:
                        self.coco_ann_id += 1
                        x1, y1, x2, y2 = head_bbox
                        self.coco_anns.append({
                            'id': self.coco_ann_id,
                            'image_id': self.coco_img_id,
                            'category_id': CLASS_IDS[f'{team}_head'],
                            'bbox': [x1, y1, x2 - x1, y2 - y1],
                            'area': (x2 - x1) * (y2 - y1),
                            'iscrowd': 0,
                        })
                        n += 1

        # Bombs (already filtered to visible during pre-compute)
        for bomb in bomb_world_data:
            bbox = project_entity_aabb(
                bomb['pos'], bomb['rotation'], bomb['mins'], bomb['maxs'],
                mat, self.win_w, self.win_h,
            )
            if bbox is None:
                continue
            origin_sx, origin_sy, origin_d = world_to_screen_via_matrix(
                bomb['pos'], mat, self.win_w, self.win_h)
            if origin_d <= 0:
                continue

            x1, y1, x2, y2 = bbox
            w_b, h_b = x2 - x1, y2 - y1
            adj = ANCHOR_ADJUST.get('bomb', {})
            shift_up_frac = adj.get('shift_up_frac', 0.0)
            pad_frac      = adj.get('horizontal_pad_frac', 0.0)
            new_w = w_b * (1 + 2 * pad_frac)
            new_h = h_b
            cx = origin_sx
            cy = origin_sy - shift_up_frac * h_b
            nx1 = cx - new_w / 2; ny1 = cy - new_h / 2
            nx2 = cx + new_w / 2; ny2 = cy + new_h / 2
            nx1 = max(0, nx1); ny1 = max(0, ny1)
            nx2 = min(self.win_w, nx2); ny2 = min(self.win_h, ny2)
            if (nx2 - nx1) < 4 or (ny2 - ny1) < 4:
                continue

            self.coco_ann_id += 1
            self.coco_anns.append({
                'id': self.coco_ann_id,
                'image_id': self.coco_img_id,
                'category_id': CLASS_IDS['bomb'],
                'bbox': [nx1, ny1, nx2 - nx1, ny2 - ny1],
                'area': (nx2 - nx1) * (ny2 - ny1),
                'iscrowd': 0,
                'pos_screen': [float(origin_sx), float(origin_sy)],
            })
            n += 1
        return n

    def close(self):
        try: self.client.host_timescale(1.0)
        except: pass
        self.client.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshots', required=True)
    ap.add_argument('--output',    required=True)
    ap.add_argument('--map',       default='de_mirage')
    ap.add_argument('--view-matrix-offset', type=lambda s: int(s, 0), default=0x232E9C0)
    ap.add_argument('--max-scenarios', type=int, default=None)
    ap.add_argument('--seed',      type=int, default=42)
    ap.add_argument('--setup-delay', type=float, default=10.0)
    args = ap.parse_args()

    with open(args.snapshots) as f:
        snapshots = json.load(f)
    if args.max_scenarios:
        snapshots = snapshots[:args.max_scenarios]
    print(f'Loaded {len(snapshots)} snapshots')

    pipeline = MultiBotPipeline(
        output_dir=args.output,
        view_matrix_offset=args.view_matrix_offset,
        seed=args.seed,
    )
    pipeline.setup(map_name=args.map, setup_delay=args.setup_delay)

    try:
        for i, snap in enumerate(snapshots):
            try:
                pipeline.run_scenario(snap, i)
            except Exception as ex:
                print(f'  [{i}] ERROR: {ex}')
        pipeline.save_coco()
    finally:
        pipeline.close()


if __name__ == '__main__':
    main()
