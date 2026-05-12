"""Simple detection-data pipeline — uses existing bots, no kick/respawn cycles.

Flow per snapshot:
  1. list_bots -> get current bot slots
  2. For each "other player" in snapshot, assign one of the bots (any team)
  3. set_poses to teleport bots to scenario positions
  4. Teleport viewer to scenario eye position
  5. Capture screenshot
  6. trace_visibility_batch for every bone of every bot (skip target bot itself)
  7. Project visible bones to screen, build per-bot bbox
  8. Save image + COCO annotation

Usage:
    python simple_pipeline.py --snapshots snapshots.json --output D:/DetectionDataset
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import List

from bot_pose_client import BotPose, BotPoseClient

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_collect_v2'))
try:
    from cs2_cmd import CS2Commander
except ImportError:
    CS2Commander = None


SKIP_BONES = {0, 24}
TOLERANCE  = 30.0
MIN_VISIBLE = 2
HEAD_BONE = 7
NECK_BONE = 6

CLASS_IDS = {'ct': 0, 't': 1, 'ct_head': 2, 't_head': 3, 'bomb': 4}
CLASS_NAMES = {v: k for k, v in CLASS_IDS.items()}

# Re-use proven projection helpers from old pipeline
from test_bbox_via_matrix import world_to_screen_via_matrix


def project_body_bbox(bone_positions, mat, w, h):
    """AABB around projected bones, expanded to cover clothes/weapon (asymmetric)."""
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
    pad_x   = 0.18 * bw      # +18% sides (arms, weapon, clothes)
    pad_top = 0.15 * bh      # +15% top (helmet/hair above head bone)
    pad_bot = 0.03 * bh
    x1 -= pad_x; x2 += pad_x; y1 -= pad_top; y2 += pad_bot
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
    if (x2 - x1) < 4 or (y2 - y1) < 4:
        return None
    return [float(x1), float(y1), float(x2), float(y2)]


def project_head_bbox(head_pos, neck_pos, mat, w, h):
    """Asymmetric head AABB built from 8 corners offset from head bone by neck-to-head distance."""
    dx = head_pos[0] - neck_pos[0]
    dy = head_pos[1] - neck_pos[1]
    dz = head_pos[2] - neck_pos[2]
    d  = math.sqrt(dx*dx + dy*dy + dz*dz)
    if d < 0.5: return None
    cx, cy, cz = head_pos
    pts = []
    for ex in (-d, d):
        for ey in (-d, d):
            for ez in (-d, 1.2 * d):
                sx, sy, dpt = world_to_screen_via_matrix(
                    (cx + ex, cy + ey, cz + ez), mat, w, h)
                if dpt > 0: pts.append((sx, sy))
    if len(pts) != 8: return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if x2 < 0 or y2 < 0 or x1 > w or y1 > h: return None
    if (x2 - x1) > w or (y2 - y1) > h: return None
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
    return [float(x1), float(y1), float(x2), float(y2)]


def occlude_by_closer_bots(per_slot_data, eye):
    """NMS-style: for each bot, mark bones occluded if they fall inside a CLOSER bot's
    body silhouette on screen. Closer = smaller world distance from camera eye.

    per_slot_data: dict slot -> {
        'world_origin': (x,y,z),         # bot world origin
        'bones': [{'idx', 'sx', 'sy', 'world': (x,y,z), 'vis_geom': bool}, ...]
    }
    Returns same structure with 'vis' key set per bone (vis_geom AND not occluded by closer).
    Also returns 'body_screen_bbox' for each bot computed from remaining visible bones.
    """
    # Sort slots by distance camera->origin (near first)
    def dist(o):
        dx, dy, dz = o[0]-eye[0], o[1]-eye[1], o[2]-eye[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    order = sorted(per_slot_data.keys(), key=lambda s: dist(per_slot_data[s]['world_origin']))

    # Build closer-bots' silhouette boxes incrementally
    closer_boxes = []   # [(slot, x1, y1, x2, y2)]

    for slot in order:
        data = per_slot_data[slot]
        my_origin = data['world_origin']
        my_dist   = dist(my_origin)

        # Mark bones occluded if inside any CLOSER bot's silhouette AND closer in world
        for b in data['bones']:
            b_world_dist = math.sqrt(sum((b['world'][i] - eye[i])**2 for i in range(3)))
            occluded = False
            for cs, x1, y1, x2, y2 in closer_boxes:
                if x1 <= b['sx'] <= x2 and y1 <= b['sy'] <= y2:
                    occluded = True
                    break
            b['vis'] = b['vis_geom'] and not occluded

        # Compute THIS bot's silhouette from remaining visible bones (asymmetric expansion)
        vis_pts = [(b['sx'], b['sy']) for b in data['bones'] if b['vis']]
        if len(vis_pts) >= 2:
            xs = [p[0] for p in vis_pts]; ys = [p[1] for p in vis_pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            bw, bh = x2-x1, y2-y1
            x1 -= 0.18*bw; x2 += 0.18*bw; y1 -= 0.15*bh; y2 += 0.03*bh
            data['body_screen_bbox'] = (x1, y1, x2, y2)
            closer_boxes.append((slot, x1, y1, x2, y2))
        else:
            data['body_screen_bbox'] = None

    return per_slot_data


def ensure_bots(client, total_needed, current_bots):
    """Spawn missing bots so we have at least total_needed total. Returns updated list."""
    missing = total_needed - len(current_bots)
    if missing <= 0:
        return current_bots
    print(f'  spawning {missing} extra bots...')
    # Spawn whichever team doesn't matter — set_poses+ChangeTeam handles team assignment
    for _ in range(missing):
        client.spawn_bot(team='t')
        time.sleep(0.4)
    # Re-list to get new slots
    return list_bots_robust(client)


def get_window_rect():
    """Find CS2 window rect via win32 API for screen-space projection."""
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, "Counter-Strike 2")
    if not hwnd:
        raise RuntimeError("CS2 window not found")
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    pt_tl = wintypes.POINT(rect.left,  rect.top)
    pt_br = wintypes.POINT(rect.right, rect.bottom)
    user32.ClientToScreen(hwnd, ctypes.byref(pt_tl))
    user32.ClientToScreen(hwnd, ctypes.byref(pt_br))
    return hwnd, pt_tl.x, pt_tl.y, pt_br.x, pt_br.y


def world_to_screen(pos, mat, w, h):
    """4x4 row-major view-projection matrix * (x,y,z,1) -> screen px + depth."""
    x, y, z = pos
    cx = mat[0]*x + mat[1]*y + mat[2]*z  + mat[3]
    cy = mat[4]*x + mat[5]*y + mat[6]*z  + mat[7]
    cz = mat[8]*x + mat[9]*y + mat[10]*z + mat[11]
    cw = mat[12]*x + mat[13]*y + mat[14]*z + mat[15]
    if cw < 1e-6:
        return None
    nx = cx / cw; ny = cy / cw
    sx = (nx * 0.5 + 0.5) * w
    sy = (1.0 - (ny * 0.5 + 0.5)) * h
    return (sx, sy, cz)


def teleport_player(cmd, x, y, z):
    cmd.send_batch(["god 1", "noclip 1", f"setpos {x:.2f} {y:.2f} {z + 10:.2f}"])
    time.sleep(0.25)
    cmd.send("noclip 0")
    time.sleep(0.25)


def list_bots_robust(client):
    """Returns list of bot slot ints. Plugin's list_bots returns
    'slot=N name=X team=M, slot=N name=Y team=M, ...' format."""
    r = client.list_bots()
    if not r.ok:
        return []
    bots = []
    for chunk in (r.message or '').split(','):
        slot = team = None
        for tok in chunk.strip().split():
            if tok.startswith('slot='): slot = int(tok.split('=', 1)[1])
            if tok.startswith('team='): team = int(tok.split('=', 1)[1])
        if slot is not None:
            bots.append({'slot': slot, 'team': team})
    return bots


def grab_screenshot(out_path):
    """PIL ImageGrab of CS2 window region. CS2 must be in the foreground
    (otherwise you'll capture whatever window is on top instead)."""
    from PIL import ImageGrab
    _, l, t, r, b = get_window_rect()
    img = ImageGrab.grab(bbox=(l, t, r, b))
    img.save(out_path, quality=92)
    return img.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshots', required=True)
    ap.add_argument('--output',    required=True)
    ap.add_argument('--max',       type=int, default=1)
    args = ap.parse_args()

    snapshots = json.load(open(args.snapshots))
    print(f'Loaded {len(snapshots)} snapshots')

    os.makedirs(os.path.join(args.output, 'frames'), exist_ok=True)

    coco = {
        'images': [],
        'annotations': [],
        'categories': [{'id': v, 'name': k} for k, v in CLASS_IDS.items()],
    }
    img_id = 0
    ann_id = 0

    cmd = CS2Commander() if CS2Commander else None
    if cmd: cmd.connect()

    with BotPoseClient(timeout=15.0) as client:
        # ---- one-time setup ----
        _, l, t, r_x, b = get_window_rect()
        win_w, win_h = r_x - l, b - t
        print(f'Window: {win_w}x{win_h}')

        # Initialize view matrix offset (one-time, lost on plugin reload)
        VIEW_MATRIX_OFFSET = 0x232E9C0
        r_vm = client.set_view_matrix_offset(VIEW_MATRIX_OFFSET)
        print(f'set_view_matrix_offset(0x{VIEW_MATRIX_OFFSET:X}): {r_vm.ok} {r_vm.message}')

        bots = list_bots_robust(client)
        print(f'Found {len(bots)} bots: {[b["slot"] for b in bots]}')

        if not bots:
            print('No bots in game. Spawn via console: bot_quota_mode fill; bot_quota 6')
            return

        for s_idx, snap in enumerate(snapshots[:args.max]):
            print(f'\n=== Scenario {s_idx} ===')

            # Restart round so bots are alive/healthy/clean (no blood, no leftover state)
            client._send({'action': 'restart_round'})
            time.sleep(2.0)
            # Re-list bots (slots may shift after respawn)
            bots = list_bots_robust(client)

            viewer = snap['viewer']
            others = list(snap['other_players'])
            if snap.get('kill_target'):
                others.append({**snap['kill_target']})

            # Spawn extra bots if we don't have enough
            need_total = len(others)
            if len(bots) < need_total:
                bots = ensure_bots(client, need_total, bots)
                print(f'  bots after spawn: {[b["slot"] for b in bots]} ({len(bots)}/{need_total})')
                if len(bots) < need_total:
                    print(f'  warn: still short — {len(bots)}/{need_total}, scenario will be partial')

            # Assign each "other" to a bot slot
            poses = []
            slot_used = []
            for i, o in enumerate(others):
                if i >= len(bots): break
                slot = bots[i]['slot']
                slot_used.append({'slot': slot, 'team': o['team']})
                poses.append(BotPose(
                    slot=slot, pos=o['pos'], yaw=o['yaw'], pitch=o['pitch'],
                    ducking=o.get('ducking', False), hp=100, freeze=True,
                ))
            if not poses:
                print(f'  no poses, skip'); continue

            # Move bots to correct teams BEFORE positioning (respawn replaces pawn)
            team_req = [{'slot': s['slot'], 'weapon': s['team']} for s in slot_used]
            r_team = client._send({'action': 'set_bot_teams', 'poses': team_req})
            if r_team.ok and r_team.applied > 0:
                print(f'  reassigned {r_team.applied} bots to correct teams')
                time.sleep(1.0)   # wait for respawn before set_poses

            # Hide UNUSED bots far away under the map so they don't appear in FOV
            # or interfere visually with the captured frame.
            used_slots = {s['slot'] for s in slot_used}
            unused = [b for b in bots if b['slot'] not in used_slots]
            if unused:
                hide_poses = [BotPose(slot=b['slot'], pos=[10000.0, 10000.0, -5000.0],
                                       yaw=0, pitch=0, ducking=False, hp=100, freeze=True)
                              for b in unused]
                client.set_poses(hide_poses)
                print(f'  hid {len(unused)} unused bots far away: {[b["slot"] for b in unused]}')

            client.set_poses(poses)
            time.sleep(0.4)

            # Teleport viewer
            if cmd:
                teleport_player(cmd, *viewer['pos'])
            client.set_player_view(yaw=viewer['yaw'], pitch=viewer['pitch'])
            time.sleep(0.3)

            # Get geometry (eye + bot bones) AFTER positioning
            r_geo = client._send({'action': 'get_geometry'})
            if not r_geo.ok:
                print(f'  get_geometry failed'); continue
            geo = json.loads(r_geo.message)
            eye = list(geo['viewer']['eye'])
            print(f'  eye={[round(v,1) for v in eye]}')

            # Get view matrix for screen projection
            r_mat = client.get_view_matrix()
            mat = json.loads(r_mat.message)['matrix'] if r_mat.ok else None
            if mat is None:
                print('  no view matrix, skip'); continue

            # Build bone target list with skip_slots = target bot's own slot
            bone_keys = []           # (slot, bone_idx, world_pos)
            targets   = []           # parallel list of [x,y,z]
            skip_slots = []          # parallel list of slot to skip
            for bot in geo['bots']:
                slot = bot['slot']
                if slot not in [s['slot'] for s in slot_used]: continue
                bones = bot.get('bones', {}) or {}
                for idx_str, pos in bones.items():
                    idx = int(idx_str)
                    if idx in SKIP_BONES or pos is None: continue
                    bone_keys.append((slot, idx, tuple(pos)))
                    targets.append(list(pos))
                    skip_slots.append(slot)

            # Batch trace_visibility (skip viewer's own pawn AND each target bot)
            if not targets:
                print(f'  no bones to trace'); continue

            # DIAG: positions
            for bot in geo['bots']:
                if bot['slot'] in [s['slot'] for s in slot_used]:
                    print(f'  bot slot={bot["slot"]} origin={[round(v,1) for v in bot["origin"]]} '
                          f'team={bot["team"]} alive={bot["alive"]} bones={len(bot.get("bones",{}) or {})}')

            res = client.trace_visibility(eye, targets, tolerance=TOLERANCE, skip_slots=skip_slots)
            if res is None:
                print('  trace_visibility failed'); continue

            vis_arr  = res.get('visible',   [])
            frac_arr = res.get('fractions', [])
            cont_arr = res.get('contents',  [])
            hit_names = res.get('hit_names', [])
            n_total  = res.get('n_total',  len(targets))
            n_vis    = res.get('n_visible', sum(vis_arr))
            print(f'  visibility: {n_vis}/{n_total}')
            # DIAG: show first 3 trace results
            for i in range(min(3, len(vis_arr))):
                slot, idx, _ = bone_keys[i]
                print(f'    trace#{i}: slot={slot} bone={idx} '
                      f'frac={frac_arr[i]:.3f} cont=0x{cont_arr[i]:X} '
                      f'name={hit_names[i] if i < len(hit_names) else "?"!r}')

            # Group visibility per bot — keep raw geom-vis + world pos for NMS
            per_slot = {}
            for bot in geo['bots']:
                slot = bot['slot']
                if slot in [s['slot'] for s in slot_used]:
                    per_slot[slot] = {'world_origin': tuple(bot['origin']), 'bones': []}

            for k_i, (slot, idx, world_pos) in enumerate(bone_keys):
                proj = world_to_screen(world_pos, mat, win_w, win_h)
                if proj is None or proj[2] <= 0: continue
                sx, sy, _ = proj
                vis_geom = vis_arr[k_i] if k_i < len(vis_arr) else False
                frac = frac_arr[k_i] if k_i < len(frac_arr) else -1.0
                per_slot[slot]['bones'].append({
                    'idx': idx, 'vis_geom': bool(vis_geom), 'frac': float(frac),
                    'sx': float(sx), 'sy': float(sy), 'world': world_pos,
                })

            # NMS: closer bot's silhouette occludes farther bot's bones
            per_slot = occlude_by_closer_bots(per_slot, eye)
            per_bot = {slot: data['bones'] for slot, data in per_slot.items()}

            # DIAG: dump per-slot screen coverage so we can tell NMS-occlude vs trace-occlude
            for slot, data in per_slot.items():
                bs = data['bones']
                if not bs: continue
                xs = [b['sx'] for b in bs]; ys = [b['sy'] for b in bs]
                geom_vis = sum(1 for b in bs if b['vis_geom'])
                final_vis = sum(1 for b in bs if b['vis'])
                in_frame = sum(1 for b in bs if 0 <= b['sx'] <= win_w and 0 <= b['sy'] <= win_h)
                print(f'    slot {slot}: screen x=[{min(xs):.0f}..{max(xs):.0f}] '
                      f'y=[{min(ys):.0f}..{max(ys):.0f}], in_frame={in_frame}/{len(bs)}, '
                      f'geom_vis={geom_vis}, final_vis={final_vis} (NMS removed {geom_vis-final_vis})')

            # Save screenshot
            img_id += 1
            fname = f'scenario_{s_idx:03d}.jpg'
            grab_screenshot(os.path.join(args.output, 'frames', fname))
            coco['images'].append({
                'id': img_id, 'file_name': fname, 'width': win_w, 'height': win_h,
            })

            # Cache world bone positions per slot for proper body+head bbox projection
            world_bones = {}
            for bot in geo['bots']:
                if bot['slot'] in [s['slot'] for s in slot_used]:
                    world_bones[bot['slot']] = bot.get('bones', {}) or {}

            for s_used in slot_used:
                slot = s_used['slot']; team = s_used['team']
                bones_dbg = per_bot.get(slot, [])
                visible_set = {b['idx'] for b in bones_dbg if b['vis']}
                if len(visible_set) < MIN_VISIBLE:
                    print(f'  slot {slot} ({team}): skip - only {len(visible_set)} visible bones')
                    continue

                # BODY bbox: build from world bone positions, expanded for clothes/weapon
                wb = world_bones.get(slot, {})
                vis_world = [tuple(wb[str(i)]) for i in visible_set
                              if str(i) in wb and wb[str(i)] is not None]
                body_bbox = project_body_bbox(vis_world, mat, win_w, win_h)
                if body_bbox is None: continue

                x1, y1, x2, y2 = body_bbox
                ann_id += 1
                coco['annotations'].append({
                    'id': ann_id, 'image_id': img_id,
                    'category_id': CLASS_IDS[team],
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'area': (x2 - x1) * (y2 - y1),
                    'iscrowd': 0,
                    'slot': slot,
                    'bones_debug': bones_dbg,
                })

                # HEAD bbox if head bone (7) visible AND neck (6) available
                head_added = False
                if HEAD_BONE in visible_set and str(NECK_BONE) in wb:
                    head_pos = wb.get(str(HEAD_BONE)); neck_pos = wb.get(str(NECK_BONE))
                    if head_pos and neck_pos:
                        hb = project_head_bbox(tuple(head_pos), tuple(neck_pos),
                                                mat, win_w, win_h)
                        if hb:
                            hx1, hy1, hx2, hy2 = hb
                            ann_id += 1
                            coco['annotations'].append({
                                'id': ann_id, 'image_id': img_id,
                                'category_id': CLASS_IDS[f'{team}_head'],
                                'bbox': [hx1, hy1, hx2 - hx1, hy2 - hy1],
                                'area': (hx2 - hx1) * (hy2 - hy1),
                                'iscrowd': 0, 'slot': slot,
                            })
                            head_added = True
                print(f'  slot {slot} ({team}): body {[round(v) for v in [x1,y1,x2,y2]]}'
                      + (' + head' if head_added else '')
                      + f', {len(visible_set)} vis bones')

    json.dump(coco, open(os.path.join(args.output, 'annotations.json'), 'w'), indent=2)
    print(f'\nDone: {len(coco["images"])} images, {len(coco["annotations"])} bboxes -> '
          f'{args.output}/annotations.json')

    if cmd: cmd.close()


if __name__ == '__main__':
    main()
