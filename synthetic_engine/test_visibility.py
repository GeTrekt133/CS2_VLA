"""
Test bone visibility check via bullet impact analysis.

For each bone (head/neck/chest), fires 1 bullet at it, reads impact location.
If impact ≈ bone position → bone visible. If impact much closer → wall in the way.

Setup: spray_setup.py first. Place a bot via prepare or pose. Aim at bot.

Usage:
    python test_visibility.py
"""

from __future__ import annotations

import json
import math
import time

from bot_pose_client import BotPose, BotPoseClient


def distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def world_to_yaw_pitch(eye, target):
    dx = target[0] - eye[0]
    dy = target[1] - eye[1]
    dz = target[2] - eye[2]
    yaw = math.degrees(math.atan2(dy, dx))
    dist = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(-dz, dist))
    return yaw, pitch


def check_bone_visibility(c, viewer_eye, bone_pos, bone_name='?',
                            settle_delay=0.15, fire_delay=0.10):
    """Fire 1 bullet at bone, return (visible, impact_pos, distance_from_bone)."""
    target_yaw, target_pitch = world_to_yaw_pitch(viewer_eye, bone_pos)

    # Aim and settle
    c.set_player_view(target_yaw, target_pitch)
    time.sleep(settle_delay)

    # Clear + record
    c._send({'action': 'clear_impacts'})
    c._send({'action': 'start_impact_record'})

    # Fire briefly
    c._send({'action': 'start_attack'})
    time.sleep(fire_delay)
    c._send({'action': 'stop_attack'})
    time.sleep(0.15)  # let impact arrive

    c._send({'action': 'stop_impact_record'})
    r = c._send({'action': 'get_impacts'})
    impacts = json.loads(r.message).get('impacts', [])

    if not impacts:
        return False, None, None

    # Use the closest impact to bone (in case multiple bullets fired)
    best = min(impacts, key=lambda i: distance((i['x'], i['y'], i['z']), bone_pos))
    imp_pos = (best['x'], best['y'], best['z'])
    d = distance(imp_pos, bone_pos)
    visible = d < 35.0  # ~head/torso radius
    return visible, imp_pos, d


def main():
    with BotPoseClient(timeout=10.0) as c:
        # Get viewer eye + bot bones
        r = c._send({'action': 'get_geometry'})
        geo = json.loads(r.message)
        viewer_eye = tuple(geo['viewer']['eye'])

        bot = next((b for b in geo['bots'] if b['alive']), None)
        if bot is None:
            print('No alive bot found')
            return

        print(f'Viewer at {viewer_eye}')
        print(f'Bot slot={bot["slot"]} team={bot["team"]} ducking={bot["ducking"]}')
        print(f'  origin   = {bot["origin"]}')
        print(f'  head     = {bot.get("head_bone")}')
        print(f'  neck     = {bot.get("neck_bone")}')
        print(f'  chest    = {bot.get("chest_bone")}')

        # Make bot invincible during the test
        c._send({'action': 'set_bot_health',
                 'poses': [{'slot': bot['slot'], 'hp': 9999, 'yaw': 0, 'pitch': 0}]})
        print('\n[Bot set to immortal: hp=9999]')
        time.sleep(0.2)

        # Named bones to test (verified for CS2 player model)
        BONE_NAMES = {
            '7':  'head',
            '13': 'shoulder_R',
            '9':  'shoulder_L',
            '15': 'hand_R',
            '11': 'hand_L',
            '22': 'foot_R',
            '19': 'foot_L',
            '1':  'torso',
        }

        bones_dict = bot.get('bones', {})
        bones_to_test = []
        for idx, name in BONE_NAMES.items():
            pos = bones_dict.get(idx)
            if pos:
                bones_to_test.append((name, tuple(pos)))

        print(f'\nTesting {len(bones_to_test)} bones...\n')

        results = {}
        for bone_name, bone_pos in bones_to_test:
            cam_to_bone = distance(viewer_eye, bone_pos)
            visible, imp_pos, d_imp_bone = check_bone_visibility(
                c, viewer_eye, bone_pos, bone_name,
            )
            cam_to_imp = distance(viewer_eye, imp_pos) if imp_pos else None

            results[bone_name] = visible
            print(f'  {bone_name:12s}: ', end='', flush=True)
            if imp_pos is None:
                print(f'NO IMPACT (off-map?)')
            else:
                short = 'VISIBLE' if visible else 'OCCLUDED'
                print(f'{short}  '
                      f'cam→bone={cam_to_bone:6.1f}  '
                      f'cam→impact={cam_to_imp:6.1f}  '
                      f'impact→bone={d_imp_bone:5.1f}')

        c._send({'action': 'cleanup_inputs'})

        n_visible = sum(1 for v in results.values() if v)
        print(f'\nSummary: {n_visible}/{len(results)} bones visible')


if __name__ == '__main__':
    main()
