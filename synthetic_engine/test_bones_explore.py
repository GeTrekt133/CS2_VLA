"""
Read skeleton bones from a single bot, project all bone positions onto screenshot
as numbered dots. Use this to identify which bone index = head, pelvis, hands, etc.

After identification, use those indices in test_bbox_via_matrix.py for precise bbox.

Usage:
    python test_bones_explore.py --offset 0x232E9C0 --slot 1 --delay 3
"""

from __future__ import annotations

import argparse
import json
import time

from PIL import Image, ImageDraw, ImageFont, ImageGrab

from bot_pose_client import BotPose, BotPoseClient
from test_bbox_projection import find_cs2_window
from test_bbox_via_matrix import world_to_screen_via_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--offset', type=lambda s: int(s, 0), required=True)
    ap.add_argument('--slot', type=int, default=None,
                    help='Bot slot to inspect (default: first alive bot)')
    ap.add_argument('--delay', type=float, default=2.0)
    ap.add_argument('--out', default='bones_overlay.png')
    args = ap.parse_args()

    with BotPoseClient(timeout=10.0) as c:
        c.set_view_matrix_offset(args.offset)

        print('[1] Querying bones...')
        req = {'action': 'get_bones'}
        if args.slot is not None:
            req['poses'] = [{'slot': args.slot, 'yaw': 0, 'pitch': 0, 'freeze': False}]
        r = c._send(req)
        if not r.ok:
            print(f'    FAILED: {r.message}')
            return
        data = json.loads(r.message)
        if 'bones' not in data:
            # Diagnostic mode — print all candidate fields & values
            print('=== DIAGNOSTIC DUMP ===')
            print(json.dumps(data, indent=2))
            return
        bones = data['bones']
        print(f'    bot slot={data["slot"]} team={data["team"]} bone_count={data["bone_count"]}')
        if 'debug' in data:
            print(f'    debug={data["debug"]}')
        print(f'    bot origin={data["origin"]}')

        # Print bone positions RELATIVE to origin so we can identify them
        ox, oy, oz = data['origin']
        print('\n    Bones (Z-relative to origin, sorted by Z):')
        rel = sorted(
            [(b['idx'], b['pos'][0] - ox, b['pos'][1] - oy, b['pos'][2] - oz) for b in bones],
            key=lambda t: -t[3],
        )
        for idx, dx, dy, dz in rel[:20]:
            print(f'      bone[{idx:3d}]  rel=({dx:+7.1f}, {dy:+7.1f}, {dz:+7.1f})')

        print('[2] Locating window + capturing...')
        bbox = find_cs2_window()
        if bbox is None:
            print('    FAILED: CS2 window not found')
            return
        L, T, R, B = bbox
        win_w, win_h = R - L, B - T

        time.sleep(args.delay)
        img = ImageGrab.grab(bbox=bbox, all_screens=True).convert('RGB')

        print('[3] Reading view matrix...')
        rv = c.get_view_matrix()
        mat = json.loads(rv.message)['matrix']

        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype('arial.ttf', 12)
        except Exception:
            font = ImageFont.load_default()

        for b in bones:
            sx, sy, d = world_to_screen_via_matrix(tuple(b['pos']), mat, win_w, win_h)
            if d <= 0:
                continue
            # tiny dot + index label
            r_ = 3
            draw.ellipse((sx - r_, sy - r_, sx + r_, sy + r_), fill=(255, 255, 0))
            draw.text((sx + 5, sy - 6), str(b['idx']), fill=(255, 255, 0), font=font)

        img.save(args.out)
        print(f'[4] Saved → {args.out}')
        print('\nNext: open the image and find which dot is on the bot\'s HEAD. '
              'That index is what we use for precise head bbox.')


if __name__ == '__main__':
    main()
