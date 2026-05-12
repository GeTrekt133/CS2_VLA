"""
Extract REAL recoil pattern from CS2 by sampling pawn.m_aimPunchAngle during spray.

Steps:
  1. Make sure CS2 setup is done (spray_setup.py) — infinite ammo, god mode, AK in hand.
  2. Aim at a wall and run this script.
  3. Plugin records punch_angle every game tick during spray.
  4. Detect "kicks" (sudden jumps) = per-bullet recoil contribution.
  5. Save as JSON pattern + diff against McDaived/fbikat for comparison.

Usage:
    python extract_recoil.py --weapon ak47 --bullets 30 --output extracted_ak47.json

After extraction, the new pattern can be loaded by trajectory_generator instead of
hardcoded REAL_RECOIL_DELTAS.
"""

from __future__ import annotations

import argparse
import json
import time

from bot_pose_client import BotPoseClient
from trajectory_generator import REAL_RECOIL_DELTAS, PIXEL_TO_DEG


def detect_kicks(samples, threshold_deg=0.3):
    """Find sudden jumps in punch_angle = bullet kicks.

    samples: list of [tick, pitch, yaw] (degrees)
    Returns list of {bullet, tick, kick_pitch, kick_yaw} for each detected kick.
    """
    kicks = []
    if not samples:
        return kicks

    prev_pitch = samples[0][1]
    prev_yaw   = samples[0][2]
    for tick, p, y in samples[1:]:
        dp = p - prev_pitch
        dy = y - prev_yaw
        if abs(dp) > threshold_deg or abs(dy) > threshold_deg:
            kicks.append({
                'bullet':     len(kicks) + 1,
                'tick':       int(tick),
                'kick_pitch': float(dp),
                'kick_yaw':   float(dy),
            })
        prev_pitch = p
        prev_yaw   = y
    return kicks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weapon',  default='ak47')
    ap.add_argument('--bullets', type=int, default=30)
    ap.add_argument('--output',  default=None,
                    help='Output JSON. Default: extracted_recoil_<weapon>.json')
    ap.add_argument('--spray-time', type=float, default=4.0,
                    help='Real-time seconds to hold +attack')
    ap.add_argument('--threshold', type=float, default=0.3,
                    help='Min jump in degrees to count as a kick')
    args = ap.parse_args()

    out_path = args.output or f'extracted_recoil_{args.weapon}.json'

    with BotPoseClient(timeout=15.0) as c:
        print('[1/5] Starting punch_angle recording...')
        r = c._send({'action': 'start_punch_record'})
        if not r.ok:
            print(f'    FAILED: {r.message}')
            return

        print(f'[2/5] Holding +attack for {args.spray_time}s...')
        c._send({'action': 'start_attack'})
        time.sleep(args.spray_time)
        c._send({'action': 'stop_attack'})

        print('[3/5] Stopping recording...')
        r = c._send({'action': 'stop_punch_record'})
        print(f'    {r.message}')

        print('[4/5] Fetching samples...')
        r = c._send({'action': 'get_punch_record'})
        data = json.loads(r.message)
        samples = data['samples']  # list of [tick, pitch, yaw]
        print(f'    {len(samples)} samples collected')

        # Cleanup
        c._send({'action': 'cleanup_inputs'})

    # 5. Detect kicks
    print('[5/5] Detecting per-bullet kicks...')
    kicks = detect_kicks(samples, threshold_deg=args.threshold)
    print(f'    detected {len(kicks)} kicks (expected {args.bullets})')

    # Compare to fbikat/McDaived pattern
    fbi_pattern = REAL_RECOIL_DELTAS.get(args.weapon)
    if fbi_pattern is not None:
        print()
        print(f'  Comparison vs fbikat (first 10 bullets):')
        print(f'  {"#":>3} | {"extracted (deg)":<22} | {"fbikat (deg)":<22} | ratio')
        print(f'  {"-"*3}-+-{"-"*22}-+-{"-"*22}-+-{"-"*10}')
        for i in range(min(10, len(kicks), len(fbi_pattern))):
            ext_p = kicks[i]['kick_pitch']
            ext_y = kicks[i]['kick_yaw']
            # fbikat values are anti-recoil (player pulls down/right). Actual punch = -anti.
            fbi_dx, fbi_dy = fbi_pattern[i]
            fbi_p = fbi_dy * PIXEL_TO_DEG  # punch_pitch (positive = looking up)
            fbi_y = -fbi_dx * PIXEL_TO_DEG
            # Note signs: punch_angle.X is pitch, negative = looking up
            ratio_p = ext_p / fbi_p if abs(fbi_p) > 0.01 else float('nan')
            print(f'  {i+1:>3} | p={ext_p:+6.2f} y={ext_y:+6.2f}    | '
                  f'p={fbi_p:+6.2f} y={fbi_y:+6.2f}    | {ratio_p:+.2f}')

    # Compute summary stats
    total_kick_pitch = sum(k['kick_pitch'] for k in kicks)
    total_kick_yaw   = sum(k['kick_yaw']   for k in kicks)
    print(f'\n  Summary:')
    print(f'    cumulative pitch over spray: {total_kick_pitch:+.2f} deg')
    print(f'    cumulative yaw   over spray: {total_kick_yaw:+.2f} deg')
    if samples:
        print(f'    first sample: pitch={samples[0][1]:+.2f} yaw={samples[0][2]:+.2f}')
        print(f'    last  sample: pitch={samples[-1][1]:+.2f} yaw={samples[-1][2]:+.2f}')

    # Build anti-recoil pattern: anti_recoil = -kick (player moves opposite to gun)
    anti_recoil = [{'bullet': k['bullet'],
                    'anti_yaw':   -k['kick_yaw'],
                    'anti_pitch': -k['kick_pitch']} for k in kicks]

    output = {
        'weapon': args.weapon,
        'n_kicks': len(kicks),
        'kicks': kicks,                   # raw measured kicks (per-bullet view-punch jumps)
        'anti_recoil': anti_recoil,        # negated → ready for V_angle compensation
        'all_samples': samples,            # full time series for offline analysis
        'summary': {
            'total_pitch_deg': total_kick_pitch,
            'total_yaw_deg':   total_kick_yaw,
            'bullets_detected': len(kicks),
        },
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved to {out_path}')
    print('To use in spray_test: edit trajectory_generator.py to load this JSON,')
    print('OR pass --pattern-file to override hardcoded fbikat data.')


if __name__ == '__main__':
    main()
