"""
Find correct m_pCameraServices offset by scanning DURING active spray.

When firing: punch_angle has non-zero values. The right offset's target_ptr+0x48
will show changing pitch/yaw values. Other "candidates" stay at 0.

Usage:
    python find_punch_offset.py
    (assumes spray_setup.py already ran — AK in hand, infinite ammo)
"""

from __future__ import annotations

import json
import time

from bot_pose_client import BotPoseClient


def main():
    with BotPoseClient(timeout=10.0) as c:
        print('[1/4] Scan IDLE state (no firing)...')
        r = c._send({'action': 'scan_punch_offset'})
        idle_data = json.loads(r.message)
        idle_cands = {x['offset_in_pawn']: x['vec_at_0x48'] for x in idle_data['candidates']}
        print(f'    {len(idle_cands)} candidates')

        print('[2/4] Start firing for 0.5s to build punch_angle...')
        c._send({'action': 'start_attack'})
        time.sleep(0.5)  # 5+ bullets, punch accumulated

        print('[3/4] Scan ACTIVE state (mid-spray)...')
        r = c._send({'action': 'scan_punch_offset'})
        active_data = json.loads(r.message)
        active_cands = {x['offset_in_pawn']: x['vec_at_0x48'] for x in active_data['candidates']}

        c._send({'action': 'stop_attack'})
        c._send({'action': 'cleanup_inputs'})
        print(f'    {len(active_cands)} candidates')

        print('[4/4] Diff: candidates whose vec CHANGED:')
        print(f'  {"offset":<10} | {"idle vec":<35} | {"active vec":<35}')
        print(f'  {"-"*10}-+-{"-"*35}-+-{"-"*35}')
        winners = []
        for off, active_vec in active_cands.items():
            idle_vec = idle_cands.get(off)
            if idle_vec is None:
                # Only present in active scan = changed from null/garbage to valid
                marker = '★ NEW'
                winners.append(off)
            elif tuple(idle_vec) != tuple(active_vec):
                marker = '★ CHANGED'
                winners.append(off)
            else:
                continue  # unchanged → not what we want
            print(f'  {off:<10} | {str(idle_vec):<35} | {str(active_vec):<35}  {marker}')

        if not winners:
            print('\nNo offsets changed. Check: was player firing? Was setup correct?')
            return

        print(f'\nLikely m_pCameraServices offset(s): {winners}')
        print('Use the FIRST one in BotPoseControl.cs CAM_SERVICES_OFFSET.')


if __name__ == '__main__':
    main()
