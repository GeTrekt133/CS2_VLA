"""
Find m_pAimPunchServices offset by scanning during active spray.

m_pAimPunchServices points to a struct with m_predictableBaseAngle (Vec3) at +0x50.
This is what affects bullet trajectory (NOT view shake which is m_vecCsViewPunchAngle).

Usage:
    python find_aim_punch_offset.py
"""

from __future__ import annotations
import json
import time
from bot_pose_client import BotPoseClient


def main():
    with BotPoseClient(timeout=10.0) as c:
        print('[1/4] Scan IDLE state (no firing)...')
        r = c._send({'action': 'scan_aim_punch_offset'})
        idle_data = json.loads(r.message)
        idle_cands = {x['offset_in_pawn']: x['vec_at_0x50'] for x in idle_data['candidates']}
        print(f'    {len(idle_cands)} candidates')

        print('[2/4] Start firing for 0.5s to build aim_punch...')
        c._send({'action': 'start_attack'})
        time.sleep(0.5)

        print('[3/4] Scan ACTIVE state (mid-spray)...')
        r = c._send({'action': 'scan_aim_punch_offset'})
        active_data = json.loads(r.message)
        active_cands = {x['offset_in_pawn']: x['vec_at_0x50'] for x in active_data['candidates']}

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
                marker = '★ NEW'
                winners.append(off)
            elif tuple(idle_vec) != tuple(active_vec):
                marker = '★ CHANGED'
                winners.append(off)
            else:
                continue
            print(f'  {off:<10} | {str(idle_vec):<35} | {str(active_vec):<35}  {marker}')

        if not winners:
            print('\nNo offsets changed. Check that player was firing.')
            return

        print(f'\nLikely m_pAimPunchServices offset(s): {winners}')


if __name__ == '__main__':
    main()
