"""
Quick test: fire 1 bullet at wall, check if impact is recorded.

Setup: spray_setup.py first, aim at wall.
"""
import json
import time
from bot_pose_client import BotPoseClient

with BotPoseClient(timeout=10.0) as c:
    c._send({'action': 'clear_impacts'})
    c._send({'action': 'start_impact_record'})

    print('Firing 3 bullets...')
    c._send({'action': 'start_attack'})
    time.sleep(0.4)
    c._send({'action': 'stop_attack'})
    time.sleep(0.3)

    c._send({'action': 'stop_impact_record'})
    r = c._send({'action': 'get_impacts'})
    data = json.loads(r.message)
    impacts = data['impacts']
    print(f'Recorded {len(impacts)} impacts:')
    for imp in impacts:
        print(f"  tick={imp['tick']}  pos=({imp['x']:.1f}, {imp['y']:.1f}, {imp['z']:.1f})")

    c._send({'action': 'cleanup_inputs'})
