"""
Diagnostic: which CS2TraceRay signatures find a non-zero pointer in current server.dll?
Tells us which patterns need updating for our pinned CS2 build.
"""

import json
from bot_pose_client import BotPoseClient


def main():
    with BotPoseClient(timeout=15.0) as c:
        r = c._send({'action': 'trace_diagnostic'})
        if not r.ok:
            print('FAILED:', r.message); return

        data = json.loads(r.message)
        print(f'Server path: {data.get("server_path", "?")}\n')

        for name in ['GameTraceManager', 'TraceFunc', 'TraceShape', 'CTraceFilterVtable']:
            entry = data.get(name, {})
            if 'error' in entry:
                print(f'  {name:24s} ERROR: {entry["error"]}')
            else:
                ptr   = entry.get('ptr', 0)
                found = entry.get('found', False)
                sig   = entry.get('signature', '')
                tag   = 'FOUND' if found else 'NOT FOUND'
                print(f'  {name:24s} {tag:9s}  ptr=0x{ptr:016x}')
                print(f'  {"":24s} sig={sig[:80]}...' if len(sig) > 80 else f'  {"":24s} sig={sig}')

        print('\nMissing signatures need patterns updated for current CS2 build.')


if __name__ == '__main__':
    main()
