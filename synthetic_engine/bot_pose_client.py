"""
BotPoseClient — Python client for the BotPoseControl CSSharp plugin.

Communicates with the plugin over TCP localhost:27040 using line-delimited JSON.

Example:
    client = BotPoseClient()
    client.connect()
    client.ping()
    client.spawn_bot(team='ct')
    client.set_poses([
        BotPose(slot=1, pos=[100.0, 200.0, 300.0], yaw=90.0, pitch=-5.0, freeze=True),
    ])
    client.close()
"""

from __future__ import annotations

import json
import math
import socket
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple


# CS2 demo Z is eye-level. Subtract these to get feet-level Z (entity origin).
EYE_OFFSET_STANDING = 64.0
EYE_OFFSET_CROUCHED = 40.0


def feet_z(eye_z: float, crouched: bool) -> float:
    return eye_z - (EYE_OFFSET_CROUCHED if crouched else EYE_OFFSET_STANDING)


@dataclass
class BotPose:
    slot: int
    pos: Optional[List[float]] = None       # [x, y, z]
    yaw: float = 0.0
    pitch: float = 0.0
    hp: Optional[int] = None
    armor: Optional[int] = None             # 0..100 (kevlar value); applied once on initial set
    helmet: Optional[bool] = None           # has helmet flag
    ducking: bool = False
    freeze: bool = True

    def to_dict(self) -> dict:
        d = {
            'slot': self.slot,
            'yaw': self.yaw,
            'pitch': self.pitch,
            'ducking': self.ducking,
            'freeze': self.freeze,
        }
        if self.pos is not None:
            d['pos'] = list(self.pos)
        if self.hp is not None:
            d['hp'] = self.hp
        if self.armor is not None:
            d['armor'] = self.armor
        if self.helmet is not None:
            d['helmet'] = self.helmet
        return d


@dataclass
class PoseResponse:
    ok: bool
    tick_id: int = 0
    applied: int = 0
    errors: List[str] = field(default_factory=list)
    message: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> 'PoseResponse':
        return cls(
            ok=bool(d.get('ok', False)),
            tick_id=int(d.get('tick_id', 0)),
            applied=int(d.get('applied', 0)),
            errors=list(d.get('errors') or []),
            message=d.get('message'),
        )


class BotPoseClient:
    def __init__(self, host: str = '127.0.0.1', port: int = 27040, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._reader = None  # type: ignore

    def connect(self) -> None:
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._sock.settimeout(self.timeout)
        # Wrap stream in a buffered reader for line reads.
        self._reader = self._sock.makefile('r', encoding='utf-8', newline='\n')

    def close(self) -> None:
        try:
            if self._reader is not None:
                self._reader.close()
        except Exception:
            pass
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        self._sock = None
        self._reader = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    # ---------------------------- Core RPC ---------------------------- #

    def _send(self, payload: dict) -> PoseResponse:
        if self._sock is None or self._reader is None:
            raise RuntimeError("Not connected — call connect() first")
        msg = (json.dumps(payload, separators=(',', ':')) + '\n').encode('utf-8')
        self._sock.sendall(msg)
        line = self._reader.readline()
        if not line:
            raise RuntimeError("Connection closed by plugin")
        return PoseResponse.from_dict(json.loads(line))

    # ---------------------------- High-level commands ---------------------------- #

    def ping(self) -> PoseResponse:
        return self._send({'action': 'ping'})

    def list_bots(self) -> PoseResponse:
        return self._send({'action': 'list_bots'})

    def spawn_bot(self, team: str = 'ct', count: int = 1) -> PoseResponse:
        return self._send({'action': 'spawn_bot', 'team': team, 'count': count})

    def kick_bots(self) -> PoseResponse:
        return self._send({'action': 'kick_bots'})

    def setup_match(self) -> PoseResponse:
        return self._send({'action': 'setup_match'})

    def prepare_match(self, n_ct: int = 0, n_t: int = 0) -> PoseResponse:
        """One-shot: kick all bots → apply settings → restart round → spawn bots."""
        return self._send({'action': 'prepare_match', 'n_ct': n_ct, 'n_t': n_t})

    def restart_round(self) -> PoseResponse:
        return self._send({'action': 'restart_round'})

    def spawn_planted_bomb(self, pos, yaw: float = 0.0) -> PoseResponse:
        return self._send({'action': 'spawn_planted_bomb',
                           'pos': list(pos), 'yaw': float(yaw)})

    def remove_planted_bombs(self) -> PoseResponse:
        return self._send({'action': 'remove_planted_bombs'})

    def get_planted_bombs(self) -> PoseResponse:
        return self._send({'action': 'get_planted_bombs'})

    def trace_visibility(self, eye, targets, tolerance: float = 30.0, skip_slots=None):
        """Engine-level batch line-of-sight check (CS2TraceRay).

        skip_slots: optional parallel list of bot slots to additionally skip per-target,
                    so a trace toward a bone passes through that bot's own collision capsule
                    and only stops at real walls / other entities in the way.

        Returns dict with keys: visible, fractions, contents, hit_ents, n_visible, n_total.
        Returns None on failure.
        """
        payload = {
            'action':    'trace_visibility_batch',
            'from':      [float(eye[0]), float(eye[1]), float(eye[2])],
            'targets':   [[float(t[0]), float(t[1]), float(t[2])] for t in targets],
            'tolerance': float(tolerance),
        }
        if skip_slots is not None:
            payload['skip_slots'] = [int(s) for s in skip_slots]
        r = self._send(payload)
        if not r.ok or not r.message:
            return None
        try:
            return json.loads(r.message)
        except json.JSONDecodeError:
            return None

    def unfreeze_all(self) -> PoseResponse:
        return self._send({'action': 'unfreeze_all'})

    def set_poses(self, poses: List[BotPose], tick_id: int = 0) -> PoseResponse:
        payload = {
            'action': 'set_poses',
            'tick_id': tick_id,
            'poses': [p.to_dict() for p in poses],
        }
        return self._send(payload)

    # ---------------------------- High-level snapshot reconstruction ---------------------------- #

    def _parse_bot_list(self, message: Optional[str]) -> List[Tuple[int, str]]:
        """Parse 'slot=N name=Bot Foo team=K, ...' (CS2 teams: 2=T, 3=CT)."""
        out: List[Tuple[int, str]] = []
        for chunk in (message or '').split(','):
            slot = team_num = None
            for tok in chunk.split():
                if tok.startswith('slot='):  slot     = int(tok.split('=', 1)[1])
                if tok.startswith('team='):  team_num = int(tok.split('=', 1)[1])
            if slot is not None and team_num is not None:
                out.append((slot, 'ct' if team_num == 3 else 't'))
        return out

    def get_module_info(self) -> PoseResponse:
        return self._send({'action': 'get_module_info'})

    def set_view_matrix_offset(self, offset: int) -> PoseResponse:
        return self._send({'action': 'set_view_matrix_offset', 'offset': offset})

    def get_view_matrix(self) -> PoseResponse:
        return self._send({'action': 'get_view_matrix'})

    def get_geometry(self) -> PoseResponse:
        return self._send({'action': 'get_geometry'})

    def set_player_view(self, yaw: float, pitch: float) -> PoseResponse:
        return self._send({'action': 'set_player_view', 'yaw': yaw, 'pitch': pitch})

    def host_timescale(self, scale: float) -> PoseResponse:
        return self._send({'action': 'host_timescale', 'scale': scale})

    def ensure_alive(self) -> PoseResponse:
        """Re-apply god, respawn if dead."""
        return self._send({'action': 'ensure_alive'})

    def cleanup_inputs(self) -> PoseResponse:
        """Release all buttons (-attack/-reload/-use/etc) + reset host_timescale to 1.
        Use after spray test if R/mouse stops working."""
        return self._send({'action': 'cleanup_inputs'})

    def reconstruct_snapshot(self, snapshot: List[dict]) -> PoseResponse:
        """End-to-end pipeline: prepare match → spawn bots → apply poses.

        snapshot = [
            {'team': 'ct'|'t', 'pos': [x, y, eye_z], 'yaw': float, 'pitch': float, 'crouched': bool},
            ...
        ]
        Z is converted from eye-level to feet-level automatically.
        """
        n_ct = sum(1 for p in snapshot if p['team'] == 'ct')
        n_t  = sum(1 for p in snapshot if p['team'] == 't')

        r = self.prepare_match(n_ct=n_ct, n_t=n_t)
        if not r.ok:
            return r
        time.sleep(0.5)

        r = self.list_bots()
        slot_team = self._parse_bot_list(r.message)
        ct_slots  = [s for s, t in slot_team if t == 'ct']
        t_slots   = [s for s, t in slot_team if t == 't']

        poses: List[BotPose] = []
        for slot, p in zip(ct_slots, [x for x in snapshot if x['team'] == 'ct']):
            poses.append(BotPose(
                slot=slot,
                pos=[p['pos'][0], p['pos'][1], feet_z(p['pos'][2], p['crouched'])],
                yaw=p['yaw'], pitch=p['pitch'], ducking=p['crouched'],
            ))
        for slot, p in zip(t_slots, [x for x in snapshot if x['team'] == 't']):
            poses.append(BotPose(
                slot=slot,
                pos=[p['pos'][0], p['pos'][1], feet_z(p['pos'][2], p['crouched'])],
                yaw=p['yaw'], pitch=p['pitch'], ducking=p['crouched'],
            ))

        return self.set_poses(poses)


def make_circle_snapshot(
    center: Tuple[float, float, float],
    n_ct: int,
    n_t: int,
    radius: float = 300.0,
    crouched: bool = False,
) -> List[dict]:
    """Bots in a circle around center, all facing the center. Z is eye-level."""
    cx, cy, cz = center
    total = n_ct + n_t
    snap = []
    for i in range(total):
        team  = 'ct' if i < n_ct else 't'
        angle = (2 * math.pi * i) / total
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        yaw = math.degrees(math.atan2(cy - y, cx - x))
        snap.append({'team': team, 'pos': [x, y, cz], 'yaw': yaw, 'pitch': 0.0, 'crouched': crouched})
    return snap


# ---------------------------- CLI for quick tests ---------------------------- #

def _cli():
    import argparse
    parser = argparse.ArgumentParser(description='BotPoseClient quick CLI')
    parser.add_argument('action', choices=[
        'ping', 'list', 'setup', 'restart', 'prepare',
        'spawn-ct', 'spawn-t', 'kick',
        'unfreeze', 'pose', 'circle',
    ])
    parser.add_argument('--slot', type=int, default=1)
    parser.add_argument('--pos', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help='Position [x y z]')
    parser.add_argument('--yaw', type=float, default=0.0)
    parser.add_argument('--pitch', type=float, default=0.0)
    parser.add_argument('--hp', type=int, default=None)
    parser.add_argument('--count', type=int, default=1, help='Number of bots to spawn')
    parser.add_argument('--n-ct', type=int, default=0, help='CT bots for prepare')
    parser.add_argument('--n-t', type=int, default=0, help='T bots for prepare')
    parser.add_argument('--no-freeze', action='store_true')
    parser.add_argument('--duck', action='store_true', help='Set FL_DUCKING (crouched stance, smaller hitbox)')
    # Circle test args
    parser.add_argument('--center', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help='Eye-level center for circle snapshot (use getpos value)')
    parser.add_argument('--radius', type=float, default=300.0)
    parser.add_argument('--crouch', action='store_true', help='All bots crouched in circle test')
    args = parser.parse_args()

    with BotPoseClient() as c:
        if args.action == 'ping':
            r = c.ping()
        elif args.action == 'list':
            r = c.list_bots()
        elif args.action == 'setup':
            r = c.setup_match()
        elif args.action == 'prepare':
            r = c.prepare_match(n_ct=args.n_ct, n_t=args.n_t)
        elif args.action == 'restart':
            r = c.restart_round()
        elif args.action == 'spawn-ct':
            r = c.spawn_bot('ct', count=args.count)
        elif args.action == 'spawn-t':
            r = c.spawn_bot('t', count=args.count)
        elif args.action == 'kick':
            r = c.kick_bots()
        elif args.action == 'unfreeze':
            r = c.unfreeze_all()
        elif args.action == 'pose':
            # --pos Z is eye-level (matching CS2 getpos / demo convention).
            # Convert to feet-level for the plugin: subtract 64 standing or 40 crouched.
            pos = list(args.pos) if args.pos else None
            if pos is not None:
                pos[2] = feet_z(pos[2], crouched=args.duck)
            pose = BotPose(
                slot=args.slot,
                pos=pos,
                yaw=args.yaw,
                pitch=args.pitch,
                hp=args.hp,
                ducking=args.duck,
                freeze=not args.no_freeze,
            )
            r = c.set_poses([pose])
        elif args.action == 'circle':
            if args.center is None:
                raise SystemExit('--center X Y Z is required (use getpos in CS2)')
            snapshot = make_circle_snapshot(
                center=tuple(args.center),
                n_ct=args.n_ct or 3,
                n_t=args.n_t or 3,
                radius=args.radius,
                crouched=args.crouch,
            )
            print(f'Circle snapshot: {len(snapshot)} bots around {tuple(args.center)} '
                  f'(radius={args.radius}, crouched={args.crouch})')
            r = c.reconstruct_snapshot(snapshot)
        else:
            raise ValueError(args.action)

    print(f"ok={r.ok} applied={r.applied} message={r.message!r} errors={r.errors}")


if __name__ == '__main__':
    _cli()
