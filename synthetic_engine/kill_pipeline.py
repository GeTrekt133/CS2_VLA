"""
KillPipeline — reconstruct kill scenarios in CS2 via plugin, recording
trajectory + frames + perfect bbox labels for NN training.

Replaces old cs2_data_engine flow:
  - Bot placement: BotPoseClient (plugin TCP, with bone access)
  - Player view:  set_player_view (plugin)
  - Aim target:   bone[7] (head bone) of bot, NOT demo's stored angle
  - Bbox labels:  computed via view-projection matrix from memory hook

Usage:
    python kill_pipeline.py \
        --kills C:/path/to/kill_scenarios.json \
        --output D:/SyntheticDataset \
        --view-matrix-offset 0x232E9C0 \
        --map de_mirage \
        --n-ticks 64 --max-scenarios 10

Output structure:
    output/scenario_NNNNNN/
        frames/tick_XXXX.jpg     (one per tick)
        metadata.json            (kill event + per-tick state + bbox)
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
from typing import List, Optional, Tuple

from PIL import ImageGrab

from bot_pose_client import (
    BotPose, BotPoseClient,
    EYE_OFFSET_STANDING, EYE_OFFSET_CROUCHED, feet_z,
)
from test_bbox_projection import find_cs2_window
from test_bbox_via_matrix import (
    project_aabb_via_matrix, world_to_screen_via_matrix,
    HALF_W, STAND_HEIGHT, DUCK_HEIGHT,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from trajectory_generator import TrajectoryGenerator  # noqa: E402
from scenario_generator import ScenarioGenerator      # noqa: E402

# For player teleport via console (setpos with noclip), we need cs2_cmd.
# This is the only thing we still go through the Win32 console for.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_collect_v2'))
try:
    from cs2_cmd import CS2Commander  # noqa: E402
except Exception:
    CS2Commander = None


# ---------------------------- Math helpers ---------------------------- #

# Spray weapons (full-auto with significant recoil) — aim at chest, recoil walks bullets up to head.
# Tap/single-shot weapons aim at head directly.
SPRAY_WEAPONS = {
    'ak47', 'm4a1', 'm4a1_silencer', 'famas', 'galilar', 'aug', 'sg556',
    'mp9', 'mac10', 'ump45', 'mp7', 'p90', 'bizon', 'negev', 'm249',
}


def sample_armor(rng: random.Random) -> Tuple[int, bool]:
    """Sample (armor_value, has_helmet) from realistic CS2 armor distribution.

    Approximates real round economics: most full-buy rounds have kevlar+helmet,
    pistol/eco rounds have less. Skews toward armored.
    """
    bucket = rng.choices(
        ['none', 'kevlar', 'kevlar_helmet'],
        weights=[18, 12, 70],   # 18% no armor, 12% kevlar only, 70% kevlar+helmet
        k=1,
    )[0]
    if bucket == 'none':
        return 0, False
    if bucket == 'kevlar':
        return 100, False
    return 100, True


def world_to_yaw_pitch(eye: Tuple[float, float, float],
                       target: Tuple[float, float, float]) -> Tuple[float, float]:
    """Compute yaw/pitch (CS2 conventions) for camera at `eye` looking at `target`."""
    dx = target[0] - eye[0]
    dy = target[1] - eye[1]
    dz = target[2] - eye[2]
    yaw   = math.degrees(math.atan2(dy, dx))
    dist  = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(-dz, dist))  # CS2: positive pitch = looking down
    return yaw, pitch


def normalize_yaw(y: float) -> float:
    while y >  180: y -= 360
    while y < -180: y += 360
    return y


def compute_head_bbox(head_bone, neck_bone, mat, w, h):
    """Asymmetric AABB: side=0.8d, down=d, up=1.2d. Same as test_bbox_via_matrix."""
    if head_bone is None or neck_bone is None:
        return None
    dx = head_bone[0] - neck_bone[0]
    dy = head_bone[1] - neck_bone[1]
    dz = head_bone[2] - neck_bone[2]
    d  = (dx * dx + dy * dy + dz * dz) ** 0.5
    if d < 0.5:
        return None
    cx, cy, cz = head_bone
    side = 0.8 * d
    pts = []
    for ex in (-side, side):
        for ey in (-side, side):
            for ez in (-d, 1.2 * d):
                sx, sy, dpt = world_to_screen_via_matrix((cx + ex, cy + ey, cz + ez), mat, w, h)
                if dpt > 0:
                    pts.append((sx, sy))
    if len(pts) < 4:
        return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def compute_body_bbox(origin, ducking, mat, w, h):
    """Body AABB from origin + standard CS2 player extents."""
    bot_h = DUCK_HEIGHT if ducking else STAND_HEIGHT
    center = (origin[0], origin[1], origin[2] + bot_h / 2)
    return project_aabb_via_matrix(center, (HALF_W, HALF_W, bot_h / 2), mat, w, h)


# ---------------------------- Pipeline ---------------------------- #

class KillPipeline:
    def __init__(self, output_dir: str, view_matrix_offset: int,
                 capture_size: Optional[Tuple[int, int]] = None,
                 tick_delay: float = 0.15, host_timescale: float = 0.05,
                 recoil_scale: float = 1.0,
                 seed: int = 42):
        self.output_dir         = output_dir
        self.view_matrix_offset = view_matrix_offset
        self.tick_delay         = tick_delay
        self.host_timescale     = host_timescale
        self.recoil_scale       = recoil_scale
        self.capture_size       = capture_size
        self.client             = BotPoseClient(timeout=15.0)
        self.cmd                = CS2Commander() if CS2Commander else None
        self.rng                = random.Random(seed)

        os.makedirs(output_dir, exist_ok=True)

    def setup(self, map_name: str, setup_delay: float = 10.0):
        self.client.connect()
        r = self.client.set_view_matrix_offset(self.view_matrix_offset)
        if not r.ok:
            raise RuntimeError(f"set_view_matrix_offset failed: {r.message}")
        # Cache CS2 window region once.
        self.cs2_bbox = find_cs2_window()
        if self.cs2_bbox is None:
            raise RuntimeError("CS2 window not found by title")
        L, T, R, B = self.cs2_bbox
        self.win_w, self.win_h = R - L, B - T

        if self.cmd:
            self.cmd.connect()
            self.cmd.send(f"map {map_name}")
            time.sleep(15)

        # Configure match: 1 CT + 1 T bots so we have one of each available
        # for whichever team the victim was on.
        # mp_restartgame inside prepare_match should respawn player on their
        # current team — no jointeam needed if already on a team.
        r = self.client.prepare_match(n_ct=1, n_t=1)
        if not r.ok:
            raise RuntimeError(f"prepare_match failed: {r.message}")
        time.sleep(setup_delay)

    def _list_bots_by_team(self) -> dict:
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

    def _teleport_player(self, x: float, y: float, z: float):
        """Teleport via console. Demoparser2 returns player entity origin (feet level)
        directly — pass through to setpos as-is, with small upward bias to avoid
        clipping into the floor."""
        if self.cmd is None:
            raise RuntimeError("CS2Commander unavailable; cannot teleport player")
        self.cmd.send_batch([
            "god 1",
            "noclip 1",
            f"setpos {x:.2f} {y:.2f} {z + 10:.2f}",
        ])
        time.sleep(0.5)
        self.cmd.send("noclip 0")
        time.sleep(0.5)
        self.cmd.send("god 1")

    def _capture(self):
        return ImageGrab.grab(bbox=self.cs2_bbox, all_screens=True).convert('RGB')

    def run_scenario(self, scenario, idx: int, traj_gen: TrajectoryGenerator,
                     n_ticks: int) -> Optional[dict]:
        scenario_id = f"scenario_{idx:06d}"
        scen_dir    = os.path.join(self.output_dir, scenario_id)
        frames_dir  = os.path.join(scen_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)

        # 0. Ensure player is alive + has god mode (re-apply each scenario)
        r = self.client.ensure_alive()
        if not r.ok:
            print(f"[{scenario_id}] ensure_alive failed: {r.message}")
            return None
        time.sleep(1.0)

        # Determine player's team via plugin to avoid friendly fire — pick bot from opposite team
        viewer_team = 'ct'
        try:
            geo_r = self.client.get_geometry()
            if geo_r.ok and geo_r.message:
                viewer_team = json.loads(geo_r.message).get('viewer', {}).get('team', 'ct')
        except Exception:
            pass
        opposite = 't' if viewer_team == 'ct' else 'ct'

        bots = self._list_bots_by_team()
        bot_slot = (bots[opposite] + [None])[0]
        if bot_slot is None:
            print(f"[{scenario_id}] no bot on opposite team ({opposite}); player_team={viewer_team}")
            return None

        # 1. Place bot at victim position. Demo Z is feet-level (entity origin) —
        # use as-is, plugin Teleport expects feet-level coords too.
        victim = scenario.bots[0]
        v_pos = victim.pos
        ducking = victim.ducking

        # Random armor: realistic distribution (most bots full kevlar+helmet)
        armor, helmet = sample_armor(self.rng)

        bot_pose = BotPose(
            slot=bot_slot,
            pos=[v_pos[0], v_pos[1], v_pos[2]],
            yaw=victim.victim_yaw,
            pitch=victim.victim_pitch,
            armor=armor,
            helmet=helmet,
            ducking=ducking,
            freeze=True,
        )
        r = self.client.set_poses([bot_pose])
        if not r.ok:
            print(f"[{scenario_id}] set_poses failed: {r.message}")
            return None

        # 2. Teleport player to attacker pos + initial view
        a_pos = scenario.player_pos
        self._teleport_player(*a_pos)
        self.client.set_player_view(yaw=scenario.initial_yaw, pitch=scenario.initial_pitch)

        # Give player the weapon used in this kill (for spray pattern + visual)
        weapon = getattr(scenario, 'weapon', '')
        if weapon and self.cmd:
            self.cmd.send(f"give weapon_{weapon}")
            time.sleep(0.2)
            # Switch to primary slot (slot1=primary, slot2=secondary, slot3=knife)
            slot = '2' if weapon in ('glock', 'usp_silencer', 'hkp2000', 'p250',
                                      'fiveseven', 'tec9', 'cz75a', 'deagle',
                                      'revolver') else '1'
            self.cmd.send(f"slot{slot}")
            time.sleep(0.2)

        time.sleep(1.0)

        # 3. Get bot bone positions to compute REAL target angles.
        # For spray weapons → aim CHEST (recoil walks bullets up through body to head).
        # For tap/single-shot weapons → aim HEAD directly.
        r = self.client.get_geometry()
        if not r.ok:
            print(f"[{scenario_id}] get_geometry failed: {r.message}")
            return None
        geo = json.loads(r.message)
        bot_data = next((b for b in geo['bots'] if b['slot'] == bot_slot), None)
        if not bot_data or bot_data.get('head_bone') is None:
            print(f"[{scenario_id}] head bone not available")
            return None

        weapon = getattr(scenario, 'weapon', '') or ''
        is_spray = weapon in SPRAY_WEAPONS

        viewer_eye = tuple(geo['viewer']['eye'])
        if is_spray and bot_data.get('chest_bone') is not None:
            target_world = tuple(bot_data['chest_bone'])
            aim_label = 'chest'
        else:
            target_world = tuple(bot_data['head_bone'])
            aim_label = 'head'
        target_yaw, target_pitch = world_to_yaw_pitch(viewer_eye, target_world)
        print(f"[{scenario_id}] weapon={weapon} aim={aim_label}")

        # 4. Generate trajectory: initial → exact head position
        scenario.target_yaw     = target_yaw
        scenario.target_pitch   = target_pitch
        scenario.delta_yaw      = normalize_yaw(target_yaw - scenario.initial_yaw)
        scenario.delta_pitch    = target_pitch - scenario.initial_pitch
        trajectory = traj_gen.generate(
            scenario, n_ticks=n_ticks,
            tick_dt=self.tick_delay, host_timescale=self.host_timescale,
            recoil_scale=self.recoil_scale,
        )

        # 5. Slow time + capture frames
        self.client.host_timescale(self.host_timescale)
        time.sleep(0.3)

        ticks = []
        attack_held = False
        for i, point in enumerate(trajectory):
            self.client.set_player_view(yaw=point.yaw, pitch=point.pitch)

            # Real fire: re-send +attack each tick while wanting to fire (some game
            # versions auto-release server-side input). Only send -attack on transition.
            wants_fire = 'MOUSE_LEFT' in point.keys
            if wants_fire:
                self.client._send({'action': 'start_attack'})
                attack_held = True
            elif attack_held:
                self.client._send({'action': 'stop_attack'})
                attack_held = False

            time.sleep(self.tick_delay)

            img = self._capture()
            img.save(os.path.join(frames_dir, f"tick_{i:04d}.jpg"), quality=85)

            # Re-read geometry + matrix at exact frame moment
            geo_r = self.client.get_geometry()
            mat_r = self.client.get_view_matrix()
            if not geo_r.ok or not mat_r.ok:
                continue
            geo  = json.loads(geo_r.message)
            mat  = json.loads(mat_r.message)['matrix']

            tick_bots = []
            for b in geo['bots']:
                if not b['alive']:
                    continue
                body_bbox = compute_body_bbox(b['origin'], b['ducking'], mat, self.win_w, self.win_h)
                head_bbox = compute_head_bbox(b.get('head_bone'), b.get('neck_bone'), mat, self.win_w, self.win_h)
                tick_bots.append({
                    'slot':       b['slot'],
                    'team':       b['team'],
                    'ducking':    b['ducking'],
                    'origin':     b['origin'],
                    'eye':        b['eye'],
                    'head_bone':  b.get('head_bone'),
                    'neck_bone':  b.get('neck_bone'),
                    'body_bbox':  body_bbox,
                    'head_bbox':  head_bbox,
                })

            ticks.append({
                'tick':        i,
                'yaw':         point.yaw,
                'pitch':       point.pitch,
                'keys':        list(point.keys),
                'viewer_eye':  geo['viewer']['eye'],
                'view_angles': geo['viewer']['view_angles'],
                'view_matrix': mat,
                'bots':        tick_bots,
            })

        # 6. Stop firing + restore time
        if attack_held:
            self.client._send({'action': 'stop_attack'})
        self.client.host_timescale(1.0)
        time.sleep(0.3)

        # 7. Restart round so bots respawn (current bot is dead) and we get a fresh
        # board for the next scenario. ensure_alive in next scenario will re-apply god.
        self.client.restart_round()
        time.sleep(2.5)  # let restart settle: player respawns, bots back at spawns

        # 7. Save metadata
        meta = {
            'scenario_id': scenario_id,
            'n_ticks':     len(ticks),
            'window':      [self.win_w, self.win_h],
            'kill_event': {
                'attacker_pos':   list(scenario.player_pos),
                'victim_pos':     list(victim.pos),
                'initial_yaw':    scenario.initial_yaw,
                'initial_pitch':  scenario.initial_pitch,
                'demo_target_yaw':   victim.victim_yaw,   # original demo target (not used as actual aim)
                'demo_target_pitch': victim.victim_pitch,
                'aim_target_yaw':    target_yaw,           # actual aim (head bone)
                'aim_target_pitch':  target_pitch,
                'victim_ducking':    ducking,
                'victim_armor':      armor,
                'victim_helmet':     helmet,
            },
            'ticks': ticks,
        }
        with open(os.path.join(scen_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, separators=(',', ':'))

        print(f"[{scenario_id}] saved {len(ticks)} ticks to {scen_dir}")
        return meta

    def close(self):
        try: self.client.host_timescale(1.0)
        except Exception: pass
        self.client.close()


# ---------------------------- CLI ---------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kills',  required=True, help='Path to kill_scenarios.json')
    ap.add_argument('--output', required=True, help='Output directory for scenarios')
    ap.add_argument('--map',    default='de_mirage')
    ap.add_argument('--view-matrix-offset', type=lambda s: int(s, 0),
                    default=0x232E9C0,
                    help='dwViewMatrix offset in client.dll')
    ap.add_argument('--n-ticks', type=int, default=64)
    ap.add_argument('--max-scenarios', type=int, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tick-delay', type=float, default=0.15)
    ap.add_argument('--timescale', type=float, default=0.3,
                    help='host_timescale during scenario. Lower = slower mo (more frames per game-tick) '
                         'but fewer bullets fire per scenario. 0.3 ≈ ~13 bullets in standard 30-tick spray.')
    ap.add_argument('--setup-delay', type=float, default=10.0,
                    help='Wait after prepare_match for player respawn + bots to spawn')
    ap.add_argument('--force-weapon', type=str, default=None,
                    help='Override weapon for all scenarios (e.g. "ak47"). Useful for '
                         'targeted spray data collection.')
    ap.add_argument('--random', action='store_true',
                    help='Use a fresh random seed each run (different scenarios sampled '
                         'every time). Overrides --seed.')
    ap.add_argument('--recoil-scale', type=float, default=1.0,
                    help='Multiplier for anti-recoil pattern. Tune empirically: '
                         '<1.0 = under-compensate (bullets drift up), >1.0 = over-compensate '
                         '(bullets drift down). Try 0.5..2.0 to find best fit.')
    args = ap.parse_args()

    if args.random:
        args.seed = random.SystemRandom().randint(0, 2**31 - 1)
        print(f'[--random] using seed={args.seed}')

    pipeline = KillPipeline(
        output_dir=args.output,
        view_matrix_offset=args.view_matrix_offset,
        tick_delay=args.tick_delay,
        host_timescale=args.timescale,
        recoil_scale=args.recoil_scale,
        seed=args.seed,
    )
    pipeline.setup(map_name=args.map, setup_delay=args.setup_delay)

    # Generate scenarios from existing kill_scenarios.json
    gen = ScenarioGenerator(map_name=args.map, seed=args.seed)
    traj_gen = TrajectoryGenerator(seed=args.seed)
    scenarios = gen.generate_from_kills(args.kills)
    if args.force_weapon:
        for sc in scenarios:
            sc.weapon = args.force_weapon
        print(f"Forced all scenarios to weapon={args.force_weapon}")
    # Extra shuffle to fully randomize selection order even with same kills file.
    random.Random(args.seed).shuffle(scenarios)
    if args.max_scenarios:
        scenarios = scenarios[:args.max_scenarios]
    print(f"Loaded {len(scenarios)} kill scenarios")

    try:
        for i, sc in enumerate(scenarios):
            try:
                pipeline.run_scenario(sc, i, traj_gen, args.n_ticks)
            except Exception as ex:
                print(f"  scenario {i} ERROR: {ex}")
    finally:
        pipeline.close()


if __name__ == '__main__':
    main()
