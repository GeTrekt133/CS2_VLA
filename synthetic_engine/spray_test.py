"""
Spray test — fire at whatever player is currently looking at (e.g. a wall).

Just stand in CS2, aim at a wall (or wherever you want to test), run this script.
Spray fires from current view direction with given recoil_scale applied.
Inspect bullet impact pattern on the wall to tune the scale.

Usage:
    python spray_test.py --recoil-scale 1.0
    python spray_test.py --recoil-scale 1.5
    python spray_test.py --weapon m4a1 --recoil-scale 1.2 --bullets 20
"""

from __future__ import annotations

import argparse
import json
import time

from bot_pose_client import BotPoseClient
from trajectory_generator import WEAPON_PATTERNS, get_recoil_pattern_deg


def load_extracted_pattern(path):
    """Load anti-recoil pattern from extract_recoil.py output. Returns cumulative
    list of (yaw_deg, pitch_deg) offsets indexed by bullets_fired.
    Supports both new ('anti_recoil') and old ('kicks') JSON formats."""
    with open(path) as f:
        data = json.load(f)
    cum_y = cum_p = 0.0
    out = [(0.0, 0.0)]
    if 'anti_recoil' in data:
        for ar in data['anti_recoil']:
            cum_y += ar['anti_yaw']
            cum_p += ar['anti_pitch']
            out.append((cum_y, cum_p))
    else:
        # Fallback: derive anti-recoil from raw kicks (just negate)
        for k in data.get('kicks', []):
            cum_y += -k['kick_yaw']
            cum_p += -k['kick_pitch']
            out.append((cum_y, cum_p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weapon', default='ak47',
                    choices=list(WEAPON_PATTERNS.keys()))
    ap.add_argument('--recoil-scale', type=float, default=1.0,
                    help='Uniform scale for both axes (overridden by --scale-x / --scale-y)')
    ap.add_argument('--scale-x', type=float, default=None,
                    help='Yaw (horizontal) compensation scale. Try 0, 0.5, 1.0, 1.5')
    ap.add_argument('--scale-y', type=float, default=None,
                    help='Pitch (vertical) compensation scale. Try 0.5, 1.0, 1.5, 2.0')
    ap.add_argument('--pattern-file', default=None,
                    help='JSON from extract_recoil.py — use as pattern instead of fbikat')
    ap.add_argument('--bullets', type=int, default=None,
                    help='Override number of bullets to fire (default: weapon mag size)')
    ap.add_argument('--tick-delay', type=float, default=0.05)
    ap.add_argument('--timescale', type=float, default=0.3)
    args = ap.parse_args()

    pat           = WEAPON_PATTERNS[args.weapon]
    fire_period   = pat['fire_period']
    n_bullets     = args.bullets or pat['max_bullets']
    if args.pattern_file:
        recoil_curve = load_extracted_pattern(args.pattern_file)
        print(f'[pattern] loaded from {args.pattern_file}: {len(recoil_curve)} entries')
    else:
        recoil_curve = get_recoil_pattern_deg(args.weapon)

    # Separate axis scales — fall back to uniform --recoil-scale if not given
    scale_x = args.scale_x if args.scale_x is not None else args.recoil_scale
    scale_y = args.scale_y if args.scale_y is not None else args.recoil_scale

    # Compute ticks needed for full spray
    game_per_tick = args.tick_delay * args.timescale
    total_ticks   = int((n_bullets * fire_period) / game_per_tick) + 5

    print(f'[plan] weapon={args.weapon}  bullets={n_bullets}  '
          f'fire_period={fire_period:.3f}s  total_ticks={total_ticks}')
    print(f'       tick_delay={args.tick_delay} timescale={args.timescale}  '
          f'scale_x={scale_x} scale_y={scale_y}')

    with BotPoseClient(timeout=10.0) as c:
        # 1. Read current view direction (where player is looking right now)
        r = c.get_geometry()
        if not r.ok:
            print(f'FAILED get_geometry: {r.message}')
            return
        geo = json.loads(r.message)
        # view_angles = [pitch, yaw, roll]
        target_pitch, target_yaw, _ = geo['viewer']['view_angles']
        print(f'[aim] using current view: yaw={target_yaw:.2f} pitch={target_pitch:.2f}')

        # 2. Slow time
        c.host_timescale(args.timescale)
        time.sleep(0.2)

        # 3. PURE SPRAY at current view direction
        elapsed_game  = 0.0
        next_fire_at  = 0.0
        bullets_fired = 0
        attack_held   = False

        try:
            for i in range(total_ticks):
                elapsed_game += args.tick_delay * args.timescale

                while elapsed_game >= next_fire_at and bullets_fired < n_bullets:
                    bullets_fired += 1
                    next_fire_at  += fire_period

                # Apply anti-recoil compensation to view
                if recoil_curve and bullets_fired > 0:
                    idx = min(bullets_fired, len(recoil_curve) - 1)
                    ar_yaw, ar_pitch = recoil_curve[idx]
                    view_yaw   = target_yaw   + ar_yaw   * scale_x
                    view_pitch = target_pitch + ar_pitch * scale_y
                else:
                    view_yaw, view_pitch = target_yaw, target_pitch

                c.set_player_view(view_yaw, view_pitch)

                # Transition-only: ONE +attack at start, ONE -attack when bullets done.
                # Spamming +attack each tick caused stuck input state.
                should_fire = bullets_fired < n_bullets
                if should_fire and not attack_held:
                    c._send({'action': 'start_attack'})
                    attack_held = True
                elif not should_fire and attack_held:
                    c._send({'action': 'stop_attack'})
                    attack_held = False

                time.sleep(args.tick_delay)
        finally:
            if attack_held:
                c._send({'action': 'stop_attack'})
            c.host_timescale(1.0)
            # Aggressive cleanup: release every button + reset timescale.
            # Prevents stuck input state (R/mouse not working after spray).
            c._send({'action': 'cleanup_inputs'})

        print(f'[done] fired {bullets_fired}/{n_bullets} bullets, scale={args.recoil_scale}')
        print('       check bullet impact pattern on the wall')


if __name__ == '__main__':
    main()
