"""
TrajectoryGenerator — realistic mouse trajectories for flicks, transfers, tracking.

Generates sequences of (yaw, pitch) angles that mimic human aiming:
  - Sigmoid flick: slow start -> fast middle -> slow correction
  - Bezier curves with overshoot and micro-adjustments
  - Tracking: smooth pursuit with lag
  - Hold: small random jitter around target

Each trajectory outputs exactly N ticks of (yaw, pitch) angles + keys pressed,
matching the format expected by DatasetIntent states.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from scenario_generator import Scenario


@dataclass
class TrajectoryPoint:
    """A single point in the trajectory."""
    yaw: float
    pitch: float
    keys: List[str]       # keys pressed at this tick


# ---------------------------- Real CS2 spray patterns ---------------------------- #
# Per-bullet INCREMENTAL anti-recoil deltas (dx_pixels, dy_pixels) at sens=2.52.
# +x = mouse moved RIGHT (= yaw decreases in CS2 convention)
# +y = mouse moved DOWN  (= pitch increases in CS2 convention)
# These ARE the corrections player should apply to compensate gun kick.
# Source: McDaived/NoRecoil-CS2 (verified open-source AHK script).
# Bullet 1 is implicit (0,0) — patterns start from bullet 2.
PIXEL_TO_DEG = 0.05544  # at sens=2.52, m_yaw=m_pitch=0.022

REAL_RECOIL_DELTAS = {
    'ak47': [
        (-4,7), (4,19), (-3,29), (-1,31), (13,31), (8,28), (13,21),
        (-17,12), (-42,-3), (-21,2), (12,11), (-15,7), (-26,-8), (-3,4),
        (40,1), (19,7), (14,10), (27,0), (33,-10), (-21,-2), (7,3),
        (-7,9), (-8,4), (19,-3), (5,6), (-20,-1), (-33,-4), (-45,-21), (-14,1),
    ],
    'm4a1': [   # M4A4 in scripts (game uses m4a1 as M4A4 internal name in CS2)
        (2,7), (0,9), (-6,16), (7,21), (-9,23), (-5,27), (16,15),
        (11,13), (22,5), (-4,11), (-18,6), (-30,-4), (-24,0), (-25,-6),
        (0,4), (8,4), (-11,1), (-13,-2), (2,2), (33,-1), (10,6),
        (27,3), (10,2), (11,0), (-12,0), (6,5), (4,5), (3,1), (4,-1),
    ],
    'm4a1_silencer': [
        (1,6), (0,4), (-4,14), (4,18), (-6,21), (-4,24), (14,14),
        (8,12), (18,5), (-4,10), (-14,5), (-25,-3), (-19,0), (-22,-3),
        (1,3), (8,3), (-9,1), (-13,-2), (3,2), (1,1),
    ],
    'famas': [
        (-4,5), (1,4), (-6,10), (-1,17), (0,20), (14,18), (16,12),
        (-6,12), (-20,8), (-16,5), (-13,2), (4,5), (23,4), (12,6),
        (20,-3), (5,0), (15,0), (3,5), (-4,3), (-25,-1), (-3,2),
        (11,0), (15,-7), (15,-10),
    ],
    'galilar': [
        (4,4), (-2,5), (6,10), (12,15), (-1,21), (2,24), (6,16),
        (11,10), (-4,14), (-22,8), (-30,-3), (-29,-13), (-9,8), (-12,2),
        (-7,1), (0,1), (4,7), (25,7), (14,4), (25,-3), (31,-9),
        (6,3), (-12,3), (13,-1), (10,-1), (16,-4), (-9,5), (-32,-5),
        (-24,-3), (-15,5), (6,8), (-14,-3), (-24,-14), (-13,-1),
    ],
    'aug': [
        (5,6), (0,13), (-5,22), (-7,26), (5,29), (9,30), (14,21),
        (6,15), (14,13), (-16,11), (-5,6), (13,0), (1,6), (-22,5),
        (-38,-11), (-31,-13), (-3,6), (-5,5), (-9,0), (24,1), (32,3),
        (15,6), (-5,1),
    ],
    'sg556': [
        (-4,9), (-13,15), (-9,25), (-6,29), (-8,31), (-7,36), (-20,14),
        (14,17), (-8,12), (-15,8), (-5,5), (6,5), (-8,6), (2,11),
        (-14,-6), (-20,-17), (-18,-9), (-8,-2), (41,3), (56,-5), (43,-1),
        (18,9), (14,9), (6,7), (21,-3), (29,-4), (-6,8), (-15,5), (-38,-5),
    ],
    'ump45': [
        (-1,6), (-4,8), (-2,18), (-4,23), (-9,23), (-3,26), (11,17),
        (-4,12), (9,13), (18,8), (15,5), (-1,3), (5,6), (0,6),
        (9,-3), (5,-1), (-12,4), (-19,1), (-1,-2), (15,-5), (17,-2),
        (-6,3), (-20,-2), (-3,-1),
    ],
}


def get_recoil_pattern_deg(weapon: str):
    """Return list of cumulative (yaw_delta, pitch_delta) in degrees for player's
    anti-recoil compensation. Bullet i (0-indexed): apply offsets[i] to current view.
    Bullet 0 is (0, 0) — first shot has no recoil."""
    deltas = REAL_RECOIL_DELTAS.get(weapon)
    if deltas is None:
        return None
    cum = [(0.0, 0.0)]
    cx = cy = 0.0
    for dx_px, dy_px in deltas:
        # CS2 conventions: +x pixel (mouse right) → yaw decreases; +y pixel (mouse down) → pitch increases.
        cx += -dx_px * PIXEL_TO_DEG  # negate because +pixel-x = move right = yaw down
        cy +=  dy_px * PIXEL_TO_DEG  # +pixel-y = mouse down = pitch up (CS2)
        cum.append((cx, cy))
    return cum


# ---------------------------- Weapon meta (fire rate / mag size / scope) ---------------------------- #
WEAPON_PATTERNS = {
    'ak47': {
        'fire_period': 0.10,    # seconds between bullets (~600 RPM)
        'max_bullets': 30,
        'kick_pitch': 1.6,      # base upward kick per bullet
        'kick_yaw_amp': 2.5,    # zigzag amplitude
        'kick_yaw_period': 12,  # bullets per zigzag cycle
        'recovery_ticks': 0,    # ticks after spray (no shoot, just hold position)
    },
    'm4a1': {
        'fire_period': 0.09,
        'max_bullets': 30,
        'kick_pitch': 1.2,
        'kick_yaw_amp': 1.8,
        'kick_yaw_period': 14,
        'recovery_ticks': 0,
    },
    'm4a1_silencer': {
        'fire_period': 0.09,
        'max_bullets': 20,
        'kick_pitch': 1.0,
        'kick_yaw_amp': 1.4,
        'kick_yaw_period': 14,
        'recovery_ticks': 0,
    },
    'galilar': {
        'fire_period': 0.11,
        'max_bullets': 35,
        'kick_pitch': 1.4,
        'kick_yaw_amp': 2.2,
        'kick_yaw_period': 11,
        'recovery_ticks': 0,
    },
    'famas': {
        'fire_period': 0.09,
        'max_bullets': 25,
        'kick_pitch': 1.0,
        'kick_yaw_amp': 1.5,
        'kick_yaw_period': 13,
        'recovery_ticks': 0,
    },
    'sg556': {
        'fire_period': 0.09,
        'max_bullets': 30,
        'kick_pitch': 1.3,
        'kick_yaw_amp': 2.0,
        'kick_yaw_period': 12,
        'recovery_ticks': 0,
    },
    'aug': {
        'fire_period': 0.09,
        'max_bullets': 30,
        'kick_pitch': 1.2,
        'kick_yaw_amp': 1.8,
        'kick_yaw_period': 13,
        'recovery_ticks': 0,
    },
    # SMGs — milder recoil, faster fire
    'mp9': {'fire_period': 0.06, 'max_bullets': 30, 'kick_pitch': 0.7, 'kick_yaw_amp': 1.0, 'kick_yaw_period': 16, 'recovery_ticks': 0},
    'mac10': {'fire_period': 0.06, 'max_bullets': 30, 'kick_pitch': 1.0, 'kick_yaw_amp': 2.0, 'kick_yaw_period': 10, 'recovery_ticks': 0},
    'ump45': {'fire_period': 0.09, 'max_bullets': 25, 'kick_pitch': 0.8, 'kick_yaw_amp': 1.0, 'kick_yaw_period': 14, 'recovery_ticks': 0},
    'mp7':   {'fire_period': 0.07, 'max_bullets': 30, 'kick_pitch': 0.7, 'kick_yaw_amp': 1.0, 'kick_yaw_period': 14, 'recovery_ticks': 0},
    'p90':   {'fire_period': 0.07, 'max_bullets': 50, 'kick_pitch': 0.6, 'kick_yaw_amp': 0.9, 'kick_yaw_period': 16, 'recovery_ticks': 0},
    # Single-shot weapons: pattern means tap-fire only (no spray)
    'awp':           {'fire_period': 1.5, 'max_bullets': 1, 'kick_pitch': 0, 'kick_yaw_amp': 0, 'kick_yaw_period': 1, 'scope': True},
    'ssg08':         {'fire_period': 1.0, 'max_bullets': 1, 'kick_pitch': 0, 'kick_yaw_amp': 0, 'kick_yaw_period': 1, 'scope': True},
    'scar20':        {'fire_period': 0.25, 'max_bullets': 3, 'kick_pitch': 0.5, 'kick_yaw_amp': 0.3, 'kick_yaw_period': 6, 'scope': True},
    'g3sg1':         {'fire_period': 0.25, 'max_bullets': 3, 'kick_pitch': 0.5, 'kick_yaw_amp': 0.3, 'kick_yaw_period': 6, 'scope': True},
    'deagle':        {'fire_period': 0.4, 'max_bullets': 2, 'kick_pitch': 2.0, 'kick_yaw_amp': 0.5, 'kick_yaw_period': 4},
    'glock':         {'fire_period': 0.15, 'max_bullets': 6, 'kick_pitch': 0.6, 'kick_yaw_amp': 0.8, 'kick_yaw_period': 8},
    'usp_silencer':  {'fire_period': 0.20, 'max_bullets': 4, 'kick_pitch': 0.5, 'kick_yaw_amp': 0.4, 'kick_yaw_period': 6},
    'hkp2000':       {'fire_period': 0.20, 'max_bullets': 4, 'kick_pitch': 0.5, 'kick_yaw_amp': 0.4, 'kick_yaw_period': 6},
    'p250':          {'fire_period': 0.18, 'max_bullets': 5, 'kick_pitch': 0.7, 'kick_yaw_amp': 0.5, 'kick_yaw_period': 6},
    'fiveseven':     {'fire_period': 0.18, 'max_bullets': 5, 'kick_pitch': 0.6, 'kick_yaw_amp': 0.4, 'kick_yaw_period': 6},
    'tec9':          {'fire_period': 0.15, 'max_bullets': 5, 'kick_pitch': 0.6, 'kick_yaw_amp': 0.6, 'kick_yaw_period': 6},
    'cz75a':         {'fire_period': 0.10, 'max_bullets': 8, 'kick_pitch': 0.7, 'kick_yaw_amp': 0.8, 'kick_yaw_period': 8},
    'revolver':      {'fire_period': 0.4, 'max_bullets': 2, 'kick_pitch': 1.5, 'kick_yaw_amp': 0.3, 'kick_yaw_period': 4},
}


class TrajectoryGenerator:
    """Generates realistic mouse movement trajectories."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def _clamp_pitch(self, p: float) -> float:
        return max(-89.0, min(89.0, p))

    def _normalize_yaw(self, y: float) -> float:
        while y > 180:
            y -= 360
        while y < -180:
            y += 360
        return y

    # ------------------------------------------------------------------ #
    # Snap (instant) flick helper                                          #
    # ------------------------------------------------------------------ #

    def _build_snap_trajectory(
        self,
        start_yaw: float, start_pitch: float,
        end_yaw: float, end_pitch: float,
        reaction_ticks: int, snap_ticks: int, correction_ticks: int,
        n_ticks: int,
    ) -> List[TrajectoryPoint]:
        """Instant snap: idle -> jump to target in 1-3 ticks -> micro-correct."""
        delta_yaw = self._normalize_yaw(end_yaw - start_yaw)
        delta_pitch = end_pitch - start_pitch

        idle_keys = self._random_movement_keys()
        points = []

        # Phase 1: idle
        for i in range(reaction_ticks):
            yaw = start_yaw + self.np_rng.normal(0, 0.05)
            pitch = self._clamp_pitch(start_pitch + self.np_rng.normal(0, 0.03))
            points.append(TrajectoryPoint(yaw=yaw, pitch=pitch, keys=list(idle_keys)))

        # Phase 2: snap (1-3 ticks to cover full delta, possibly with overshoot)
        overshoot = self.rng.uniform(1.0, 1.10)
        for i in range(snap_ticks):
            frac = (i + 1) / snap_ticks * overshoot
            yaw = start_yaw + delta_yaw * frac
            pitch = self._clamp_pitch(start_pitch + delta_pitch * frac)
            points.append(TrajectoryPoint(yaw=yaw, pitch=pitch, keys=list(idle_keys)))

        # Phase 3: micro-correction back to exact target + shoot
        shoot_start = max(0, correction_ticks - int(correction_ticks * self.rng.uniform(0.3, 0.6)))
        for i in range(correction_ticks):
            # Exponential decay toward target
            remaining = max(1, correction_ticks - i)
            blend = 1.0 - 0.5 ** (i / max(1, correction_ticks * 0.3))
            current_yaw = start_yaw + delta_yaw * overshoot
            yaw = current_yaw + (end_yaw - current_yaw) * blend
            pitch_ov = start_pitch + delta_pitch * overshoot
            pitch = pitch_ov + (end_pitch - pitch_ov) * blend

            # Tremor
            yaw += self.np_rng.normal(0, 0.1)
            pitch = self._clamp_pitch(pitch + self.np_rng.normal(0, 0.06))

            keys = list(idle_keys)
            if i >= shoot_start:
                keys.append("MOUSE_LEFT")
            points.append(TrajectoryPoint(yaw=yaw, pitch=pitch, keys=keys))

        # Pad or trim to exact n_ticks
        while len(points) < n_ticks:
            points.append(TrajectoryPoint(yaw=end_yaw, pitch=end_pitch,
                                          keys=list(idle_keys) + ["MOUSE_LEFT"]))
        points = points[:n_ticks]
        return points

    # ------------------------------------------------------------------ #
    # Sigmoid flick                                                        #
    # ------------------------------------------------------------------ #

    def generate_flick(
        self,
        scenario: Scenario,
        n_ticks: int = 64,
    ) -> List[TrajectoryPoint]:
        """Generate a sigmoid flick trajectory.

        Phases:
          1. Pre-flick idle (first ~10-20%): player hasn't reacted yet
          2. Acceleration (20-50%): fast ballistic mouse movement
          3. Deceleration (50-80%): slowing down near target
          4. Micro-correction (80-100%): small adjustments + shoot

        Includes random overshoot (1-5%) and correction.
        """
        start_yaw = scenario.initial_yaw
        start_pitch = scenario.initial_pitch
        end_yaw = scenario.target_yaw
        end_pitch = scenario.target_pitch

        # Flick speed style: slow (smooth), normal, fast (snap), instant
        style = self.rng.choices(
            ["slow", "normal", "fast", "instant"],
            weights=[0.15, 0.35, 0.30, 0.20],
        )[0]

        if style == "instant":
            # Instant snap: 2-5 tick reaction, then jump to target in 1-3 ticks
            reaction_ticks = self.rng.randint(2, max(3, n_ticks // 10))
            snap_ticks = self.rng.randint(1, 3)
            correction_ticks = n_ticks - reaction_ticks - snap_ticks
            return self._build_snap_trajectory(
                start_yaw, start_pitch, end_yaw, end_pitch,
                reaction_ticks, snap_ticks, correction_ticks, n_ticks,
            )

        # Reaction delay varies by style
        if style == "fast":
            reaction_frac = self.rng.uniform(0.05, 0.12)
        elif style == "slow":
            reaction_frac = self.rng.uniform(0.15, 0.30)
        else:
            reaction_frac = self.rng.uniform(0.08, 0.20)
        reaction_ticks = int(n_ticks * reaction_frac)

        # Overshoot: 1-8% past target (bigger for fast flicks)
        overshoot_frac = self.rng.uniform(1.01, 1.04 if style == "slow" else 1.08)

        # Build sigmoid curve for the active part
        active_ticks = n_ticks - reaction_ticks
        t = np.linspace(-6, 6, active_ticks)

        # Steepness varies widely by style
        if style == "fast":
            steepness = self.rng.uniform(1.5, 4.0)   # steep = fast flick
        elif style == "slow":
            steepness = self.rng.uniform(0.5, 0.9)
        else:
            steepness = self.rng.uniform(0.8, 2.0)
        sigma = 1.0 / (1.0 + np.exp(-t * steepness))

        # Apply overshoot in the 60-85% region, then correct back
        overshoot_start = int(active_ticks * 0.6)
        overshoot_peak = int(active_ticks * 0.75)
        overshoot_end = int(active_ticks * 0.85)

        progress = sigma.copy()
        for i in range(overshoot_start, min(overshoot_end, active_ticks)):
            # Smooth overshoot bump
            local_t = (i - overshoot_start) / max(1, overshoot_end - overshoot_start)
            bump = math.sin(local_t * math.pi) * (overshoot_frac - 1.0)
            progress[i] = min(progress[i] + bump, overshoot_frac)

        # Ensure ends at 1.0
        if active_ticks > 0:
            progress[-1] = 1.0
            # Smooth last few ticks to 1.0
            tail = min(5, active_ticks // 4)
            for i in range(active_ticks - tail, active_ticks):
                blend = (i - (active_ticks - tail)) / tail
                progress[i] = progress[i] * (1 - blend) + 1.0 * blend

        # Add micro-noise (human hand tremor)
        tremor_scale = self.rng.uniform(0.002, 0.008)
        noise_yaw = self.np_rng.normal(0, tremor_scale, active_ticks)
        noise_pitch = self.np_rng.normal(0, tremor_scale * 0.5, active_ticks)

        # Build trajectory
        points = []

        # Phase 1: reaction delay (idle at start position, maybe walking)
        idle_keys = self._random_movement_keys()
        for i in range(reaction_ticks):
            yaw = start_yaw + self.np_rng.normal(0, 0.05)
            pitch = self._clamp_pitch(start_pitch + self.np_rng.normal(0, 0.03))
            points.append(TrajectoryPoint(yaw=yaw, pitch=pitch, keys=list(idle_keys)))

        # Phase 2-4: flick + correction
        delta_yaw = self._normalize_yaw(end_yaw - start_yaw)
        delta_pitch = end_pitch - start_pitch

        for i in range(active_ticks):
            p = progress[i]
            yaw = start_yaw + delta_yaw * p + noise_yaw[i] * abs(delta_yaw)
            pitch = start_pitch + delta_pitch * p + noise_pitch[i] * abs(delta_pitch)
            pitch = self._clamp_pitch(pitch)

            # Keys: shoot near the end (last 15-25%)
            keys = list(idle_keys)
            shoot_start = int(active_ticks * self.rng.uniform(0.75, 0.88))
            if i >= shoot_start:
                keys.append("MOUSE_LEFT")

            points.append(TrajectoryPoint(yaw=yaw, pitch=pitch, keys=keys))

        return points

    # ------------------------------------------------------------------ #
    # Transfer                                                             #
    # ------------------------------------------------------------------ #

    def generate_transfer(
        self,
        scenario: Scenario,
        n_ticks: int = 64,
    ) -> List[TrajectoryPoint]:
        """Generate a transfer trajectory: shoot target 1, then flick to target 2.

        Split into two halves:
          - First half: small adjustment to target 1 + shoot
          - Second half: flick to target 2 + shoot
        """
        if len(scenario.bots) < 2:
            return self.generate_flick(scenario, n_ticks)

        eye_pos = scenario.eye_pos
        bot1_head = scenario.bots[0].head_pos
        bot2_head = scenario.bots[1].head_pos

        # Angles to each target
        yaw1 = math.degrees(math.atan2(bot1_head[1] - eye_pos[1], bot1_head[0] - eye_pos[0]))
        dist1 = math.sqrt(sum((a - b) ** 2 for a, b in zip(eye_pos[:2], bot1_head[:2])))
        pitch1 = math.degrees(math.atan2(-(bot1_head[2] - eye_pos[2]), dist1))

        yaw2 = math.degrees(math.atan2(bot2_head[1] - eye_pos[1], bot2_head[0] - eye_pos[0]))
        dist2 = math.sqrt(sum((a - b) ** 2 for a, b in zip(eye_pos[:2], bot2_head[:2])))
        pitch2 = math.degrees(math.atan2(-(bot2_head[2] - eye_pos[2]), dist2))

        # Split: 40-55% on first target, rest on second
        split = self.rng.uniform(0.40, 0.55)
        n1 = int(n_ticks * split)
        n2 = n_ticks - n1

        # Phase 1: aim at target 1 (small correction from initial + shoot)
        phase1_scenario = Scenario(
            scenario_type="hold",
            map_name=scenario.map_name,
            player_pos=scenario.player_pos,
            player_eye_height=scenario.player_eye_height,
            initial_yaw=scenario.initial_yaw,
            initial_pitch=scenario.initial_pitch,
            bots=[scenario.bots[0]],
            target_yaw=yaw1,
            target_pitch=pitch1,
            delta_yaw=0,
            delta_pitch=0,
        )
        points1 = self.generate_hold_with_shoot(phase1_scenario, n1)

        # Phase 2: flick from target 1 to target 2
        phase2_scenario = Scenario(
            scenario_type="flick",
            map_name=scenario.map_name,
            player_pos=scenario.player_pos,
            player_eye_height=scenario.player_eye_height,
            initial_yaw=yaw1,
            initial_pitch=pitch1,
            bots=[scenario.bots[1]],
            target_yaw=yaw2,
            target_pitch=pitch2,
            delta_yaw=self._normalize_yaw(yaw2 - yaw1),
            delta_pitch=pitch2 - pitch1,
        )
        points2 = self.generate_flick(phase2_scenario, n2)

        return points1 + points2

    # ------------------------------------------------------------------ #
    # Tracking                                                             #
    # ------------------------------------------------------------------ #

    def generate_tracking(
        self,
        scenario: Scenario,
        n_ticks: int = 64,
    ) -> List[TrajectoryPoint]:
        """Generate a tracking trajectory: follow a moving target.

        Bot moves linearly from bots[0] to bots[1]. Player tracks with lag.
        """
        if len(scenario.bots) < 2:
            return self.generate_flick(scenario, n_ticks)

        eye_pos = scenario.eye_pos
        start_pos = np.array(scenario.bots[0].head_pos)
        end_pos = np.array(scenario.bots[1].head_pos)

        # Bot movement speed variation
        speed_mult = self.rng.uniform(0.3, 0.7)  # don't traverse full distance

        # Tracking lag in ticks
        lag_ticks = self.rng.randint(2, 6)
        tracking_noise = self.rng.uniform(0.01, 0.04)

        points = []
        movement_keys = self._random_movement_keys()

        for i in range(n_ticks):
            # Bot position at this tick
            bot_t = (i / max(1, n_ticks - 1)) * speed_mult
            bot_pos = start_pos + (end_pos - start_pos) * bot_t

            # Player aims with lag (tracks where bot WAS a few ticks ago)
            lag_t = max(0, ((i - lag_ticks) / max(1, n_ticks - 1))) * speed_mult
            aim_pos = start_pos + (end_pos - start_pos) * lag_t

            # Calculate angle to lagged position
            dx = aim_pos[0] - eye_pos[0]
            dy = aim_pos[1] - eye_pos[1]
            dz = aim_pos[2] - eye_pos[2]
            yaw = math.degrees(math.atan2(dy, dx))
            dist_xy = math.sqrt(dx * dx + dy * dy)
            pitch = math.degrees(math.atan2(-dz, dist_xy))

            # Add tracking noise
            yaw += self.np_rng.normal(0, tracking_noise * 5)
            pitch += self.np_rng.normal(0, tracking_noise * 3)
            pitch = self._clamp_pitch(pitch)

            keys = list(movement_keys)
            # Burst fire while tracking
            if i > lag_ticks and self.rng.random() < 0.6:
                keys.append("MOUSE_LEFT")

            points.append(TrajectoryPoint(yaw=yaw, pitch=pitch, keys=keys))

        return points

    # ------------------------------------------------------------------ #
    # Hold (micro-adjustments)                                             #
    # ------------------------------------------------------------------ #

    def generate_hold(
        self,
        scenario: Scenario,
        n_ticks: int = 64,
    ) -> List[TrajectoryPoint]:
        """Generate a hold trajectory: small jitter around target position."""
        target_yaw = scenario.target_yaw
        target_pitch = scenario.target_pitch
        start_yaw = scenario.initial_yaw
        start_pitch = scenario.initial_pitch

        movement_keys = self._random_movement_keys()
        points = []

        # Slow drift to exact target position
        for i in range(n_ticks):
            t = i / max(1, n_ticks - 1)
            # Smooth interpolation to target
            blend = 1 - (1 - t) ** 2  # ease-in quadratic
            yaw = start_yaw + (target_yaw - start_yaw) * blend
            pitch = start_pitch + (target_pitch - start_pitch) * blend

            # Small jitter
            yaw += self.np_rng.normal(0, 0.1)
            pitch += self.np_rng.normal(0, 0.06)
            pitch = self._clamp_pitch(pitch)

            points.append(TrajectoryPoint(yaw=yaw, pitch=pitch, keys=list(movement_keys)))

        return points

    def generate_hold_with_shoot(
        self,
        scenario: Scenario,
        n_ticks: int = 32,
    ) -> List[TrajectoryPoint]:
        """Hold + shoot at the end."""
        points = self.generate_hold(scenario, n_ticks)
        # Start shooting in last 30-50%
        shoot_start = int(n_ticks * self.rng.uniform(0.5, 0.7))
        for i in range(shoot_start, n_ticks):
            if "MOUSE_LEFT" not in points[i].keys:
                points[i].keys.append("MOUSE_LEFT")
        return points

    # ------------------------------------------------------------------ #
    # Dispatch                                                             #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        scenario: Scenario,
        n_ticks: int = 64,
        tick_dt: float = 0.15,
        host_timescale: float = 0.05,
        recoil_scale: float = 1.0,
    ) -> List[TrajectoryPoint]:
        """Generate trajectory based on scenario type."""
        weapon = getattr(scenario, 'weapon', None)
        if weapon and weapon in WEAPON_PATTERNS:
            return self.generate_aim_and_spray(scenario, n_ticks, weapon,
                                                 tick_dt, host_timescale, recoil_scale)

        dispatch = {
            "flick": self.generate_flick,
            "transfer": self.generate_transfer,
            "tracking": self.generate_tracking,
            "hold": self.generate_hold,
        }
        fn = dispatch.get(scenario.scenario_type, self.generate_flick)
        return fn(scenario, n_ticks)

    def generate_aim_and_spray(
        self,
        scenario: Scenario,
        n_ticks: int,
        weapon: str,
        tick_dt: float = 0.15,         # real-time delay per Python tick
        host_timescale: float = 0.05,  # CS2 slow-mo factor → game time = real time × this
        recoil_scale: float = 1.0,     # multiplier on McDaived pattern; tune to match game's punch_angle
    ) -> List[TrajectoryPoint]:
        """Realistic kill scenario:
          1. Approach phase: idle/walk with movement keys (W/A/D/SHIFT)
          2. Aim phase: smooth s-curve flick to target (sigmoid easing, no overshoot for small flicks)
          3. Stop & shoot: counter-strafe (release W, brief S), then fire with weapon-specific recoil
        """
        pat = WEAPON_PATTERNS[weapon]
        delta_yaw   = self._normalize_yaw(scenario.target_yaw - scenario.initial_yaw)
        delta_pitch = scenario.target_pitch - scenario.initial_pitch

        # Tick budget: 25% approach, 35% flick, 40% shoot
        approach_ticks = max(2, int(n_ticks * 0.25))
        flick_ticks    = max(3, int(n_ticks * 0.35))
        shoot_ticks    = n_ticks - approach_ticks - flick_ticks

        movement_keys = self._random_movement_keys()
        points: List[TrajectoryPoint] = []

        # 1. Approach: small jitter at initial angle, with movement keys
        for i in range(approach_ticks):
            yaw   = scenario.initial_yaw   + self.np_rng.normal(0, 0.15)
            pitch = self._clamp_pitch(scenario.initial_pitch + self.np_rng.normal(0, 0.10))
            points.append(TrajectoryPoint(yaw=yaw, pitch=pitch, keys=list(movement_keys)))

        # 2. Flick: sigmoid easing toward target (smooth, no overshoot for small flicks)
        for i in range(flick_ticks):
            t = (i + 1) / flick_ticks
            # Smoothstep: 3t² - 2t³ — gentle ease-in-out
            ease = t * t * (3 - 2 * t)
            yaw   = scenario.initial_yaw   + delta_yaw   * ease
            pitch = self._clamp_pitch(scenario.initial_pitch + delta_pitch * ease)
            yaw   += self.np_rng.normal(0, 0.08)
            pitch  = self._clamp_pitch(pitch + self.np_rng.normal(0, 0.05))
            # Counter-strafe in last 2 ticks of flick (release W, brief S to stop)
            if i >= flick_ticks - 2:
                keys = []
                if 'W' in movement_keys:
                    keys.append('S')  # counter
            else:
                keys = list(movement_keys)
            points.append(TrajectoryPoint(yaw=yaw, pitch=pitch, keys=keys))

        # 3. Shoot phase: hold +attack, apply real per-bullet anti-recoil pattern.
        # Real patterns are time-based (~99ms between AK bullets), so we track GAME time
        # which is real time × host_timescale (game runs slower in slow-mo).
        max_bullets    = pat['max_bullets']
        fire_period    = pat['fire_period']  # game time between bullets (sec)
        scoped         = pat.get('scope', False)
        scope_held     = scoped
        recoil_curve   = get_recoil_pattern_deg(weapon) if not scoped else None

        bullets_fired = 0
        elapsed_game  = 0.0
        next_fire_at  = 0.0  # game time when next bullet should fire

        for i in range(shoot_ticks):
            elapsed_game += tick_dt * host_timescale  # advance game time
            keys = []
            if scope_held:
                keys.append('MOUSE_RIGHT')

            # Fire all bullets that should have fired by now
            while elapsed_game >= next_fire_at and bullets_fired < max_bullets:
                bullets_fired += 1
                next_fire_at += fire_period
                if scoped:
                    scope_held = False  # AWP unscopes after shot

            if bullets_fired > 0:
                keys.append('MOUSE_LEFT')

            # Apply anti-recoil pattern to V_angle. CS2 has TWO separate angle systems:
            #   1. V_angle  — what we set via set_player_view ("where player intends to look")
            #   2. punch_angle (m_aimPunchAngle) — game's mechanical recoil, accumulates per shot
            # Effective aim = V_angle + punch_angle (used for both view AND bullet trajectory).
            # We control V_angle but NOT punch_angle — so we must compensate by writing
            # V_angle = target + anti_recoil_offset to cancel punch_angle accumulation.
            # Correct indexing: after bullets_fired=N, punch_angle has accumulated
            # kicks from N bullets. To compensate, V_angle should = target + cum[N].
            # cum[0]=(0,0) compensates pre-shot state, cum[N] compensates after Nth shot.
            if recoil_curve:
                idx = min(bullets_fired, len(recoil_curve) - 1)
                ar_yaw, ar_pitch = recoil_curve[idx]
                cur_yaw   = scenario.target_yaw   + ar_yaw   * recoil_scale
                cur_pitch = scenario.target_pitch + ar_pitch * recoil_scale
            else:
                cur_yaw   = scenario.target_yaw
                cur_pitch = scenario.target_pitch

            yaw   = cur_yaw + self.np_rng.normal(0, 0.04)
            pitch = self._clamp_pitch(cur_pitch + self.np_rng.normal(0, 0.03))
            points.append(TrajectoryPoint(yaw=yaw, pitch=pitch, keys=keys))

        # Pad if rounding off
        while len(points) < n_ticks:
            last = points[-1]
            points.append(TrajectoryPoint(yaw=last.yaw, pitch=last.pitch, keys=list(last.keys)))

        return points[:n_ticks]

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _normalize_yaw(self, y: float) -> float:
        while y > 180:
            y -= 360
        while y < -180:
            y += 360
        return y

    def _random_movement_keys(self) -> set:
        """Random movement keys (W/A/D/SHIFT) for idle periods."""
        keys = set()
        r = self.rng.random()
        if r < 0.7:
            keys.add("W")
        if self.rng.random() < 0.15:
            keys.add("A")
        if self.rng.random() < 0.15:
            keys.add("D")
        if self.rng.random() < 0.5:
            keys.add("SHIFT")
        return keys
