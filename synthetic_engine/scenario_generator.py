"""
ScenarioGenerator — generates random player/bot positions on CS2 maps.

Each scenario defines:
  - Player position + initial viewing angle
  - One or more bot positions (with head height)
  - Scenario type: flick, transfer, tracking, hold
  - Required mouse delta to reach each target

Uses predefined navmesh-like spawn points per map (hand-picked positions
where players can actually stand without clipping through walls).
"""

import json
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class BotTarget:
    """A bot placed in the world."""
    pos: Tuple[float, float, float]       # (x, y, z) feet position
    head_offset: float = 62.0             # head height above Z
    body_offset: float = 36.0             # body center above Z
    label: str = ""                       # optional label
    victim_yaw: float = 0.0              # bot facing direction
    victim_pitch: float = 0.0
    ducking: bool = False                 # bot is crouching

    @property
    def head_pos(self) -> Tuple[float, float, float]:
        return (self.pos[0], self.pos[1], self.pos[2] + self.head_offset)

    @property
    def body_pos(self) -> Tuple[float, float, float]:
        return (self.pos[0], self.pos[1], self.pos[2] + self.body_offset)


@dataclass
class Scenario:
    """A single training scenario."""
    scenario_type: str                          # "flick", "transfer", "tracking", "hold"
    map_name: str
    player_pos: Tuple[float, float, float]      # player feet position
    player_eye_height: float                    # 64.06 standing, 46.05 crouching
    initial_yaw: float                          # where player looks initially (degrees)
    initial_pitch: float
    bots: List[BotTarget]
    # Computed
    target_yaw: float = 0.0                     # ideal yaw to first bot head
    target_pitch: float = 0.0
    delta_yaw: float = 0.0                      # required flick delta
    delta_pitch: float = 0.0
    weapon: str = ''                            # weapon name from demo (e.g. 'ak47', 'awp')

    @property
    def eye_pos(self) -> Tuple[float, float, float]:
        return (self.player_pos[0], self.player_pos[1],
                self.player_pos[2] + self.player_eye_height)


# ============================================================================
# Map spawn points — hand-picked positions where players can stand
# Format: (x, y, z) — Z is feet level
# ============================================================================

# These are approximate playable positions on common maps.
# Each position has been selected to have clear sightlines to nearby positions.
# Positions extracted from real Faceit demo files on de_mirage.
# XY quantized to grid=300, Z = median from actual player feet positions.
MAP_POSITIONS = {
    "de_mirage": {
        "name": "Mirage",
        "positions": [
            # --- B site / B apps ---
            (-2100, 600, -129.8),   # B apartments
            (-2100, 900, -83.0),    # B apartments balcony
            (-1800, -600, -168.0),  # B short / market
            (-1200, 300, -165.6),   # B short corner
            (-1200, 600, -80.0),    # B apps window
            # --- CT spawn ---
            (-1800, -2100, -267.5), # CT spawn main
            (-1800, -1800, -267.2), # CT spawn back
            (-1200, -2400, -176.9), # CT connector
            (-900, -2400, -168.0),  # CT toward A
            # --- Kitchen / connector ---
            (-1500, -1200, -257.0), # kitchen door
            (-1500, -900, -168.0),  # kitchen
            (-1200, -1200, -168.0), # connector mid
            (-1200, -900, -168.0),  # underpass exit
            (-1200, -600, -168.0),  # window room
            # --- A site ---
            (-900, -1500, -168.0),  # A stairs
            (-900, -1200, -168.0),  # A tetris
            (-900, -600, -264.0),   # jungle
            (-900, -300, -167.2),   # connector A side
            (-900, 0, -168.0),      # A ramp top
            (-900, 300, -172.2),    # A ramp
            (-900, 600, -80.0),     # palace balcony
            # --- Mid ---
            (-600, -2100, -180.0),  # mid CT side
            (-600, -1500, -81.1),   # catwalk (elevated)
            (-600, -1200, -168.0),  # mid window
            (-600, -900, -227.1),   # mid boxes
            # --- A main / palace ---
            (-300, -2100, -171.9),  # A main
            (-300, 600, -80.0),     # palace exit
            (0, -2100, -101.6),     # A site high
            (0, -1800, -168.0),     # palace entrance
            (0, -1500, -170.6),     # palace hall
            # --- Top mid / T side ---
            (300, -1500, -176.0),   # top mid T
            (300, -300, -165.8),    # mid T awp
            (300, 0, -183.1),       # top mid
            (300, 300, -258.3),     # T ramp
            (600, -1500, -262.5),   # T mid push
            (600, 600, -136.0),     # T apps
            # --- T spawn ---
            (900, 600, -253.3),     # T spawn side
            (1200, -300, -164.8),   # T spawn main
            (1200, 0, -164.5),      # T spawn center
            (1200, 300, -230.0),    # T spawn back
        ],
    },
}

DEFAULT_MAP = "de_mirage"


class ScenarioGenerator:
    """Generates random scenarios for synthetic data collection."""

    def __init__(
        self,
        map_name: str = DEFAULT_MAP,
        seed: Optional[int] = None,
    ):
        self.map_name = map_name
        if map_name not in MAP_POSITIONS:
            raise ValueError(f"Unknown map: {map_name}. Available: {list(MAP_POSITIONS.keys())}")
        self.positions = MAP_POSITIONS[map_name]["positions"]
        self.rng = random.Random(seed)

    def _calc_angle(
        self,
        eye_pos: Tuple[float, float, float],
        target_pos: Tuple[float, float, float],
    ) -> Tuple[float, float]:
        """Calculate yaw/pitch from eye to target (CS2 convention).

        CS2: yaw 0=+X (East), 90=+Y (North), CCW. pitch negative=up.
        """
        dx = target_pos[0] - eye_pos[0]
        dy = target_pos[1] - eye_pos[1]
        dz = target_pos[2] - eye_pos[2]
        yaw = math.degrees(math.atan2(dy, dx))
        dist_xy = math.sqrt(dx * dx + dy * dy)
        pitch = math.degrees(math.atan2(-dz, dist_xy))
        return yaw, pitch

    def _normalize_yaw_delta(self, delta: float) -> float:
        """Normalize yaw delta to [-180, 180]."""
        while delta > 180:
            delta -= 360
        while delta < -180:
            delta += 360
        return delta

    def _pick_positions(self, n: int, min_dist: float = 200.0) -> List[Tuple[float, float, float]]:
        """Pick n positions that are at least min_dist apart."""
        available = list(self.positions)
        self.rng.shuffle(available)
        picked = []
        for pos in available:
            if len(picked) >= n:
                break
            too_close = False
            for p in picked:
                d = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos, p)))
                if d < min_dist:
                    too_close = True
                    break
            if not too_close:
                picked.append(pos)
        return picked

    def generate_flick(
        self,
        min_delta_deg: float = 30.0,
        max_delta_deg: float = 170.0,
    ) -> Optional[Scenario]:
        """Generate a flick scenario: player looks away, must flick to enemy.

        Returns None if no valid positions found.
        """
        positions = self._pick_positions(2, min_dist=300.0)
        if len(positions) < 2:
            return None

        player_pos = positions[0]
        bot_pos = positions[1]
        eye_height = 64.06
        eye_pos = (player_pos[0], player_pos[1], player_pos[2] + eye_height)

        bot = BotTarget(pos=bot_pos, label="flick_target")
        target_yaw, target_pitch = self._calc_angle(eye_pos, bot.head_pos)

        # Generate initial angle offset (the flick amount)
        delta_mag = self.rng.uniform(min_delta_deg, max_delta_deg)
        delta_sign = self.rng.choice([-1, 1])
        delta_yaw = delta_mag * delta_sign

        # Small pitch offset
        delta_pitch = self.rng.uniform(-10, 10)

        initial_yaw = target_yaw - delta_yaw
        initial_pitch = target_pitch - delta_pitch

        # Clamp pitch
        initial_pitch = max(-89, min(89, initial_pitch))

        return Scenario(
            scenario_type="flick",
            map_name=self.map_name,
            player_pos=player_pos,
            player_eye_height=eye_height,
            initial_yaw=initial_yaw,
            initial_pitch=initial_pitch,
            bots=[bot],
            target_yaw=target_yaw,
            target_pitch=target_pitch,
            delta_yaw=self._normalize_yaw_delta(delta_yaw),
            delta_pitch=delta_pitch,
        )

    def generate_transfer(self, n_targets: int = 2) -> Optional[Scenario]:
        """Generate a transfer scenario: flick between multiple targets.

        Player starts aimed at first target, must transfer to subsequent targets.
        """
        positions = self._pick_positions(n_targets + 1, min_dist=250.0)
        if len(positions) < n_targets + 1:
            return None

        player_pos = positions[0]
        eye_height = 64.06
        eye_pos = (player_pos[0], player_pos[1], player_pos[2] + eye_height)

        bots = []
        for i in range(1, len(positions)):
            bots.append(BotTarget(pos=positions[i], label=f"target_{i}"))

        # Start aimed at first target
        target_yaw, target_pitch = self._calc_angle(eye_pos, bots[0].head_pos)

        # Transfer delta = angle from first to second target
        if len(bots) >= 2:
            yaw2, pitch2 = self._calc_angle(eye_pos, bots[1].head_pos)
            d_yaw = self._normalize_yaw_delta(yaw2 - target_yaw)
            d_pitch = pitch2 - target_pitch
        else:
            d_yaw, d_pitch = 0.0, 0.0

        return Scenario(
            scenario_type="transfer",
            map_name=self.map_name,
            player_pos=player_pos,
            player_eye_height=eye_height,
            initial_yaw=target_yaw,
            initial_pitch=target_pitch,
            bots=bots,
            target_yaw=yaw2 if len(bots) >= 2 else target_yaw,
            target_pitch=pitch2 if len(bots) >= 2 else target_pitch,
            delta_yaw=d_yaw,
            delta_pitch=d_pitch,
        )

    def generate_tracking(self) -> Optional[Scenario]:
        """Generate a tracking scenario: player follows a moving target.

        Bot has a second position it will move to (simulated via interpolation).
        """
        positions = self._pick_positions(3, min_dist=200.0)
        if len(positions) < 3:
            return None

        player_pos = positions[0]
        eye_height = 64.06
        eye_pos = (player_pos[0], player_pos[1], player_pos[2] + eye_height)

        # Bot starts at positions[1], will move toward positions[2]
        bot = BotTarget(pos=positions[1], label="tracking_target")
        target_yaw, target_pitch = self._calc_angle(eye_pos, bot.head_pos)

        # Bot end position (for trajectory generation)
        bot_end = BotTarget(pos=positions[2], label="tracking_end")

        return Scenario(
            scenario_type="tracking",
            map_name=self.map_name,
            player_pos=player_pos,
            player_eye_height=eye_height,
            initial_yaw=target_yaw,
            initial_pitch=target_pitch,
            bots=[bot, bot_end],
            target_yaw=target_yaw,
            target_pitch=target_pitch,
            delta_yaw=0.0,
            delta_pitch=0.0,
        )

    def generate_hold(self) -> Optional[Scenario]:
        """Generate a hold scenario: player aims at a position, small corrections.

        Used for crosshair placement / micro-adjustments training data.
        """
        positions = self._pick_positions(2, min_dist=400.0)
        if len(positions) < 2:
            return None

        player_pos = positions[0]
        bot_pos = positions[1]
        eye_height = 64.06
        eye_pos = (player_pos[0], player_pos[1], player_pos[2] + eye_height)
        bot = BotTarget(pos=bot_pos, label="hold_target")
        target_yaw, target_pitch = self._calc_angle(eye_pos, bot.head_pos)

        # Small offset — player is almost aimed correctly
        delta_yaw = self.rng.uniform(-5, 5)
        delta_pitch = self.rng.uniform(-3, 3)

        return Scenario(
            scenario_type="hold",
            map_name=self.map_name,
            player_pos=player_pos,
            player_eye_height=eye_height,
            initial_yaw=target_yaw + delta_yaw,
            initial_pitch=max(-89, min(89, target_pitch + delta_pitch)),
            bots=[bot],
            target_yaw=target_yaw,
            target_pitch=target_pitch,
            delta_yaw=-delta_yaw,
            delta_pitch=-delta_pitch,
        )

    def generate_from_kills(
        self,
        kills_json: str,
    ) -> List[Scenario]:
        """Generate flick scenarios from real kill data with realistic crosshair-placement
        distribution: most flicks are SMALL (player saw enemy, micro-flick), few are large.

        Each kill gives:
          - Player position + victim position (real, validated by gameplay)
          - Target angle = attacker's yaw/pitch at kill moment
          - Initial angle = target + offset sampled from realistic distribution
          - Weapon = the weapon used (used for spray/recoil patterns)
        """
        with open(kills_json, 'r') as f:
            kills = json.load(f)

        # Realistic flick magnitude distribution (degrees of yaw delta).
        # Most engagements: crosshair already near enemy, only small adjustment needed.
        # Real frags have yaw deltas of 5-30° in the majority of cases.
        FLICK_BUCKETS = [
            ((2.0, 15.0),  60),   # 60% — crosshair placed close, micro-flick / tap correction
            ((15.0, 40.0), 25),   # 25% — small flick (peeking at known angle)
            ((40.0, 90.0), 12),   # 12% — medium flick (off-angle target)
            ((90.0, 160.0), 3),   #  3% — big flick (180° turn after hearing footsteps)
        ]
        ranges, weights = zip(*FLICK_BUCKETS)

        scenarios = []
        for kill in kills:
            player_pos   = tuple(kill['attacker_pos'])
            victim_pos   = tuple(kill['victim_pos'])
            target_yaw   = kill['attacker_yaw']
            target_pitch = kill['attacker_pitch']
            weapon       = self._normalize_weapon_name(kill.get('weapon', ''))
            eye_height   = 64.06

            # Sample flick magnitude from realistic distribution
            chosen_range = self.rng.choices(ranges, weights=weights, k=1)[0]
            delta_mag    = self.rng.uniform(*chosen_range)
            delta_sign   = self.rng.choice([-1, 1])
            delta_yaw    = delta_mag * delta_sign
            delta_pitch  = self.rng.uniform(-5, 5)

            initial_yaw   = target_yaw - delta_yaw
            initial_pitch = max(-89, min(89, target_pitch - delta_pitch))

            bot = BotTarget(
                pos=victim_pos,
                label="kill_target",
                victim_yaw=kill.get('victim_yaw', 0.0),
                victim_pitch=kill.get('victim_pitch', 0.0),
                ducking=kill.get('victim_ducking', False),
            )

            scenarios.append(Scenario(
                scenario_type="flick",
                map_name=self.map_name,
                player_pos=player_pos,
                player_eye_height=eye_height,
                initial_yaw=initial_yaw,
                initial_pitch=initial_pitch,
                bots=[bot],
                target_yaw=target_yaw,
                target_pitch=target_pitch,
                delta_yaw=self._normalize_yaw_delta(delta_yaw),
                delta_pitch=delta_pitch,
                weapon=weapon,
            ))

        self.rng.shuffle(scenarios)
        return scenarios

    @staticmethod
    def _normalize_weapon_name(name: str) -> str:
        """Strip 'weapon_' prefix and normalize to plugin-compatible name."""
        n = name.lower().strip()
        if n.startswith('weapon_'):
            n = n[len('weapon_'):]
        # Common aliases
        aliases = {
            'm4a4': 'm4a1', 'm4a1-s': 'm4a1_silencer',
            'usps': 'usp_silencer', 'usp-s': 'usp_silencer',
            'cz': 'cz75a',
        }
        return aliases.get(n, n)

    def generate_batch(
        self,
        n_flicks: int = 50,
        n_transfers: int = 20,
        n_tracking: int = 15,
        n_holds: int = 15,
    ) -> List[Scenario]:
        """Generate a mixed batch of scenarios."""
        scenarios = []

        for _ in range(n_flicks):
            s = self.generate_flick()
            if s:
                scenarios.append(s)

        for _ in range(n_transfers):
            s = self.generate_transfer()
            if s:
                scenarios.append(s)

        for _ in range(n_tracking):
            s = self.generate_tracking()
            if s:
                scenarios.append(s)

        for _ in range(n_holds):
            s = self.generate_hold()
            if s:
                scenarios.append(s)

        self.rng.shuffle(scenarios)
        return scenarios
