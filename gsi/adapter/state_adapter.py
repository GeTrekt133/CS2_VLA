"""
State Adapter - converts GSI JSON to model-compatible state_vec.

Matches exactly the format from src/DatasetIntent.py:
- state_vec shape: (95,)
- [0:9] scalars: hp, armor, helmet, ammo, ct_alive, t_alive, round_time_left, bomb_planted, freeze_time
- [9:11] side one-hot (CT, T)
- [11:53] weapon one-hot (42 classes)
- [53:95] weapon_list multi-hot (42 classes)
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..config.gsi_config import WEAPONS, SIDES, GSI_WEAPON_MAP
from ..server.http_server import GSIPayload


@dataclass
class GameStateSnapshot:
    """Processed game state matching DatasetIntent.py format."""

    timestamp: float
    tick: int  # Estimated from time

    # Scalar features (9)
    hp: int
    armor: int
    helmet: bool
    ammo: int
    ct_alive: int
    t_alive: int
    round_time_left: float
    bomb_planted: bool
    freeze_time: bool

    # Categorical
    side: str  # "CT" or "T"
    weapon: str  # Current active weapon (project format)
    weapon_list: List[str]  # Inventory (project format)

    # Additional info (not in state_vec but useful)
    round_num: int
    score_ct: int
    score_t: int
    phase: str  # "live", "freezetime", "over"
    money: int

    # Extra info (not in state_vec)
    xp_level: int = 0  # XP Overload level if available


class GSIStateAdapter:
    """
    Converts GSI JSON to state_vec format used by the model.

    Output: np.ndarray of shape (95,) matching DatasetIntent.py

    Structure:
        [0:9]   - Scalars: hp, armor, helmet, ammo, ct_alive, t_alive,
                          round_time_left, bomb_planted, freeze_time
        [9:11]  - Side one-hot: [CT, T]
        [11:53] - Weapon one-hot (42 classes)
        [53:95] - Weapon list multi-hot (42 classes)
    """

    TICK_RATE = 64  # CS2 tickrate
    ROUND_DURATION = 115  # seconds (competitive)
    FREEZE_DURATION = 15  # seconds (freeze time)
    BOMB_TIMER = 40  # seconds (C4 timer)

    def __init__(self):
        self._round_start_time: Optional[float] = None
        self._phase_start_time: Optional[float] = None
        self._last_round: int = 0
        self._last_phase: str = ""
        self._estimated_tick: int = 0

    def _map_weapon(self, gsi_weapon_name: str) -> str:
        """Map GSI weapon name to project format."""
        if gsi_weapon_name is None:
            return "None"

        name = gsi_weapon_name.lower()

        # Try direct mapping
        weapon = GSI_WEAPON_MAP.get(name)
        if weapon:
            return weapon

        # Handle knife variants not in map
        if "knife" in name or "bayonet" in name or "karambit" in name or "daggers" in name:
            return "knife"

        return "None"

    def _encode_side(self, side: str) -> np.ndarray:
        """One-hot encode side (CT/T)."""
        vec = np.zeros(len(SIDES), dtype=np.float32)
        if side in SIDES:
            vec[SIDES.index(side)] = 1.0
        return vec

    def _encode_weapon(self, weapon: str) -> np.ndarray:
        """One-hot encode active weapon."""
        vec = np.zeros(len(WEAPONS), dtype=np.float32)
        w = weapon.lower()
        if w in WEAPONS:
            vec[WEAPONS.index(w)] = 1.0
        elif "knife" in w or "bayonet" in w or "karambit" in w or "daggers" in w:
            vec[WEAPONS.index("knife")] = 1.0
        else:
            vec[-1] = 1.0  # "None"
        return vec

    def _encode_weapon_list(self, weapon_list: List[str]) -> np.ndarray:
        """Multi-hot encode inventory."""
        vec = np.zeros(len(WEAPONS), dtype=np.float32)
        for weapon in weapon_list:
            w = weapon.lower()
            if w in WEAPONS:
                vec[WEAPONS.index(w)] = 1.0
            elif "knife" in w or "bayonet" in w or "karambit" in w or "daggers" in w:
                vec[WEAPONS.index("knife")] = 1.0
            else:
                vec[-1] = 1.0
        return vec

    def _count_alive_players(
        self,
        allplayers: Optional[Dict],
        team: str
    ) -> int:
        """Count alive players on a team."""
        if not allplayers:
            return 5  # Assume all alive if no data

        count = 0
        team_upper = team.upper()

        for player_id, player_data in allplayers.items():
            player_team = player_data.get("team", "")
            if player_team.upper() == team_upper:
                # Check health in state dict
                state = player_data.get("state", {})
                health = state.get("health", 0)
                if health > 0:
                    count += 1

        return min(count, 5)  # Cap at 5

    def _extract_weapons(self, weapons_dict: Optional[Dict]) -> Tuple[str, List[str], int]:
        """
        Extract active weapon, inventory, and ammo from GSI weapons dict.

        Returns:
            (active_weapon, weapon_list, active_ammo)
        """
        if not weapons_dict:
            return "None", [], 0

        active_weapon = "None"
        weapon_list = []
        active_ammo = 0

        for weapon_key, weapon_data in weapons_dict.items():
            if not isinstance(weapon_data, dict):
                continue

            weapon_name = weapon_data.get("name", "")
            mapped_name = self._map_weapon(weapon_name)

            if mapped_name != "None":
                weapon_list.append(mapped_name)

            # Check if this is the active weapon
            if weapon_data.get("state") == "active":
                active_weapon = mapped_name
                active_ammo = weapon_data.get("ammo_clip", 0)

        return active_weapon, weapon_list, active_ammo

    def parse(self, payload: GSIPayload) -> Optional[GameStateSnapshot]:
        """
        Parse GSI payload into GameStateSnapshot.

        Returns None if player data is not available (e.g., in menu).
        """
        if not payload.player:
            return None

        player = payload.player
        map_info = payload.map_info or {}
        round_info = payload.round_info or {}

        # Extract player state
        state = player.get("state", {})

        # Get HP/armor - can be in player directly or in state
        hp = player.get("health", state.get("health", 100))
        armor = player.get("armor", state.get("armor", 0))
        helmet = state.get("helmet", False)

        # Get phase
        phase = round_info.get("phase", map_info.get("phase", "unknown"))
        freeze_time = phase == "freezetime"

        # Detect round change for tick estimation
        current_round = map_info.get("round", 0)
        if current_round != self._last_round:
            self._round_start_time = payload.timestamp
            self._phase_start_time = payload.timestamp
            self._last_round = current_round
            self._last_phase = phase
            self._estimated_tick = 0
        elif self._round_start_time:
            elapsed = payload.timestamp - self._round_start_time
            self._estimated_tick = int(elapsed * self.TICK_RATE)

        # Detect phase change (freezetime -> live)
        if phase != self._last_phase:
            self._phase_start_time = payload.timestamp
            self._last_phase = phase

        # Bomb status
        bomb_state = ""
        if payload.bomb:
            bomb_state = payload.bomb.get("state", "")
        elif round_info:
            bomb_state = round_info.get("bomb", "")
        bomb_planted = bomb_state == "planted"

        # Calculate round time remaining
        round_time_left = 0.0

        # Try phase_countdowns first (spectator only)
        if payload.phase_countdowns:
            phase_ends = payload.phase_countdowns.get("phase_ends_in")
            if phase_ends:
                try:
                    round_time_left = float(phase_ends)
                except (ValueError, TypeError):
                    pass

        # Fallback: calculate from phase tracking
        if round_time_left == 0.0 and self._phase_start_time:
            elapsed_in_phase = payload.timestamp - self._phase_start_time

            if freeze_time:
                # During freeze time
                round_time_left = max(0.0, self.FREEZE_DURATION - elapsed_in_phase)
            elif bomb_planted:
                # Bomb planted - 40 second timer
                round_time_left = max(0.0, self.BOMB_TIMER - elapsed_in_phase)
            elif phase == "live":
                # Normal round time
                round_time_left = max(0.0, self.ROUND_DURATION - elapsed_in_phase)
            elif phase in ("over", "gameover"):
                round_time_left = 0.0

        # Extract weapons
        active_weapon, weapon_list, ammo = self._extract_weapons(player.get("weapons"))

        # Count alive players
        ct_alive = self._count_alive_players(payload.allplayers, "CT")
        t_alive = self._count_alive_players(payload.allplayers, "T")

        # Determine side
        team = player.get("team", "T")
        side = "CT" if team.upper() == "CT" else "T"

        # Get scores
        team_ct = map_info.get("team_ct", {})
        team_t = map_info.get("team_t", {})

        # Get money
        money = state.get("money", 0)

        # Get XP level if available
        xp_level = player.get("xpoverload_level", 0)

        return GameStateSnapshot(
            timestamp=payload.timestamp,
            tick=self._estimated_tick,
            hp=hp,
            armor=armor,
            helmet=helmet,
            ammo=ammo,
            ct_alive=ct_alive,
            t_alive=t_alive,
            round_time_left=round_time_left,
            bomb_planted=bomb_planted,
            freeze_time=freeze_time,
            side=side,
            weapon=active_weapon,
            weapon_list=weapon_list,
            round_num=current_round,
            score_ct=team_ct.get("score", 0),
            score_t=team_t.get("score", 0),
            phase=phase,
            money=money,
            xp_level=xp_level,
        )

    def to_state_vec(self, snapshot: GameStateSnapshot) -> np.ndarray:
        """
        Convert GameStateSnapshot to state_vec (95,) for model input.

        Exactly matches DatasetIntent.py format.
        """
        state_vec = np.concatenate([
            np.array([
                snapshot.hp / 100.0,
                snapshot.armor / 100.0,
                float(snapshot.helmet),
                snapshot.ammo / 100.0,
                snapshot.ct_alive / 5.0,
                snapshot.t_alive / 5.0,
                snapshot.round_time_left / 115.0,
                float(snapshot.bomb_planted),
                float(snapshot.freeze_time),
            ], dtype=np.float32),
            self._encode_side(snapshot.side),
            self._encode_weapon(snapshot.weapon),
            self._encode_weapon_list(snapshot.weapon_list),
        ]).astype(np.float32)

        return state_vec

    def to_tensor(self, snapshot: GameStateSnapshot) -> torch.Tensor:
        """Convert to PyTorch tensor."""
        return torch.from_numpy(self.to_state_vec(snapshot))

    def to_dict(self, snapshot: GameStateSnapshot) -> Dict:
        """Convert to dict format matching train_dataset.json."""
        return {
            "tick": snapshot.tick,
            "hp": snapshot.hp,
            "armor": snapshot.armor,
            "helmet": snapshot.helmet,
            "ammo": snapshot.ammo,
            "ct_alive": snapshot.ct_alive,
            "t_alive": snapshot.t_alive,
            "round_time_left": snapshot.round_time_left,
            "bomb_planted": snapshot.bomb_planted,
            "freeze_time": snapshot.freeze_time,
            "side": snapshot.side,
            "weapon": snapshot.weapon,
            "weapon_list": snapshot.weapon_list,
            "keys": [],  # GSI doesn't provide key presses
            "mouse": [0.0, 0.0],  # GSI doesn't provide mouse data
        }
