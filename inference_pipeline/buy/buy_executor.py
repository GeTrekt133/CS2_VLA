"""
Automatic buy system executor.
Integrates with the buy prediction model (BuyAgent) to automatically
purchase items during freeze time using only GSI-available features.
"""

import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from ..actions.input_sender import InputSender


@dataclass
class BuyState:
    """State for buy decisions (GSI-only features)."""
    money: int
    equipment_value: int
    team: str            # 'T' or 'CT'
    round_num: int
    team_score: int
    enemy_score: int
    loss_streak: int = 0
    freeze_time: bool = False


class BuyExecutor:
    """
    Executes automatic buy decisions during freeze time.

    Uses the BuyAgent (LightGBM) to recommend exact items and sends
    console commands to purchase them.
    """

    BUY_COMMANDS = {
        # Rifles
        'ak47': 'buy ak47',
        'm4a1_silencer': 'buy m4a1_silencer',
        'm4a4': 'buy m4a1',
        'awp': 'buy awp',
        'aug': 'buy aug',
        'sg553': 'buy sg556',
        'famas': 'buy famas',
        'galil': 'buy galilar',
        'ssg08': 'buy ssg08',

        # SMGs
        'mac10': 'buy mac10',
        'mp9': 'buy mp9',
        'mp7': 'buy mp7',
        'ump45': 'buy ump45',
        'p90': 'buy p90',
        'bizon': 'buy bizon',

        # Heavy
        'nova': 'buy nova',
        'xm1014': 'buy xm1014',
        'mag7': 'buy mag7',
        'sawedoff': 'buy sawedoff',
        'm249': 'buy m249',
        'negev': 'buy negev',

        # Pistols
        'glock': 'buy glock',
        'p2000': 'buy hkp2000',
        'usp_silencer': 'buy usp_silencer',
        'p250': 'buy p250',
        'fiveseven': 'buy fiveseven',
        'tec9': 'buy tec9',
        'cz75': 'buy cz75a',
        'deagle': 'buy deagle',
        'revolver': 'buy revolver',
        'dualies': 'buy elite',

        # Equipment
        'vest': 'buy vest',
        'vesthelm': 'buy vesthelm',
        'defuser': 'buy defuser',
        'zeus': 'buy taser',

        # Grenades
        'hegrenade': 'buy hegrenade',
        'flashbang': 'buy flashbang',
        'smokegrenade': 'buy smokegrenade',
        'molotov': 'buy molotov',
        'incgrenade': 'buy incgrenade',
        'decoy': 'buy decoy',
    }

    def __init__(
        self,
        input_sender: InputSender,
        buy_model_path: str = "./buy_models/buy_v2",
        console_key: str = '`',
        command_delay: float = 0.05,
    ):
        self.input_sender = input_sender
        self.buy_model_path = buy_model_path
        self.console_key = console_key
        self.command_delay = command_delay

        self.buy_agent = None
        self._load_buy_agent()

        self._last_buy_round = -1
        self._enabled = True

    def _load_buy_agent(self):
        """Load buy prediction model (BuyAgent from buy_prediction package)."""
        try:
            base_path = Path(__file__).parent.parent.parent
            buy_path = base_path / "buy_prediction"
            if str(buy_path) not in sys.path:
                sys.path.insert(0, str(buy_path))

            from buy_prediction.inference import BuyAgent
            self.buy_agent = BuyAgent(self.buy_model_path)
            print(f"[BuyExecutor] Loaded model from {self.buy_model_path}")

        except Exception as e:
            print(f"[BuyExecutor] Failed to load buy model: {e}")
            self.buy_agent = None

    def should_buy(self, state: BuyState) -> bool:
        if not self._enabled or self.buy_agent is None:
            return False
        if not state.freeze_time:
            return False
        if state.round_num == self._last_buy_round:
            return False
        return True

    def execute_buy(self, state: BuyState) -> Optional[Dict[str, Any]]:
        """Get recommendation and execute buy commands."""
        if self.buy_agent is None:
            return None

        try:
            recommendation = self.buy_agent.recommend(
                money=state.money,
                round_num=state.round_num,
                team_score=state.team_score,
                enemy_score=state.enemy_score,
                loss_streak=state.loss_streak,
                equipment_value=state.equipment_value,
                team=state.team,
            )
        except Exception as e:
            print(f"[BuyExecutor] Recommendation failed: {e}")
            return None

        items = recommendation.get('items', [])
        if items:
            self._execute_commands(items)
            self._last_buy_round = state.round_num

            print(f"[BuyExecutor] Round {state.round_num}: {recommendation['primary']} | "
                  f"Items: {items} | Cost: ${recommendation['total_cost']}")

        return recommendation

    def _execute_commands(self, items: List[str]):
        """Execute buy commands via console."""
        self.input_sender.send_key_tap(self.console_key)
        time.sleep(0.1)

        for item in items:
            command = self.BUY_COMMANDS.get(item.lower())
            if command is None:
                continue
            self._type_command(command)
            time.sleep(self.command_delay)
            self.input_sender.send_key_tap('enter')
            time.sleep(self.command_delay)

        self.input_sender.send_key_tap(self.console_key)

    def _type_command(self, command: str):
        for char in command:
            if char == ' ':
                self.input_sender.send_key_tap('space')
            elif char == '_':
                self.input_sender.send_key('shift', down=True)
                self.input_sender.send_key_tap('-')
                self.input_sender.send_key('shift', down=False)
            elif char.isalnum():
                self.input_sender.send_key_tap(char.lower())
            time.sleep(0.01)

    def set_enabled(self, enabled: bool):
        self._enabled = enabled

    def reset_round_tracking(self):
        self._last_buy_round = -1

    @property
    def is_available(self) -> bool:
        return self.buy_agent is not None

    @property
    def is_enabled(self) -> bool:
        return self._enabled


class SimpleBuyExecutor:
    """Simple rule-based buy executor without ML model."""

    def __init__(self, input_sender: InputSender):
        self.input_sender = input_sender
        self._last_buy_round = -1

    def should_buy(self, state: BuyState) -> bool:
        if not state.freeze_time:
            return False
        if state.round_num == self._last_buy_round:
            return False
        return True

    def execute_buy(self, state: BuyState) -> Dict[str, Any]:
        """Execute simple rule-based buy."""
        items = []
        budget = state.money
        is_ct = state.team.upper() == 'CT'

        # Pistol rounds (1, 13)
        if state.round_num in [1, 13]:
            if budget >= 800:
                items.append('vest')
            return {'items': items, 'primary': 'pistol', 'total_cost': 0}

        # Full buy
        if budget >= 4100:
            if budget >= 5750:
                items.append('awp')
                budget -= 4750
            elif is_ct:
                items.append('m4a1_silencer')
                budget -= 2900
            else:
                items.append('ak47')
                budget -= 2700

            if budget >= 1000:
                items.append('vesthelm')
                budget -= 1000
            if budget >= 300:
                items.append('smokegrenade')
                budget -= 300
            if budget >= 200:
                items.append('flashbang')
                budget -= 200

            return {'items': items, 'primary': 'rifle', 'total_cost': state.money - budget}

        # Eco
        if budget < 2000:
            return {'items': [], 'primary': 'none', 'total_cost': 0}

        # Force buy
        if budget >= 650:
            items.append('vest')
        if budget >= 500:
            items.append('tec9' if not is_ct else 'fiveseven')

        return {'items': items, 'primary': 'pistol', 'total_cost': state.money - budget}
