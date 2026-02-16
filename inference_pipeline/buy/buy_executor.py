"""
Automatic buy system executor.
Integrates with the buy prediction model to automatically purchase items during freeze time.
"""

import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from ..actions.input_sender import InputSender


@dataclass
class BuyState:
    """State for buy decisions."""
    money: int
    teammates_money: List[int]
    team_num: int  # 2 = T, 3 = CT
    round_num: int
    score_t: int
    score_ct: int
    freeze_time: bool = False


class BuyExecutor:
    """
    Executes automatic buy decisions during freeze time.

    Uses the buy prediction model to recommend items and sends
    console commands to purchase them.
    """

    # Buy binds for console commands
    # These require the player to have set up binds in their config
    # or we can type the commands directly
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
        buy_model_path: str = "./buy_models/buy_predictor_v1",
        console_key: str = '`',
        command_delay: float = 0.05,
    ):
        """
        Initialize buy executor.

        Args:
            input_sender: InputSender for sending key events
            buy_model_path: Path to buy prediction model
            console_key: Key to open console
            command_delay: Delay between commands
        """
        self.input_sender = input_sender
        self.buy_model_path = buy_model_path
        self.console_key = console_key
        self.command_delay = command_delay

        # Load buy agent
        self.buy_agent = None
        self._load_buy_agent()

        # State tracking
        self._last_buy_round = -1
        self._enabled = True

    def _load_buy_agent(self):
        """Load buy prediction model."""
        try:
            # Add buy_prediction to path
            base_path = Path(__file__).parent.parent.parent
            buy_path = base_path / "buy_prediction"
            if str(buy_path) not in sys.path:
                sys.path.insert(0, str(buy_path))

            from inference import BuyAgent, GameState
            self.buy_agent = BuyAgent(self.buy_model_path)
            self._game_state_class = GameState
            print(f"[BuyExecutor] Loaded model from {self.buy_model_path}")

        except Exception as e:
            print(f"[BuyExecutor] Failed to load buy model: {e}")
            self.buy_agent = None

    def should_buy(self, state: BuyState) -> bool:
        """
        Check if we should execute a buy this round.

        Args:
            state: Current game state

        Returns:
            True if we should buy
        """
        if not self._enabled:
            return False

        if self.buy_agent is None:
            return False

        if not state.freeze_time:
            return False

        if state.round_num == self._last_buy_round:
            return False

        return True

    def execute_buy(self, state: BuyState) -> Optional[Dict[str, Any]]:
        """
        Get recommendation and execute buy commands.

        Args:
            state: Current game state

        Returns:
            Buy recommendation dict or None if failed
        """
        if self.buy_agent is None:
            return None

        # Convert to GameState format
        game_state = self._game_state_class(
            money=state.money,
            teammates_money=state.teammates_money,
            team_num=state.team_num,
            round_num=state.round_num,
            score_t=state.score_t,
            score_ct=state.score_ct,
        )

        # Get recommendation
        try:
            recommendation = self.buy_agent.recommend(game_state)
        except Exception as e:
            print(f"[BuyExecutor] Recommendation failed: {e}")
            return None

        # Execute buy commands
        items = recommendation.get('suggested_items', [])
        if items:
            self._execute_commands(items)
            self._last_buy_round = state.round_num

            print(f"[BuyExecutor] Round {state.round_num}: {recommendation['buy_type']}")
            print(f"[BuyExecutor] Bought: {items}")

        return recommendation

    def _execute_commands(self, items: List[str]):
        """
        Execute buy commands for given items.

        Opens console, types commands, closes console.
        """
        # Open console
        self.input_sender.send_key_tap(self.console_key)
        time.sleep(0.1)

        for item in items:
            command = self.BUY_COMMANDS.get(item.lower())
            if command is None:
                continue

            # Type command
            self._type_command(command)
            time.sleep(self.command_delay)

            # Press Enter
            self.input_sender.send_key_tap('enter')
            time.sleep(self.command_delay)

        # Close console
        self.input_sender.send_key_tap(self.console_key)

    def _type_command(self, command: str):
        """Type a command string."""
        for char in command:
            if char == ' ':
                self.input_sender.send_key_tap('space')
            elif char == '_':
                # Underscore requires shift
                self.input_sender.send_key('shift', down=True)
                self.input_sender.send_key_tap('-')
                self.input_sender.send_key('shift', down=False)
            elif char.isalnum():
                self.input_sender.send_key_tap(char.lower())
            time.sleep(0.01)

    def set_enabled(self, enabled: bool):
        """Enable/disable auto-buy."""
        self._enabled = enabled

    def reset_round_tracking(self):
        """Reset round tracking (call on match start)."""
        self._last_buy_round = -1

    @property
    def is_available(self) -> bool:
        """Whether buy agent is loaded and available."""
        return self.buy_agent is not None

    @property
    def is_enabled(self) -> bool:
        """Whether auto-buy is enabled."""
        return self._enabled


class SimpleBuyExecutor:
    """
    Simple buy executor without ML model.
    Uses basic rules for buying.
    """

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
        is_ct = state.team_num == 3

        # Pistol rounds (1, 13)
        if state.round_num in [1, 13]:
            if budget >= 800:
                items.append('vest')
            return {'buy_type': 'pistol', 'suggested_items': items}

        # Full buy
        if budget >= 4100:
            # Primary weapon
            if budget >= 5750:
                items.append('awp')
                budget -= 4750
            elif is_ct:
                items.append('m4a1_silencer')
                budget -= 2900
            else:
                items.append('ak47')
                budget -= 2700

            # Armor
            if budget >= 1000:
                items.append('vesthelm')
                budget -= 1000

            # Utilities
            if budget >= 300:
                items.append('smokegrenade')
                budget -= 300
            if budget >= 200:
                items.append('flashbang')
                budget -= 200

            return {'buy_type': 'full_buy', 'suggested_items': items}

        # Eco
        if budget < 2000:
            return {'buy_type': 'eco', 'suggested_items': []}

        # Force buy
        if budget >= 650:
            items.append('vest')
        if budget >= 500:
            items.append('tec9' if not is_ct else 'fiveseven')

        return {'buy_type': 'force_buy', 'suggested_items': items}


def test_buy_executor():
    """Test buy executor (mock)."""
    print("Testing BuyExecutor...")

    # Mock input sender
    class MockInputSender:
        def send_key_tap(self, key):
            print(f"  [TAP] {key}")

        def send_key(self, key, down):
            action = "DOWN" if down else "UP"
            print(f"  [{action}] {key}")

    sender = MockInputSender()

    # Test simple executor
    print("\nTesting SimpleBuyExecutor...")
    simple = SimpleBuyExecutor(sender)

    states = [
        BuyState(800, [800]*4, 2, 1, 0, 0, True),   # Pistol
        BuyState(4500, [4500]*4, 2, 5, 2, 2, True),  # Full buy
        BuyState(1500, [1500]*4, 2, 3, 1, 2, True),  # Eco
    ]

    for state in states:
        if simple.should_buy(state):
            result = simple.execute_buy(state)
            print(f"\nRound {state.round_num}, ${state.money}:")
            print(f"  Type: {result['buy_type']}")
            print(f"  Items: {result['suggested_items']}")
            simple._last_buy_round = state.round_num

    print("\nTest complete!")


if __name__ == "__main__":
    test_buy_executor()
