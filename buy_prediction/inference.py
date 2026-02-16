"""
Inference module for CS2 buy prediction.
Use this to get buy recommendations during gameplay.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from model import BuyPredictor
from features import WEAPON_PRICES, WEAPON_CATEGORIES


@dataclass
class GameState:
    """Current game state for buy prediction."""
    money: int
    teammates_money: List[int]  # List of 4 teammate money values
    team_num: int  # 2 = T, 3 = CT
    round_num: int
    score_t: int
    score_ct: int


class BuyAgent:
    """
    Agent for making buy decisions in CS2.

    Usage:
        agent = BuyAgent("./buy_models/buy_predictor_v1")
        state = GameState(
            money=4500,
            teammates_money=[3200, 5100, 2800, 4000],
            team_num=2,  # T side
            round_num=5,
            score_t=2,
            score_ct=2
        )
        recommendation = agent.recommend(state)
    """

    def __init__(self, model_path: str):
        """
        Initialize the buy agent.

        Args:
            model_path: Path to saved model directory
        """
        self.predictor = BuyPredictor()
        self.predictor.load(model_path)

    def _create_features(self, state: GameState) -> pd.DataFrame:
        """Convert GameState to feature DataFrame."""
        teammates = state.teammates_money

        # Calculate team statistics
        team_money_total = sum(teammates) + state.money
        team_money_avg = team_money_total / 5

        # Score difference from player's perspective
        if state.team_num == 2:  # T side
            score_diff = state.score_t - state.score_ct
        else:  # CT side
            score_diff = state.score_ct - state.score_t

        features = {
            'money': state.money,
            'money_log': np.log1p(state.money),
            'team_money_total': team_money_total,
            'team_money_avg': team_money_avg,
            'can_full_buy': int(state.money >= 4100),
            'can_awp': int(state.money >= 5750),
            'can_half_buy': int(state.money >= 2000),
            'money_for_save': int(state.money < 1500),
            'team_can_full_buy': int(team_money_avg >= 4100),
            'team_poor': int(team_money_avg < 2000),
            'money_diff_from_team': state.money - team_money_avg,
            'money_ratio_to_team': state.money / (team_money_avg + 1),
            'teammate_money_min': min(teammates) if teammates else 0,
            'teammate_money_max': max(teammates) if teammates else 0,
            'teammate_money_std': np.std(teammates) if teammates else 0,
            'teammates_can_buy': sum(1 for m in teammates if m >= 4100),
            'round_num': state.round_num,
            'score_t': state.score_t,
            'score_ct': state.score_ct,
            'score_diff': score_diff,
            'total_rounds_played': state.score_t + state.score_ct,
            'is_winning': int(score_diff > 0),
            'is_losing': int(score_diff < 0),
            'is_tied': int(score_diff == 0),
            'is_first_half': int(state.round_num <= 12),
            'is_second_half': int(state.round_num > 12),
            'rounds_until_half': max(0, 12 - state.round_num) if state.round_num <= 12 else 0,
            'is_last_round_first_half': int(state.round_num == 12),
            'is_last_round_second_half': int(state.round_num == 24),
            'is_t_side': int(state.team_num == 2),
            'is_ct_side': int(state.team_num == 3),
            'round_after_pistol': int(state.round_num in [2, 14]),
            'is_pistol_round': int(state.round_num in [1, 13]),
        }

        return pd.DataFrame([features])

    def recommend(self, state: GameState) -> Dict[str, Any]:
        """
        Get buy recommendation for current game state.

        Args:
            state: Current GameState

        Returns:
            Dictionary with:
            - buy_type: 'eco', 'force_buy', 'half_buy', 'full_buy'
            - confidence: float 0-1
            - suggested_items: list of item names
            - estimated_cost: total cost of suggested items
            - money_remaining: money after purchase
        """
        features = self._create_features(state)
        predictions = self.predictor.predict(features)

        buy_types = ['eco', 'force_buy', 'half_buy', 'full_buy']
        buy_idx = predictions['buy_type'][0]
        buy_type = buy_types[buy_idx]
        confidence = float(predictions['buy_type_probs'][0, buy_idx])

        # Get item probabilities
        rifle_prob = predictions.get('rifle_prob', [0])[0]
        armor_prob = predictions.get('armor_prob', [0])[0]
        awp_prob = predictions.get('awp_prob', [0])[0]

        # Generate suggested items based on buy type and probabilities
        suggested_items = self._suggest_items(
            state, buy_type, rifle_prob, armor_prob, awp_prob
        )

        # Calculate costs
        estimated_cost = sum(WEAPON_PRICES.get(item, 0) for item in suggested_items)
        money_remaining = state.money - estimated_cost

        return {
            'buy_type': buy_type,
            'confidence': confidence,
            'suggested_items': suggested_items,
            'estimated_cost': estimated_cost,
            'money_remaining': max(0, money_remaining),
            'probabilities': {
                'rifle': float(rifle_prob),
                'armor': float(armor_prob),
                'awp': float(awp_prob),
            }
        }

    def _suggest_items(
        self,
        state: GameState,
        buy_type: str,
        rifle_prob: float,
        armor_prob: float,
        awp_prob: float
    ) -> List[str]:
        """Generate suggested items based on predictions and budget."""
        items = []
        budget = state.money
        is_ct = state.team_num == 3

        if buy_type == 'eco':
            # Maybe a pistol upgrade if rich enough
            if budget >= 500:
                items.append('p250')
                budget -= 300
            return items

        if buy_type == 'force_buy':
            # Upgrade pistol + utility
            if budget >= 500:
                items.append('tec9' if not is_ct else 'fiveseven')
                budget -= 500

            if budget >= 200:
                items.append('flashbang')
                budget -= 200

            if armor_prob > 0.5 and budget >= 650:
                items.append('vest')
                budget -= 650

            return items

        if buy_type == 'half_buy':
            # SMG + armor
            if budget >= 1250:
                items.append('mac10' if not is_ct else 'mp9')
                budget -= 1050 if not is_ct else 1250

            if budget >= 650:
                items.append('vest')
                budget -= 650

            return items

        # Full buy
        if awp_prob > 0.5 and budget >= 5750:
            items.append('awp')
            budget -= 4750

            if budget >= 1000:
                items.append('vesthelm')
                budget -= 1000
        elif rifle_prob > 0.3 or budget >= 4100:
            # Rifle
            if is_ct:
                items.append('m4a1_silencer')
                budget -= 2900
            else:
                items.append('ak47')
                budget -= 2700

            # Armor
            if budget >= 1000:
                items.append('vesthelm')
                budget -= 1000
            elif budget >= 650:
                items.append('vest')
                budget -= 650

        # Add utilities with remaining budget
        if budget >= 300:
            items.append('smokegrenade')
            budget -= 300

        if budget >= 200:
            items.append('flashbang')
            budget -= 200

        if budget >= 300 and not is_ct:
            items.append('molotov')
            budget -= 400
        elif budget >= 600 and is_ct:
            items.append('incgrenade')
            budget -= 600

        if budget >= 300:
            items.append('hegrenade')
            budget -= 300

        return items

    def recommend_batch(self, states: List[GameState]) -> List[Dict[str, Any]]:
        """Get recommendations for multiple game states."""
        return [self.recommend(state) for state in states]


def quick_recommend(
    money: int,
    teammates_money: List[int],
    team: str = 'T',
    round_num: int = 1,
    score: tuple = (0, 0),
    model_path: str = "./buy_models/buy_predictor_v1"
) -> Dict[str, Any]:
    """
    Quick function for getting buy recommendation.

    Args:
        money: Player's current money
        teammates_money: List of 4 teammate money values
        team: 'T' or 'CT'
        round_num: Current round number (1-30)
        score: Tuple of (T score, CT score)
        model_path: Path to model

    Returns:
        Buy recommendation dictionary
    """
    agent = BuyAgent(model_path)

    state = GameState(
        money=money,
        teammates_money=teammates_money,
        team_num=2 if team.upper() == 'T' else 3,
        round_num=round_num,
        score_t=score[0],
        score_ct=score[1]
    )

    return agent.recommend(state)


if __name__ == "__main__":
    # Example usage
    print("CS2 Buy Prediction Agent")
    print("=" * 50)

    # Check if model exists
    model_path = "./buy_models/buy_predictor_v1"
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Train a model first using train.py")
        print("\nShowing example with dummy predictions...")

        # Demo without trained model
        state = GameState(
            money=4500,
            teammates_money=[3200, 5100, 2800, 4000],
            team_num=2,
            round_num=5,
            score_t=2,
            score_ct=2
        )
        print(f"\nGame State:")
        print(f"  Money: ${state.money}")
        print(f"  Team money: {state.teammates_money}")
        print(f"  Round: {state.round_num}")
        print(f"  Score: {state.score_t}-{state.score_ct}")
    else:
        agent = BuyAgent(model_path)

        # Test different scenarios
        scenarios = [
            GameState(800, [800, 800, 800, 800], 2, 1, 0, 0),  # Pistol round
            GameState(1400, [1400, 1400, 1400, 1400], 2, 2, 0, 1),  # After losing pistol
            GameState(4500, [4200, 5100, 3800, 4000], 2, 5, 2, 2),  # Mid-game full buy
            GameState(2100, [1500, 3000, 2500, 2800], 3, 8, 3, 4),  # Force buy CT
            GameState(6500, [6000, 5500, 7000, 6200], 2, 12, 5, 6),  # Last round first half
        ]

        for i, state in enumerate(scenarios):
            rec = agent.recommend(state)
            side = 'T' if state.team_num == 2 else 'CT'
            print(f"\nScenario {i+1}: Round {state.round_num}, {side} side, ${state.money}")
            print(f"  Recommendation: {rec['buy_type']} (conf: {rec['confidence']:.2f})")
            print(f"  Suggested: {rec['suggested_items']}")
            print(f"  Cost: ${rec['estimated_cost']}, Remaining: ${rec['money_remaining']}")
