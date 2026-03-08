"""
Inference for CS2 buy prediction.
Uses only GSI-available features at runtime.

Usage:
    agent = BuyAgent("./buy_models/buy_v2")
    items = agent.recommend(
        money=4500, round_num=5, team_score=2, enemy_score=2,
        loss_streak=0, equipment_value=200, team='T'
    )
    # -> ['ak47', 'vesthelm', 'smokegrenade', 'flashbang', 'molotov']
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path

from .model import BuyPredictor
from .features import (
    WEAPON_PRICES, PRIMARY_LABELS, ARMOR_LABELS,
    LOSS_BONUS, engineer_features,
)


# Default weapon picks per side and primary class
WEAPON_MAP = {
    # primary_class: {team_num: weapon_name}
    1: {2: 'tec9', 3: 'fiveseven'},        # pistol upgrade
    2: {2: 'mac10', 3: 'mp9'},             # smg
    3: {2: 'ak47', 3: 'm4a1_silencer'},    # rifle
    4: {2: 'awp', 3: 'awp'},               # awp
    5: {2: 'ssg08', 3: 'ssg08'},           # scout
}

ARMOR_MAP = {1: 'vest', 2: 'vesthelm'}

GRENADE_MAP = {
    'buy_smoke': 'smokegrenade',
    'buy_flash': 'flashbang',
    'buy_he': 'hegrenade',
    'buy_molotov': {2: 'molotov', 3: 'incgrenade'},
}


class BuyAgent:
    """
    Buy recommendation agent using only GSI data.

    All inputs are available from CS2 Game State Integration:
      - money: player.state.money
      - round_num: map.round + 1
      - team_score/enemy_score: from map.team_ct.score / map.team_t.score
      - loss_streak: computed from map.round_wins pattern
      - equipment_value: player.state.equip_value
      - team: player.team (T/CT)
    """

    def __init__(self, model_path: str):
        self.predictor = BuyPredictor()
        self.predictor.load(model_path)

    def _build_features(
        self,
        money: int,
        round_num: int,
        team_score: int,
        enemy_score: int,
        loss_streak: int,
        equipment_value: int,
        team_num: int,
    ) -> pd.DataFrame:
        """Build feature DataFrame from GSI values."""
        row = pd.DataFrame([{
            'money': money,
            'equipment_value': equipment_value,
            'round_num': round_num,
            'team_score': team_score,
            'enemy_score': enemy_score,
            'loss_streak': min(loss_streak, 4),
            'team_num': team_num,
        }])
        return engineer_features(row)

    def recommend(
        self,
        money: int,
        round_num: int,
        team_score: int,
        enemy_score: int,
        loss_streak: int = 0,
        equipment_value: int = 0,
        team: str = 'T',
    ) -> Dict[str, Any]:
        """
        Get exact buy recommendation.

        Returns:
            {
                'items': ['ak47', 'vesthelm', 'smokegrenade', ...],
                'total_cost': 4400,
                'money_remaining': 100,
                'primary': 'rifle',
                'armor': 'vesthelm',
                'grenades': ['smokegrenade', 'flashbang'],
            }
        """
        team_num = 2 if team.upper() == 'T' else 3
        features = self._build_features(
            money, round_num, team_score, enemy_score,
            loss_streak, equipment_value, team_num
        )

        preds = self.predictor.predict(features)
        items = []
        budget = money

        # Primary weapon
        primary_cls = int(preds.get('primary', [0])[0])
        if primary_cls > 0 and primary_cls in WEAPON_MAP:
            weapon = WEAPON_MAP[primary_cls][team_num]
            cost = WEAPON_PRICES.get(weapon, 0)
            if budget >= cost:
                items.append(weapon)
                budget -= cost

        # Armor
        armor_cls = int(preds.get('armor', [0])[0])
        if armor_cls > 0 and armor_cls in ARMOR_MAP:
            armor = ARMOR_MAP[armor_cls]
            cost = WEAPON_PRICES.get(armor, 0)
            if budget >= cost:
                items.append(armor)
                budget -= cost

        # Grenades
        grenades = []
        for key, item_or_map in GRENADE_MAP.items():
            if int(preds.get(key, [0])[0]):
                if isinstance(item_or_map, dict):
                    item = item_or_map[team_num]
                else:
                    item = item_or_map
                cost = WEAPON_PRICES.get(item, 0)
                if budget >= cost:
                    items.append(item)
                    grenades.append(item)
                    budget -= cost

        # Defuser (CT only)
        if team_num == 3 and int(preds.get('buy_defuser', [0])[0]):
            cost = WEAPON_PRICES.get('defuser', 400)
            if budget >= cost:
                items.append('defuser')
                budget -= cost

        total_cost = money - budget

        return {
            'items': items,
            'total_cost': total_cost,
            'money_remaining': budget,
            'primary': PRIMARY_LABELS.get(primary_cls, 'none'),
            'armor': ARMOR_LABELS.get(armor_cls, 'none'),
            'grenades': grenades,
        }

    def recommend_from_gsi(self, gsi_state: dict) -> Dict[str, Any]:
        """
        Recommend directly from GSI JSON state.

        Expected gsi_state structure:
            player.state.money, player.state.equip_value, player.team
            map.round, map.team_ct.score, map.team_t.score, map.round_wins
        """
        player = gsi_state.get('player', {})
        state = player.get('state', {})
        map_data = gsi_state.get('map', {})

        money = state.get('money', 0)
        equip = state.get('equip_value', 0)
        team = player.get('team', 'T')
        round_num = map_data.get('round', 0) + 1

        if team.upper() in ('T', 'TERRORIST'):
            team_score = map_data.get('team_t', {}).get('score', 0)
            enemy_score = map_data.get('team_ct', {}).get('score', 0)
        else:
            team_score = map_data.get('team_ct', {}).get('score', 0)
            enemy_score = map_data.get('team_t', {}).get('score', 0)

        # Compute loss streak from round_wins
        loss_streak = self._compute_loss_streak(
            map_data.get('round_wins', {}), team
        )

        return self.recommend(
            money=money,
            round_num=round_num,
            team_score=team_score,
            enemy_score=enemy_score,
            loss_streak=loss_streak,
            equipment_value=equip,
            team=team,
        )

    @staticmethod
    def _compute_loss_streak(round_wins: dict, team: str) -> int:
        """
        Compute current loss streak from GSI round_wins.

        round_wins is like {"1": "ct", "2": "t", "3": "ct", ...}
        """
        if not round_wins:
            return 0

        team_lower = 'ct' if team.upper() in ('CT', 'COUNTER-TERRORIST') else 't'
        # Get rounds in order
        rounds = sorted(round_wins.items(), key=lambda x: int(x[0]), reverse=True)

        streak = 0
        for _, winner in rounds:
            if winner.lower() != team_lower:
                streak += 1
            else:
                break
        return min(streak, 4)
