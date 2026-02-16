"""
CS2 Buy Prediction Module

Gradient boosting model for predicting optimal buy decisions
based on money, team economy, round score, and game state.
"""
from .demo_parser import parse_buy_data, parse_multiple_demos
from .features import engineer_features, create_target, prepare_dataset
from .model import BuyPredictor
from .inference import BuyAgent, GameState, quick_recommend

__all__ = [
    'parse_buy_data',
    'parse_multiple_demos',
    'engineer_features',
    'create_target',
    'prepare_dataset',
    'BuyPredictor',
    'BuyAgent',
    'GameState',
    'quick_recommend',
]
