"""
CS2 Game State Integration (GSI) module.

Provides real-time game state data from CS2 for:
- Data collection (training)
- Real-time inference
"""

from .server.http_server import GSIServer, GSIPayload
from .adapter.state_adapter import GSIStateAdapter, GameStateSnapshot
from .config.gsi_config import GSIConfig

__all__ = [
    'GSIServer',
    'GSIPayload',
    'GSIStateAdapter',
    'GameStateSnapshot',
    'GSIConfig',
]
