"""
CS2 VLA Agent Inference Pipeline.

Real-time inference pipeline for the CS2 Vision-Language-Action agent.
Captures screen, audio, and game state, runs neural network inference,
and applies predictions as mouse/keyboard actions in-game.

Usage:
    python -m inference_pipeline.main --checkpoint ./checkpoints2/run_xxx/epoch_10.pth
"""

from .config import Config

__version__ = "1.0.0"
__all__ = ["Config"]
