#!/usr/bin/env python3
"""
Start GSI inference mode.

Usage:
    python gsi/run_inference.py --checkpoint ./checkpoints2/run_xxx/epoch_10.pth
    python gsi/run_inference.py --checkpoint ./checkpoints2/run_xxx/epoch_10.pth --device cpu
"""
import sys
import argparse
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gsi.inference.inference_engine import GSIInferenceEngine, InferenceResult
from gsi.config.gsi_config import GSIConfig


def on_prediction(result: InferenceResult):
    """Handle inference results - print to console."""
    # Format key predictions
    keys_str = ", ".join(result.predicted_keys) if result.predicted_keys else "none"

    # Mouse delta
    mouse_str = f"({result.mouse_delta[0]:+.2f}, {result.mouse_delta[1]:+.2f})"

    # Top 3 key probabilities
    sorted_probs = sorted(
        result.key_probs.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    top_probs = " | ".join([f"{k}:{v:.2f}" for k, v in sorted_probs])

    print(f"\r[R{result.round_num}] HP:{result.hp:3d} | "
          f"Mouse: {mouse_str} | "
          f"Keys: [{keys_str:30s}] | "
          f"Top: {top_probs}",
          end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description="CS2 GSI Inference")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=3000,
        help="GSI server port (default: 3000)"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        help="Device: cuda or cpu (default: cuda)"
    )

    args = parser.parse_args()

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    config = GSIConfig(
        port=args.port,
        device=args.device,
    )

    print("=" * 60)
    print("  CS2 GSI Inference Engine")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Port:       {args.port}")
    print(f"  Device:     {args.device}")
    print("=" * 60)
    print()
    print("  1. Ensure gamestate_integration_cs2nn.cfg is installed")
    print("  2. Launch CS2 and join a match")
    print("  3. Model predictions will appear below")
    print("  4. Press Ctrl+C to stop")
    print()
    print("=" * 60)

    engine = GSIInferenceEngine(
        checkpoint_path=args.checkpoint,
        config=config,
        device=args.device,
    )

    engine.register_callback(on_prediction)

    try:
        engine.start()

        # Keep running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n[!] Stopping inference...")

    finally:
        engine.stop()


if __name__ == "__main__":
    main()
