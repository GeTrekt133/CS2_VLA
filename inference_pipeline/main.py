"""
Main entry point for the CS2 VLA Agent inference pipeline.

Usage:
    python -m inference_pipeline.main --checkpoint ./checkpoints2/run_xxx/epoch_10.pth
    python -m inference_pipeline.main --checkpoint ./checkpoints2/run_xxx/epoch_10.pth --use-audio --use-buy
    python -m inference_pipeline.main --checkpoint ./checkpoints2/run_xxx/epoch_10.pth --no-actions  # Observation mode
"""

import sys
import os
import time
import signal
import argparse
import threading
from pathlib import Path

try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False
    print("[Warning] 'keyboard' library not installed. Global hotkey disabled.")
    print("         Install with: pip install keyboard")

from .config import Config, ACTION_KEYS
from .inference.engine import InferenceEngine
from .actions.input_sender import InputSender
from .actions.mouse_controller import MouseController
from .actions.keyboard_controller import KeyboardController
from .overlay.debug_overlay import DebugOverlay
from .buy.buy_executor import BuyExecutor, SimpleBuyExecutor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CS2 VLA Agent Inference Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device (cuda/cpu)"
    )

    # Features
    parser.add_argument(
        "--use-audio",
        action="store_true",
        help="Enable audio encoder"
    )

    parser.add_argument(
        "--use-buy",
        action="store_true",
        help="Enable automatic buy system"
    )

    parser.add_argument(
        "--use-trt",
        action="store_true",
        help="Use TensorRT FP16 engines for faster inference (requires conversion)"
    )

    parser.add_argument(
        "--trt-dir",
        type=str,
        default="./trt_engines",
        help="Directory containing TensorRT engines"
    )

    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable debug overlay"
    )

    parser.add_argument(
        "--no-actions",
        action="store_true",
        help="Observation mode - don't send mouse/keyboard inputs"
    )

    # Tuning
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.0,
        help="Mouse sensitivity multiplier"
    )

    parser.add_argument(
        "--key-threshold",
        type=float,
        default=0.5,
        help="Key press probability threshold"
    )

    parser.add_argument(
        "--inference-rate",
        type=int,
        default=16,
        help="Target inference rate (Hz)"
    )

    # Screen capture
    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="Monitor index to capture (1 = primary)"
    )

    parser.add_argument(
        "--screen-width",
        type=int,
        default=640,
        help="Capture width"
    )

    parser.add_argument(
        "--screen-height",
        type=int,
        default=480,
        help="Capture height"
    )

    return parser.parse_args()


def create_config(args) -> Config:
    """Create config from arguments."""
    config = Config()

    config.checkpoint_path = args.checkpoint
    config.device = args.device
    config.use_audio = args.use_audio
    config.use_buy = args.use_buy
    config.use_trt = args.use_trt
    config.trt_dir = args.trt_dir
    config.use_overlay = not args.no_overlay
    config.apply_actions = not args.no_actions

    config.mouse_sensitivity = args.sensitivity
    config.key_threshold = args.key_threshold
    config.inference_rate = args.inference_rate

    config.monitor_index = args.monitor
    config.screen_width = args.screen_width
    config.screen_height = args.screen_height

    return config


def print_banner():
    """Print startup banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════╗
    ║           CS2 VLA Agent - Inference Pipeline          ║
    ╠═══════════════════════════════════════════════════════╣
    ║  Vision-Language-Action Agent for Counter-Strike 2    ║
    ╚═══════════════════════════════════════════════════════╝
    """
    print(banner)


def print_config(config: Config, args):
    """Print configuration summary."""
    print("\n[Configuration]")
    print(f"  Checkpoint: {config.checkpoint_path}")
    print(f"  Device: {config.device}")
    print(f"  Inference rate: {config.inference_rate} Hz")
    print(f"  Screen: {config.screen_width}x{config.screen_height} (monitor {config.monitor_index})")
    print(f"\n[Features]")
    print(f"  Audio encoder: {'Enabled' if config.use_audio else 'Disabled'}")
    print(f"  TensorRT FP16: {'Enabled' if config.use_trt else 'Disabled'}")
    if config.use_trt:
        print(f"    TRT engines: {config.trt_dir}")
    print(f"  Auto-buy: {'Enabled' if config.use_buy else 'Disabled'}")
    print(f"  Debug overlay: {'Enabled' if config.use_overlay else 'Disabled'}")
    print(f"  Apply actions: {'Enabled' if config.apply_actions else 'DISABLED (observation mode)'}")
    print(f"\n[Tuning]")
    print(f"  Mouse sensitivity: {config.mouse_sensitivity}")
    print(f"  Key threshold: {config.key_threshold}")


def main():
    """Main entry point."""
    print_banner()

    # Parse arguments
    args = parse_args()

    # Verify checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"\nERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Create config
    config = create_config(args)
    print_config(config, args)

    # Create components
    print("\n[Initializing components...]")

    # Input sender
    input_sender = InputSender()

    # Controllers
    mouse_controller = MouseController(
        input_sender=input_sender,
        sensitivity=config.mouse_sensitivity,
        degrees_per_pixel=config.degrees_per_pixel,
    )

    keyboard_controller = KeyboardController(
        input_sender=input_sender,
        key_mapping=ACTION_KEYS,
        threshold=config.key_threshold,
    )

    # Overlay
    overlay = None
    if config.use_overlay:
        overlay = DebugOverlay()
        overlay.start()
        print("  Debug overlay started")

    # Buy executor
    buy_executor = None
    if config.use_buy:
        buy_executor = BuyExecutor(
            input_sender=input_sender,
            buy_model_path=config.buy_model_path,
        )
        if buy_executor.is_available:
            print("  Buy executor loaded")
        else:
            print("  Buy executor not available (model not found)")
            buy_executor = SimpleBuyExecutor(input_sender)
            print("  Using simple rule-based buy executor")

    # Create inference engine
    print("\n[Starting inference engine...]")
    engine = InferenceEngine(
        config=config,
        checkpoint_path=config.checkpoint_path,
        device=config.device,
        use_audio=config.use_audio,
        use_buy=config.use_buy,
    )

    # Attach components
    engine.set_mouse_controller(mouse_controller)
    engine.set_keyboard_controller(keyboard_controller)
    engine.set_overlay(overlay)
    if buy_executor:
        engine.set_buy_executor(buy_executor)

    # Shutdown flag
    shutdown_event = threading.Event()

    # Handle shutdown
    def signal_handler(sig, frame):
        print("\n\n[Shutdown signal received]")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    # Global hotkey (works even when game is in focus)
    if HAS_KEYBOARD:
        def on_hotkey():
            print("\n\n[Global hotkey pressed - Ctrl+Shift+Q]")
            shutdown_event.set()

        keyboard.add_hotkey('ctrl+shift+q', on_hotkey, suppress=False)
        print("  Global hotkey registered: Ctrl+Shift+Q to stop")

    # Start engine
    try:
        engine.start()

        print("\n" + "=" * 55)
        print("  Agent is running!")
        print("  Press Ctrl+Shift+Q (in-game) or Ctrl+C (terminal) to stop")
        print("=" * 55)

        if not config.apply_actions:
            print("\n  NOTE: Running in OBSERVATION MODE - no inputs sent")

        # Main loop - wait for shutdown signal
        while engine.is_running and not shutdown_event.is_set():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")

    finally:
        # Cleanup
        print("\n[Cleaning up...]")
        engine.stop()

        if overlay:
            overlay.stop()

        # Release all keys
        keyboard_controller.release_all()

        # Unhook keyboard
        if HAS_KEYBOARD:
            keyboard.unhook_all()

        print("[Agent stopped]")

        # Force exit to kill any remaining threads
        time.sleep(0.5)
        os._exit(0)


if __name__ == "__main__":
    main()
