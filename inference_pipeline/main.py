"""
Main entry point for the CS2 VLA Agent inference pipeline.

Usage:
    python -m inference_pipeline.main --checkpoint ./checkpoints_final/run_xxx/epoch_best.pth
    python -m inference_pipeline.main --checkpoint ... --use-audio --no-actions
    python -m inference_pipeline.main --checkpoint ... --write-gsi-cfg   # write CS2 GSI cfg file

GSI (Game State Integration):
    The agent uses CS2's GSI to receive game state (HP, weapons, score, etc.).
    Start the server automatically; CS2 must have gamestate_integration_cs2nn.cfg
    in its cfg/ folder. Run with --write-gsi-cfg to create it.
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
from .gsi.gsi_server import GSIServer, create_gsi_cfg


def parse_args():
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
    parser.add_argument("--device", type=str, default="cuda", help="Compute device (cuda/cpu)")

    # Features
    parser.add_argument("--use-audio", action="store_true", help="Enable stereo audio encoder")
    parser.add_argument("--use-buy", action="store_true", help="Enable automatic buy system")
    parser.add_argument("--use-trt", action="store_true", help="Use TensorRT FP16 engines")
    parser.add_argument("--trt-dir", type=str, default="./trt_engines", help="TensorRT engines directory")
    parser.add_argument("--no-overlay", action="store_true", help="Disable debug overlay")
    parser.add_argument("--no-actions", action="store_true", help="Observation mode (no inputs sent)")
    parser.add_argument("--no-gsi", action="store_true", help="Disable GSI server (no game state)")

    # GSI
    parser.add_argument("--gsi-port", type=int, default=3000, help="GSI server port")
    parser.add_argument("--gsi-host", type=str, default="127.0.0.1", help="GSI server host")
    parser.add_argument(
        "--write-gsi-cfg",
        action="store_true",
        help="Write gamestate_integration_cs2nn.cfg to CS2 cfg folder and exit"
    )
    parser.add_argument(
        "--gsi-cfg-path",
        type=str,
        default=None,
        help="Custom path for CS2 GSI config file (used with --write-gsi-cfg)"
    )

    # Tuning
    parser.add_argument("--sensitivity", type=float, default=1.0, help="Mouse sensitivity multiplier")
    parser.add_argument("--key-threshold", type=float, default=0.5, help="Key press probability threshold")
    parser.add_argument("--inference-rate", type=int, default=16, help="Target inference rate (Hz)")

    # Screen capture
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index (1=primary)")
    parser.add_argument("--screen-width", type=int, default=640, help="Capture width (YOLO: 640)")
    parser.add_argument("--screen-height", type=int, default=640, help="Capture height (YOLO: 640)")

    return parser.parse_args()


def create_config(args) -> Config:
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

    config.gsi_host = args.gsi_host
    config.gsi_port = args.gsi_port

    return config


def print_banner():
    banner = """
    ╔═══════════════════════════════════════════════════════╗
    ║           CS2 VLA Agent - Inference Pipeline          ║
    ╠═══════════════════════════════════════════════════════╣
    ║  Vision-Language-Action Agent for Counter-Strike 2    ║
    ╚═══════════════════════════════════════════════════════╝
    """
    print(banner)


def print_config(config: Config, args):
    print("\n[Configuration]")
    print(f"  Checkpoint: {config.checkpoint_path}")
    print(f"  Device: {config.device}")
    print(f"  Inference rate: {config.inference_rate} Hz")
    print(f"  Screen: {config.screen_width}x{config.screen_height} (monitor {config.monitor_index})")
    print(f"\n[Features]")
    print(f"  Audio encoder: {'Enabled (stereo 16sec)' if config.use_audio else 'Disabled'}")
    print(f"  TensorRT FP16: {'Enabled' if config.use_trt else 'Disabled'}")
    if config.use_trt:
        print(f"    TRT engines: {config.trt_dir}")
    print(f"  Auto-buy: {'Enabled' if config.use_buy else 'Disabled'}")
    print(f"  Debug overlay: {'Enabled' if config.use_overlay else 'Disabled'}")
    print(f"  Apply actions: {'Enabled' if config.apply_actions else 'DISABLED (observation mode)'}")
    print(f"  GSI server: {'Disabled (--no-gsi)' if args.no_gsi else f'http://{config.gsi_host}:{config.gsi_port}'}")
    print(f"\n[Tuning]")
    print(f"  Mouse sensitivity: {config.mouse_sensitivity}")
    print(f"  Key threshold: {config.key_threshold}")


def main():
    print_banner()

    args = parse_args()

    # Handle --write-gsi-cfg
    if args.write_gsi_cfg:
        gsi_path = args.gsi_cfg_path
        if gsi_path is None:
            gsi_path = (
                r"C:\Program Files (x86)\Steam\steamapps\common"
                r"\Counter-Strike Global Offensive\game\csgo\cfg"
                r"\gamestate_integration_cs2nn.cfg"
            )
        create_gsi_cfg(
            output_path=gsi_path,
            host=args.gsi_host,
            port=args.gsi_port,
            auth_token=Config().gsi_auth_token,
        )
        print("\nDone. Restart CS2 for the config to take effect.")
        print("Then run the agent normally (without --write-gsi-cfg).")
        return

    # Verify checkpoint
    if not Path(args.checkpoint).exists():
        print(f"\nERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    config = create_config(args)
    print_config(config, args)

    # Create components
    print("\n[Initializing components...]")

    input_sender = InputSender()

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

    overlay = None
    if config.use_overlay:
        overlay = DebugOverlay()
        overlay.start()
        print("  Debug overlay started")

    buy_executor = None
    if config.use_buy:
        buy_executor = BuyExecutor(
            input_sender=input_sender,
            buy_model_path=config.buy_model_path,
        )
        if buy_executor.is_available:
            print("  Buy executor loaded")
        else:
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

    engine.set_mouse_controller(mouse_controller)
    engine.set_keyboard_controller(keyboard_controller)
    engine.set_overlay(overlay)
    if buy_executor:
        engine.set_buy_executor(buy_executor)

    # Start GSI server
    gsi_server = None
    if not args.no_gsi:
        gsi_server = GSIServer(
            host=config.gsi_host,
            port=config.gsi_port,
            auth_token=config.gsi_auth_token,
            callback=engine.update_game_state,
        )
        gsi_server.start()

    # Shutdown flag
    shutdown_event = threading.Event()

    def signal_handler(sig, frame):
        print("\n\n[Shutdown signal received]")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)

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
        if gsi_server:
            print(f"  GSI: http://{config.gsi_host}:{config.gsi_port}")
            print(f"  Run with --write-gsi-cfg to create CS2 config file")
        print("  Press Ctrl+Shift+Q (in-game) or Ctrl+C (terminal) to stop")
        print("=" * 55)

        if not config.apply_actions:
            print("\n  NOTE: Running in OBSERVATION MODE - no inputs sent")

        if gsi_server:
            print("\n  Tip: If GSI state is not updating, run:")
            print(f"    python -m inference_pipeline.main --checkpoint ... --write-gsi-cfg")

        # Main loop
        while engine.is_running and not shutdown_event.is_set():
            time.sleep(0.1)

            # Log GSI status every 30 sec
            if gsi_server and engine._inference_count % (30 * config.inference_rate) == 0 and engine._inference_count > 0:
                secs = gsi_server.seconds_since_update()
                if secs > 5:
                    print(f"[WARNING] No GSI update for {secs:.0f}s — game state may be stale")

    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")

    finally:
        print("\n[Cleaning up...]")
        engine.stop()

        if gsi_server:
            gsi_server.stop()

        if overlay:
            overlay.stop()

        keyboard_controller.release_all()

        if HAS_KEYBOARD:
            keyboard.unhook_all()

        print("[Agent stopped]")

        time.sleep(0.5)
        os._exit(0)


if __name__ == "__main__":
    main()
