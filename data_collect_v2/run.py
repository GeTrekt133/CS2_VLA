"""
CLI entry point for CS2 Dataset Collector v2.

Usage:
    python run.py --demo-dir "C:/path/to/demos" --output-dir "D:/dataset" --steam-id 76561198386265483
    python run.py --test                           # diagnostic test
    python run.py --test --sandbox CS2             # test with Sandboxie
    python run.py --test --demo-dir "C:/path"      # test including demo loading
    python run.py --list-devices                   # list WASAPI loopback devices
    python run.py --validate "D:/dataset/dataset.json"
    python run.py --demo-dir "..." --output-dir "..." --resume
"""

import argparse
import sys
import os
import time
import tempfile

# Add this directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from config import CollectorConfig
from collector import SingleWindowCollector
from audio import list_audio_devices, AudioRecorder
from json_builder import DatasetJsonBuilder
from virtual_display import list_monitors, print_monitors
from demo_control import DemoController
from capture import create_capture


def run_test(args):
    """Quick diagnostic: test console command, demo loading, audio, screen capture."""
    window_name = args.window_name
    sandbox_name = args.sandbox
    cs2_dir = CollectorConfig.cs2_demo_dir

    print("=" * 60)
    print("CS2 Collector — Diagnostic Test")
    if sandbox_name:
        print(f"  Sandbox: {sandbox_name}")
    print("=" * 60)

    # ---- TEST 1: CS2 Connection + Direct Console Command ----
    print("\n[TEST 1/4] CS2 Connection + Console Command")
    demo_ctrl = DemoController(window_name, cs2_dir, sandbox_name=sandbox_name)
    if not demo_ctrl.connect():
        print("  FAIL: CS2 window not found. Is CS2 running?")
        if sandbox_name:
            print(f"  Expected window: [{sandbox_name}#] {window_name}")
        return

    print("  Typing test command into CS2 console...")
    demo_ctrl._commander.send('echo ">>> CONSOLE_TEST_OK <<<"')
    time.sleep(1)

    answer = input("  Do you see '>>> CONSOLE_TEST_OK <<<' in CS2 console? [y/n]: ").strip().lower()
    if answer == 'y':
        print("  OK: Direct console input works!")
    else:
        print("  FAIL: Command not reaching CS2.")
        print("  Check:")
        print("    1. Developer Console enabled (CS2 Settings > Game > Yes)")
        print("    2. Console key is ` (backtick/tilde)")
        print("    3. CS2 window is not minimized")
        demo_ctrl.cleanup()
        return

    # ---- TEST 2: Demo Loading ----
    print("\n[TEST 2/4] Demo Loading")
    demo_dir = args.demo_dir
    if demo_dir:
        demos = [f for f in os.listdir(demo_dir) if f.endswith('.dem')]
        if demos:
            demo_name = demos[0]
            demo_stem = demo_name[:-4] if demo_name.endswith('.dem') else demo_name
            print(f"  Found demo: {demo_name}")

            print(f"  Sending: playdemo {demo_stem}")
            demo_ctrl._commander.send_batch([
                'echo ">>> LOADING DEMO <<<"',
                f'playdemo {demo_stem}',
            ])

            print("  Waiting 15s for demo to load...")
            time.sleep(15)

            answer = input("  Did CS2 start playing the demo? [y/n]: ").strip().lower()
            if answer == 'y':
                print("  OK: Demo loading works!")
                demo_ctrl._commander.send("stopdemo")
                time.sleep(2)
            else:
                print("  FAIL: Demo didn't load.")
                full_path = os.path.join(demo_dir, demo_name)
                size_mb = os.path.getsize(full_path) / (1024 * 1024) if os.path.exists(full_path) else 0
                print(f"  File: {full_path} ({size_mb:.1f} MB)")
                print(f"  CS2 game dir: {cs2_dir}")
                if os.path.normpath(demo_dir).lower() != os.path.normpath(cs2_dir).lower():
                    print(f"  WARNING: --demo-dir differs from CS2 game dir!")
                    print(f"  .dem files must be in: {cs2_dir}")
                print(f"  TIP: Try manually in CS2 console: playdemo {demo_stem}")
        else:
            print(f"  SKIP: No .dem files in {demo_dir}")
    else:
        print("  SKIP: No --demo-dir provided. Add --demo-dir to test demo loading.")
        print(f"  NOTE: .dem files must be in CS2 game dir: {cs2_dir}")

    # ---- TEST 3: Audio (WASAPI loopback) ----
    print("\n[TEST 3/4] Audio Recording (5 seconds)")
    print("  Make sure some audio is playing (YouTube, music, CS2 menu, etc.)")
    input("  Press Enter when audio is playing...")

    recorder = AudioRecorder(16000)

    if not recorder._loopback_device:
        print("  FAIL: No WASAPI loopback device found.")
        print("  Install: pip install PyAudioWPatch")
    else:
        test_wav = os.path.join(tempfile.gettempdir(), "cs2_audio_test.wav")
        recorder.start_recording(test_wav)

        for i in range(5, 0, -1):
            print(f"  Recording... {i}s remaining", end='\r')
            time.sleep(1)
        print("  Recording... done!              ")

        path = recorder.stop_recording()
        if path:
            try:
                import numpy as np
                import soundfile as sf
                audio, sr = sf.read(path)
                peak = float(np.abs(audio).max())
                rms = float(np.sqrt(np.mean(audio ** 2)))

                if peak > 0.01:
                    print(f"  OK: Audio captured! (peak={peak:.3f}, rms={rms:.4f})")
                elif peak > 0.001:
                    print(f"  WARNING: Audio very quiet (peak={peak:.4f}). Check volume.")
                else:
                    print(f"  FAIL: Audio is silent (peak={peak:.6f})")
            except ImportError:
                size = os.path.getsize(path)
                print(f"  OK: Audio file saved ({size} bytes). Install soundfile for analysis.")
        else:
            print("  FAIL: No audio data captured.")
            print("  Run: python run.py --list-devices")

        recorder.cleanup()

    # ---- TEST 4: Screen Capture ----
    print(f"\n[TEST 4/4] Screen Capture ({args.backend})")
    capture = create_capture(args.backend, window_name)
    capture.start()
    time.sleep(1)

    if not capture.is_running:
        print("  FAIL: Capture backend failed to start.")
        capture.stop()
        demo_ctrl.cleanup()
        return

    frame = capture.get_frame()
    capture.stop()

    if frame is not None:
        import numpy as np
        h, w = frame.shape[:2]
        mean_val = float(np.mean(frame))
        print(f"  OK: Frame captured ({w}x{h}, mean brightness={mean_val:.1f}/255)")
        if mean_val < 5:
            print("  WARNING: Frame is very dark — CS2 may be minimized or on loading screen")
    else:
        print("  FAIL: No frame captured. Is CS2 visible (not minimized)?")

    demo_ctrl.cleanup()

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Diagnostic test complete!")
    print("If all 4 tests passed, you're ready to collect data.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="CS2 Dataset Collector v2 — direct console input + ffmpeg audio"
    )

    # Required
    parser.add_argument('--demo-dir', type=str, help='Directory with .dem files')
    parser.add_argument('--output-dir', type=str, help='Output directory for frames/audio/dataset.json')
    parser.add_argument('--steam-id', type=int, default=76561198386265483, help='Target player Steam64 ID')

    # Capture
    parser.add_argument('--backend', type=str, default='wgc', choices=['wgc', 'dxcam'],
                        help='Capture backend (default: wgc)')
    parser.add_argument('--window-name', type=str, default='Counter-Strike 2',
                        help='CS2 window name')
    parser.add_argument('--frame-size', type=int, default=640, help='Output frame size (square)')
    parser.add_argument('--jpeg-quality', type=int, default=85, help='JPEG quality (1-100)')

    # Audio
    parser.add_argument('--no-audio', action='store_true', help='Disable audio recording')

    # Radar
    parser.add_argument('--no-radar', action='store_true', help='Skip radar overlay rendering')

    # Demo
    parser.add_argument('--skip-rounds', type=int, default=3, help='Skip first N rounds (warmup)')
    parser.add_argument('--tick-stride', type=int, default=1,
                        help='Save every Nth tick (default: 1 = every tick)')

    # Performance
    parser.add_argument('--saver-threads', type=int, default=8, help='JPEG compression threads')

    # Sandbox
    parser.add_argument('--sandbox', type=str, default=None, metavar='NAME',
                        help='Sandboxie sandbox name (e.g. CS2)')

    # Virtual display
    parser.add_argument('--virtual-display', type=int, nargs='?', const=-1, default=None,
                        metavar='INDEX',
                        help='Move CS2 to virtual monitor (auto-detect if no index given)')

    # Modes
    parser.add_argument('--offset', type=int, default=0, metavar='N',
                        help='Skip first N demos (for splitting work across VMs)')
    parser.add_argument('--limit', type=int, default=0, metavar='N',
                        help='Process only N demos after offset (0 = all)')
    parser.add_argument('--resume', action='store_true', help='Skip already-processed demos')
    parser.add_argument('--test', action='store_true', help='Run diagnostic test')
    parser.add_argument('--list-devices', action='store_true', help='List ffmpeg audio devices and exit')
    parser.add_argument('--list-monitors', action='store_true', help='List monitors and exit')
    parser.add_argument('--validate', type=str, metavar='JSON_PATH',
                        help='Validate a dataset.json file')
    parser.add_argument('--merge', type=str, nargs='+', metavar='JSON_PATH',
                        help='Merge multiple dataset.json files into one. '
                             'Usage: --merge vm1.json vm2.json --output-dir /out')

    args = parser.parse_args()

    # List audio devices
    if args.list_devices:
        list_audio_devices()
        return

    # List monitors
    if args.list_monitors:
        monitors = list_monitors()
        print_monitors(monitors)
        return

    # Validate dataset
    if args.validate:
        builder = DatasetJsonBuilder()
        ok = builder.validate(args.validate)
        sys.exit(0 if ok else 1)

    # Merge multiple dataset.json files
    if args.merge:
        if not args.output_dir:
            parser.error("--output-dir is required with --merge")
        builder = DatasetJsonBuilder()
        merged = []
        seen_paths = set()
        for json_path in args.merge:
            entries = builder.load(json_path)
            added = 0
            for e in entries:
                if e['demo_path'] not in seen_paths:
                    merged.append(e)
                    seen_paths.add(e['demo_path'])
                    added += 1
            print(f"  Loaded {json_path}: {added} demos")
        out_path = os.path.join(args.output_dir, 'dataset.json')
        builder.save(merged, out_path)
        print(f"Merged {len(merged)} demos → {out_path}")
        return

    # Diagnostic test
    if args.test:
        run_test(args)
        return

    # Collect data
    if not args.demo_dir or not args.output_dir:
        parser.error("--demo-dir and --output-dir are required for data collection")

    # Virtual display: None=disabled, -1=auto-detect, >=0=specific monitor
    vd_value = args.virtual_display  # None, -1, or specific index

    config = CollectorConfig(
        demo_dir=args.demo_dir,
        output_dir=args.output_dir,
        steam_id=args.steam_id,
        cs2_window_name=args.window_name,
        capture_backend=args.backend,
        frame_size=args.frame_size,
        jpeg_quality=args.jpeg_quality,
        record_audio=not args.no_audio,
        skip_first_rounds=args.skip_rounds,
        tick_stride=args.tick_stride,
        saver_threads=args.saver_threads,
        sandbox_name=args.sandbox,
        virtual_display=vd_value,
        render_radar=not args.no_radar,
    )

    print("=" * 60)
    print("CS2 Dataset Collector v2")
    print("=" * 60)
    print(f"  Demo dir:     {config.demo_dir}")
    print(f"  Output dir:   {config.output_dir}")
    print(f"  Steam ID:     {config.steam_id}")
    print(f"  Backend:      {config.capture_backend}")
    print(f"  Frame size:   {config.frame_size}x{config.frame_size}")
    print(f"  Tick stride:  {config.tick_stride} (every {'tick' if config.tick_stride == 1 else f'{config.tick_stride}th tick'})")
    audio_str = 'disabled' if not config.record_audio else 'WASAPI loopback'
    print(f"  Audio:        {audio_str}")
    radar_str = 'enabled' if config.render_radar else 'disabled'
    print(f"  Radar:        {radar_str}")
    print(f"  Commands:     direct console input")
    sandbox_str = config.sandbox_name or 'none'
    print(f"  Sandbox:      {sandbox_str}")
    print(f"  Saver threads: {config.saver_threads}")
    vd_str = 'disabled'
    if config.virtual_display == -1:
        vd_str = 'auto-detect'
    elif config.virtual_display is not None:
        vd_str = f'monitor [{config.virtual_display}]'
    print(f"  Virtual disp: {vd_str}")
    print(f"  Resume:       {args.resume}")
    if args.offset > 0:
        print(f"  Offset:       skip first {args.offset} demos")
    if args.limit > 0:
        print(f"  Limit:        {args.limit} demos after offset")
    print("=" * 60)

    collector = SingleWindowCollector(config)
    collector.collect_all(resume=args.resume, limit=args.limit, offset=args.offset)


if __name__ == "__main__":
    main()
