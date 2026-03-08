#!/usr/bin/env python3
"""
Start GSI data collection mode.

Usage:
    python gsi/run_collector.py
    python gsi/run_collector.py --output ./my_data --port 3000

Then:
1. Launch CS2
2. Play matches - data will be collected automatically
3. Press Ctrl+C to stop and save
"""
import sys
import argparse
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gsi.collector.data_collector import GSIDataCollector
from gsi.config.gsi_config import GSIConfig


def main():
    parser = argparse.ArgumentParser(description="CS2 GSI Data Collector")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./gsi_data",
        help="Output directory for collected data"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=3000,
        help="GSI server port (default: 3000)"
    )
    parser.add_argument(
        "--no-screenshots",
        action="store_true",
        help="Disable screenshot capture"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=4,
        help="Save state every N ticks (default: 4)"
    )

    args = parser.parse_args()

    config = GSIConfig(
        output_dir=args.output,
        port=args.port,
        save_screenshots=not args.no_screenshots,
        screenshot_interval=args.interval,
    )

    collector = GSIDataCollector(config)

    print("=" * 60)
    print("  CS2 GSI Data Collector")
    print("=" * 60)
    print(f"  Output:      {args.output}")
    print(f"  Port:        {args.port}")
    print(f"  Screenshots: {'Enabled' if config.save_screenshots else 'Disabled'}")
    print(f"  Interval:    Every {args.interval} ticks")
    print("=" * 60)
    print()
    print("  1. Copy gamestate_integration_cs2nn.cfg to CS2 cfg folder")
    print("  2. Launch CS2 and start a match")
    print("  3. Data will be collected automatically")
    print("  4. Press Ctrl+C to stop and save")
    print()
    print("=" * 60)

    try:
        collector.start()

        # Status updates
        while True:
            time.sleep(5)
            stats = collector.stats
            print(f"\r[Stats] Rounds: {stats['rounds']} | "
                  f"States: {stats['states']} | "
                  f"Frames: {stats['frames']} | "
                  f"Current round: {stats['current_round'] or 'N/A'}",
                  end="", flush=True)

    except KeyboardInterrupt:
        print("\n\n[!] Stopping collection...")

    finally:
        collector.stop()
        if collector.stats['states'] > 0:
            output_path = collector.save()
            print(f"\n[OK] Data saved to: {output_path}")
        else:
            print("\n[!] No data collected")


if __name__ == "__main__":
    main()
