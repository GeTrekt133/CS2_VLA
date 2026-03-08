#!/usr/bin/env python3
"""
Install GSI config file to CS2.

Usage:
    python gsi/install_config.py
    python gsi/install_config.py --cs2-path "D:\\Games\\Steam\\..."
"""
import sys
import shutil
import argparse
from pathlib import Path


# Default CS2 cfg paths to try
DEFAULT_PATHS = [
    Path(r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg"),
    Path(r"C:\Program Files\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg"),
    Path(r"D:\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg"),
    Path(r"D:\Games\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg"),
]


def find_cs2_cfg_dir() -> Path | None:
    """Try to find CS2 cfg directory."""
    for path in DEFAULT_PATHS:
        if path.exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Install GSI config to CS2")
    parser.add_argument(
        "--cs2-path",
        type=str,
        help="Custom path to CS2 cfg directory"
    )
    args = parser.parse_args()

    # Find source config
    script_dir = Path(__file__).parent
    source_config = script_dir / "config" / "gamestate_integration_cs2nn.cfg"

    if not source_config.exists():
        print(f"[ERROR] Source config not found: {source_config}")
        sys.exit(1)

    # Find target directory
    if args.cs2_path:
        target_dir = Path(args.cs2_path)
    else:
        target_dir = find_cs2_cfg_dir()

    if target_dir is None or not target_dir.exists():
        print("[ERROR] Could not find CS2 cfg directory")
        print()
        print("Please specify the path manually:")
        print("  python gsi/install_config.py --cs2-path \"C:\\path\\to\\cfg\"")
        print()
        print("The cfg folder is usually at:")
        print("  Steam\\steamapps\\common\\Counter-Strike Global Offensive\\game\\csgo\\cfg")
        sys.exit(1)

    # Copy config
    target_path = target_dir / "gamestate_integration_cs2nn.cfg"

    try:
        shutil.copy(source_config, target_path)
        print(f"[OK] Config installed to:")
        print(f"     {target_path}")
        print()
        print("[!] Restart CS2 for changes to take effect")

    except PermissionError:
        print("[ERROR] Permission denied. Try running as administrator.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to copy: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
