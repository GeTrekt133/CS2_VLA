"""
Reparse .dem files and/or merge existing dataset JSONs → train/val split.

Usage examples:

  # Reparse all .dem files from a directory:
  python tools/reparse_and_merge.py reparse \
    --dem-dir "C:/demos/" \
    --steam-id 76561198012345678 \
    --frames-dir /data/msirotkin/demos \
    --output /data/msirotkin/reparsed.json

  # Reparse specific .dem files:
  python tools/reparse_and_merge.py reparse \
    --dem "C:/demos/match1.dem" "C:/demos/match2.dem" \
    --steam-id 76561198012345678 \
    --frames-dir /data/msirotkin/demos \
    --output /data/msirotkin/reparsed.json

  # Merge two existing JSONs and split 90/10:
  python tools/reparse_and_merge.py merge \
    --inputs dataset_a.json dataset_b.json \
    --output /data/msirotkin/merged.json \
    --val-ratio 0.1

  # Reparse + merge with existing:
  python tools/reparse_and_merge.py reparse \
    --dem-dir "C:/demos/" \
    --steam-id 76561198012345678 \
    --frames-dir /data/msirotkin/demos \
    --merge-with existing_dataset.json \
    --output /data/msirotkin/merged.json
"""

import sys
import os
import json
import argparse

# Add universal_demo_parser to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'audio_adaptation', 'data_collect'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_collect_v2'))

import numpy as np
from universal_demo_parser import UniversalDemoParser
from json_builder import DatasetJsonBuilder


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)


def print_stats(label: str, demos: list):
    total_rounds = sum(len(d['rounds']) for d in demos)
    total_states = sum(len(r['states']) for d in demos for r in d['rounds'])
    print(f"  {label}: {len(demos)} demos, {total_rounds} rounds, {total_states:,} states")


def split_train_val(demos: list, val_ratio: float) -> tuple:
    """Split rounds within each demo into train/val."""
    train_demos, val_demos = [], []
    for demo in demos:
        rounds = demo['rounds']
        split = max(1, int(len(rounds) * (1.0 - val_ratio)))
        if split >= len(rounds):
            split = len(rounds) - 1
        train_demos.append(dict(demo, rounds=rounds[:split]))
        val_demos.append(dict(demo, rounds=rounds[split:]))
    return train_demos, val_demos


def merge_demos(demo_lists: list) -> list:
    """Merge multiple lists of demos, deduplicating by demo_path."""
    seen = {}
    for demos in demo_lists:
        for demo in demos:
            key = demo['demo_path']
            if key not in seen:
                seen[key] = demo
            else:
                # Merge rounds from duplicate demo_path
                existing_round_ids = {r['round_id'] for r in seen[key]['rounds']}
                new_rounds = [r for r in demo['rounds'] if r['round_id'] not in existing_round_ids]
                seen[key]['rounds'].extend(new_rounds)
                seen[key]['rounds'].sort(key=lambda r: r['round_id'])
                print(f"  [MERGE] {os.path.basename(key)}: added {len(new_rounds)} new rounds")
    return list(seen.values())


# ──────────────────────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────────────────────

def collect_dem_files(args) -> list:
    """Collect .dem file paths.

    If --dem-dir is set, --dem names are relative to it.
    If --dem-dir is set but --dem is empty, all .dem files in the dir are used.
    """
    if args.dem_dir:
        if args.dem:
            # Names relative to dem_dir
            paths = [os.path.join(args.dem_dir, name) for name in args.dem]
        else:
            # All .dem files in dir
            paths = sorted(
                os.path.join(args.dem_dir, f)
                for f in os.listdir(args.dem_dir)
                if f.endswith('.dem')
            )
            print(f"Found {len(paths)} .dem files in {args.dem_dir}")
    else:
        paths = list(args.dem or [])

    if not paths:
        raise ValueError("Provide --dem and/or --dem-dir")
    return paths


def cmd_reparse(args):
    dem_paths = collect_dem_files(args)
    parser = UniversalDemoParser(target_steam_id=args.steam_id)
    demos = []

    for dem_path in dem_paths:
        dem_path = os.path.abspath(dem_path)
        demo_name = os.path.splitext(os.path.basename(dem_path))[0]
        print(f"\nParsing: {demo_name}")

        try:
            demo_data = parser.parse_demo(dem_path, skip_first_rounds=args.skip_rounds)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        rounds = demo_data.get('rounds', [])
        if not rounds:
            print("  [SKIP] no rounds found")
            continue

        frames_dir = os.path.join(args.frames_dir, demo_name).replace('\\', '/')
        audio_dir  = os.path.join(args.audio_dir, demo_name).replace('\\', '/')

        # Add default drift_info / frame_coverage to rounds that don't have it
        for rnd in rounds:
            if 'drift_info' not in rnd:
                rnd['drift_info'] = {
                    'start_tick_real':    rnd['start_tick'],
                    'end_tick_real':      rnd['end_tick'] - 10,
                    'end_tick_expected':  rnd['end_tick'],
                    'drift_ticks':        -10,
                }
            if 'frame_coverage' not in rnd:
                n = len(rnd.get('states', []))
                rnd['frame_coverage'] = {
                    'frames_saved':    n,
                    'frames_expected': n,
                    'coverage_pct':    100.0,
                }

        demos.append({
            'demo_path':  frames_dir,
            'audio_path': audio_dir,
            'rounds': rounds,
        })
        print(f"  {len(rounds)} rounds -> frames: {frames_dir}")

    if not demos:
        print("[ERROR] No demos parsed successfully.")
        return

    # Merge with existing JSON if provided
    if args.merge_with:
        print(f"\nMerging with {args.merge_with}")
        existing = load_json(args.merge_with).get('demos', [])
        demos = merge_demos([existing, demos])

    _finish(demos, args.output, args.val_ratio)


def cmd_merge(args):
    all_demos = []
    for path in args.inputs:
        print(f"Loading {path}")
        data = load_json(path).get('demos', [])
        print_stats(f"  {os.path.basename(path)}", data)
        all_demos.append(data)

    merged = merge_demos(all_demos)
    _finish(merged, args.output, args.val_ratio)


def _finish(demos: list, output: str, val_ratio: float):
    print(f"\nTotal before split:")
    print_stats("combined", demos)

    train_demos, val_demos = split_train_val(demos, val_ratio)

    train_out = output.replace('.json', '_train.json')
    val_out   = output.replace('.json', '_val.json')

    save_json({'demos': train_demos}, train_out)
    save_json({'demos': val_demos},   val_out)

    print(f"\nSaved:")
    print_stats(train_out, train_demos)
    print_stats(val_out,   val_demos)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Reparse .dem files and/or merge dataset JSONs')
    sub = p.add_subparsers(dest='command', required=True)

    # reparse
    rp = sub.add_parser('reparse', help='Parse .dem files into dataset JSON')
    rp.add_argument('--dem', nargs='+', default=[], help='.dem file paths (explicit list)')
    rp.add_argument('--dem-dir', default=None, help='Directory with .dem files (all will be parsed)')
    rp.add_argument('--steam-id', type=int, required=True, help='Player Steam ID')
    rp.add_argument('--frames-dir', required=True, help='Base dir where tick_*.jpg are stored (subdir = dem name)')
    rp.add_argument('--audio-dir', required=True, help='Base dir where round_*.wav are stored (subdir = dem name)')
    rp.add_argument('--skip-rounds', type=int, default=3, help='Skip first N rounds (warmup)')
    rp.add_argument('--merge-with', default=None, help='Existing dataset JSON to merge into')
    rp.add_argument('--output', required=True, help='Output JSON path (will create _train/_val)')
    rp.add_argument('--val-ratio', type=float, default=0.1, help='Fraction of rounds for val set')

    # merge
    mp = sub.add_parser('merge', help='Merge existing dataset JSONs')
    mp.add_argument('--inputs', nargs='+', required=True, help='Input JSON files')
    mp.add_argument('--output', required=True, help='Output JSON path (will create _train/_val)')
    mp.add_argument('--val-ratio', type=float, default=0.1, help='Fraction of rounds for val set')

    args = p.parse_args()

    if args.command == 'reparse':
        cmd_reparse(args)
    elif args.command == 'merge':
        cmd_merge(args)


if __name__ == '__main__':
    main()
