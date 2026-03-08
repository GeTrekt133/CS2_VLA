"""
Clean dataset JSON — removes states with missing or corrupt frame images.

Usage:
    python clean_dataset.py /path/to/final_dataset_train.json
    python clean_dataset.py /path/to/final_dataset_train.json --check-readable
    python clean_dataset.py /path/to/final_dataset_train.json -o /path/to/cleaned.json
"""

import os
import sys
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def check_exists(path: str) -> bool:
    return os.path.exists(path)


def check_readable(path: str) -> bool:
    """Check that file exists AND can be decoded by cv2."""
    if not os.path.exists(path):
        return False
    try:
        import cv2
        img = cv2.imread(path)
        return img is not None
    except Exception:
        return False


def frame_path(demo_path: str, tick: int) -> str:
    return os.path.join(demo_path, f"tick_{tick}.jpg")


def clean_dataset(input_json: str, output_json: str, do_check_readable: bool = False,
                  workers: int = 8):
    with open(input_json) as f:
        data = json.load(f)

    check_fn = check_readable if do_check_readable else check_exists

    total_states_before = 0
    total_states_after = 0
    total_rounds_before = 0
    total_rounds_removed = 0

    t0 = time.time()

    for demo_idx, demo in enumerate(data['demos']):
        demo_path = demo['demo_path']
        demo_name = os.path.basename(demo_path)

        rounds_to_keep = []
        for rnd in demo['rounds']:
            total_rounds_before += 1
            states = rnd['states']
            total_states_before += len(states)

            # Collect all paths to check
            paths = [frame_path(demo_path, s['tick']) for s in states]

            # Parallel existence check
            valid = [False] * len(paths)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(check_fn, p): i for i, p in enumerate(paths)}
                for fut in as_completed(futures):
                    valid[futures[fut]] = fut.result()

            # Keep only valid states
            clean_states = [s for s, v in zip(states, valid) if v]
            removed = len(states) - len(clean_states)

            if len(clean_states) < 100:
                # Round too short after cleaning — drop entirely
                total_rounds_removed += 1
                if removed > 0:
                    print(f'  [{demo_name}] round {rnd.get("round_id", "?")}:'
                          f' {len(states)} → {len(clean_states)} states — DROPPED (< 100)')
                continue

            if removed > 0:
                print(f'  [{demo_name}] round {rnd.get("round_id", "?")}:'
                      f' {len(states)} → {len(clean_states)} states'
                      f' (removed {removed})')

            rnd['states'] = clean_states
            total_states_after += len(clean_states)
            rounds_to_keep.append(rnd)

        demo['rounds'] = rounds_to_keep
        elapsed = time.time() - t0
        print(f'[{demo_idx+1}/{len(data["demos"])}] {demo_name} done ({elapsed:.1f}s)')

    # Remove demos with no rounds
    data['demos'] = [d for d in data['demos'] if d['rounds']]

    # Save
    with open(output_json, 'w') as f:
        json.dump(data, f, ensure_ascii=False)

    elapsed = time.time() - t0
    removed_states = total_states_before - total_states_after
    print(f'\n--- Summary ---')
    print(f'States:  {total_states_before:>8,} → {total_states_after:>8,}'
          f'  (removed {removed_states:,}, {removed_states/max(total_states_before,1)*100:.1f}%)')
    print(f'Rounds:  {total_rounds_before:>8,} → {total_rounds_before - total_rounds_removed:>8,}'
          f'  (dropped {total_rounds_removed})')
    print(f'Saved:   {output_json}')
    print(f'Time:    {elapsed:.1f}s')


def main():
    parser = argparse.ArgumentParser(description='Clean dataset JSON — remove missing frames')
    parser.add_argument('input', help='Input dataset JSON path')
    parser.add_argument('-o', '--output', help='Output path (default: input with _clean suffix)')
    parser.add_argument('--check-readable', action='store_true',
                        help='Also verify images can be decoded by cv2 (slower)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallel workers for file checks (default: 8)')
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f'{base}_clean{ext}'

    print(f'Input:  {args.input}')
    print(f'Output: {args.output}')
    print(f'Mode:   {"check readable (cv2)" if args.check_readable else "check exists (fast)"}')
    print(f'Workers: {args.workers}')
    print()

    clean_dataset(args.input, args.output, args.check_readable, args.workers)


if __name__ == '__main__':
    main()
