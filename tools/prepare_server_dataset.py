"""
Prepare dataset JSON for server training.

Usage:
    python tools/prepare_server_dataset.py \
        --input dataset_test.json \
        --output /tmp/server_dataset.json \
        --server-base /mnt/ml/cs2_data/demos \
        --local-data "C:/Users/misas/Downloads/vla_output/vla_output"

This rewrites demo_path and audio_path to server-side paths,
assuming all demo folders are copied to --server-base.
"""

import json
import os
import argparse


def prepare_dataset(input_json: str, output_json: str,
                    server_base: str, local_data: str):
    with open(input_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    updated_demos = []
    for demo in dataset['demos']:
        demo_id = os.path.basename(demo['demo_path'])
        local_demo_dir = os.path.join(local_data, demo_id)

        # Check if this demo's frames exist locally
        if not os.path.isdir(local_demo_dir):
            print(f"[SKIP] {demo_id} — folder not found at {local_demo_dir}")
            continue

        # Count frames and audio
        try:
            entries = os.listdir(local_demo_dir)
            n_frames = sum(1 for e in entries if e.endswith('.jpg'))
            n_audio  = sum(1 for e in entries if e.endswith('.wav'))
            total_mb = sum(os.path.getsize(os.path.join(local_demo_dir, e))
                           for e in entries) / 1024**2
        except Exception:
            n_frames = n_audio = 0
            total_mb = 0

        # Server path — both demo_path and audio_path point to same dir
        server_dir = os.path.join(server_base, demo_id).replace('\\', '/')

        new_demo = dict(demo)
        new_demo['demo_path']  = server_dir
        new_demo['audio_path'] = server_dir  # same flat dir has round_*.wav
        updated_demos.append(new_demo)

        print(f"[OK] {demo_id}")
        print(f"     frames: {n_frames:,}  audio: {n_audio}  size: {total_mb:.0f} MB")
        print(f"     -> {server_dir}")

    if not updated_demos:
        print("\n[ERROR] No demos found. Check --local-data path.")
        return

    # Split 90/10 train/val by round (same demo, different rounds)
    train_demos, val_demos = [], []
    for demo in updated_demos:
        rounds = demo['rounds']
        split = max(1, int(len(rounds) * 0.9))
        train_demo = dict(demo, rounds=rounds[:split])
        val_demo   = dict(demo, rounds=rounds[split:])
        train_demos.append(train_demo)
        val_demos.append(val_demo)

    train_out = output_json.replace('.json', '_train.json')
    val_out   = output_json.replace('.json', '_val.json')

    for path, demos in [(train_out, train_demos), (val_out, val_demos)]:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'demos': demos}, f, indent=2, ensure_ascii=False)
        total_rounds = sum(len(d['rounds']) for d in demos)
        total_states = sum(len(r['states']) for d in demos for r in d['rounds'])
        print(f"\nSaved: {path}")
        print(f"  demos: {len(demos)}  rounds: {total_rounds}  states: {total_states:,}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',       default='dataset_test.json')
    parser.add_argument('--output',      default='server_dataset.json')
    parser.add_argument('--server-base', default='/mnt/ml/cs2_data/demos',
                        help='Base dir on server where demo folders will live')
    parser.add_argument('--local-data',
                        default=r'C:\Users\misas\Downloads\vla_output\vla_output',
                        help='Local dir containing demo folders')
    args = parser.parse_args()
    prepare_dataset(args.input, args.output, args.server_base, args.local_data)
