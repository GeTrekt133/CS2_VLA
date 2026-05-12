"""
CS2 Synthetic Data Engine — entry point.

Usage:
  # Generate 100 flick scenarios on dust2 (CS2 must be running)
  python run.py --map de_dust2 --n-flicks 100

  # Generate mixed scenarios
  python run.py --map de_mirage --n-flicks 50 --n-transfers 20 --n-tracking 15 --n-holds 15

  # Dry run (generate scenarios + trajectories without CS2, save metadata only)
  python run.py --dry-run --n-flicks 200

  # Merge with real dataset after collection
  python run.py --merge-only --real-dataset C:/Users/misas/CS2_NN/train_dataset.json

Prerequisites:
  1. CS2 running, console enabled (developer 1, con_enable 1)
  2. On main menu or any map (engine will load the target map)
  3. Run as admin if using Sandboxie
"""

import argparse
import math
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(__file__))

from scenario_generator import ScenarioGenerator
from trajectory_generator import TrajectoryGenerator
from data_exporter import DataExporter


def dry_run(args):
    """Generate scenarios + trajectories without CS2, save metadata JSON."""
    print("=== DRY RUN (no CS2 needed) ===\n")

    gen = ScenarioGenerator(map_name=args.map, seed=args.seed)
    traj_gen = TrajectoryGenerator(seed=args.seed)

    scenarios = gen.generate_batch(
        n_flicks=args.n_flicks,
        n_transfers=args.n_transfers,
        n_tracking=args.n_tracking,
        n_holds=args.n_holds,
    )
    print(f"Generated {len(scenarios)} scenarios")

    # Show distribution
    from collections import Counter
    types = Counter(s.scenario_type for s in scenarios)
    for t, c in types.most_common():
        print(f"  {t}: {c}")

    # Show delta distribution for flicks
    flick_deltas = [abs(s.delta_yaw) for s in scenarios if s.scenario_type == "flick"]
    if flick_deltas:
        import numpy as np
        print(f"\nFlick delta_yaw distribution:")
        print(f"  min={min(flick_deltas):.1f}  max={max(flick_deltas):.1f}  "
              f"mean={np.mean(flick_deltas):.1f}  median={np.median(flick_deltas):.1f}")

    # Generate trajectories and save as dry-run metadata
    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    for i, scenario in enumerate(scenarios):
        trajectory = traj_gen.generate(scenario, n_ticks=args.n_ticks)

        # Build states from trajectory (no frames saved)
        states = []
        tick_base = 1000
        for j, pt in enumerate(trajectory):
            states.append({
                "tick": tick_base + j,
                "keys": list(pt.keys),
                "mouse": [pt.yaw, pt.pitch],
                "detects": [],
                "hp": 100, "armor": 100, "helmet": True,
                "defuser": False, "weapon": "AK-47", "ammo": 30,
                "weapon_list": ["AK-47", "Glock-18", "Knife"],
                "side": "T", "ct_alive": 5, "t_alive": 5,
                "round_time_left": 90, "bomb_planted": False,
                "freeze_time": False, "score": [0, 0],
            })

        scenario_id = f"synth_{i:06d}"
        results.append({
            "scenario_id": scenario_id,
            "scenario_type": scenario.scenario_type,
            "map_name": scenario.map_name,
            "demo_path": os.path.join(args.output_dir, "frames", scenario_id),
            "audio_path": None,
            "states": states,
            "delta_yaw": scenario.delta_yaw,
            "delta_pitch": scenario.delta_pitch,
        })

    exporter = DataExporter(args.output_dir)
    path = exporter.export(results)
    print(f"\nDry-run dataset: {path}")

    # Show sample trajectory
    sample = scenarios[0]
    traj = traj_gen.generate(sample, n_ticks=args.n_ticks)
    print(f"\nSample {sample.scenario_type} trajectory ({len(traj)} ticks):")
    print(f"  Start:  yaw={traj[0].yaw:.2f}  pitch={traj[0].pitch:.2f}  keys={traj[0].keys}")
    mid = len(traj) // 2
    print(f"  Mid:    yaw={traj[mid].yaw:.2f}  pitch={traj[mid].pitch:.2f}  keys={traj[mid].keys}")
    print(f"  End:    yaw={traj[-1].yaw:.2f}  pitch={traj[-1].pitch:.2f}  keys={traj[-1].keys}")
    print(f"  Target: yaw={sample.target_yaw:.2f}  pitch={sample.target_pitch:.2f}")
    print(f"  Delta:  yaw={sample.delta_yaw:.2f}  pitch={sample.delta_pitch:.2f}")


def live_run(args):
    """Run data collection with CS2."""
    from cs2_data_engine import CS2DataEngine

    print("=== LIVE DATA COLLECTION ===\n")

    engine = CS2DataEngine(
        output_dir=args.output_dir,
        timescale=args.timescale,
        tick_delay=args.tick_delay,
        record_audio=not args.no_audio,
    )

    # Connect to CS2
    if not engine.connect():
        print("[ERROR] Could not connect to CS2. Is it running?")
        return

    # Start screen capture
    engine.start_capture()

    # Setup map
    engine.setup_map(args.map)

    # Generate scenarios
    gen = ScenarioGenerator(map_name=args.map, seed=args.seed)
    traj_gen = TrajectoryGenerator(seed=args.seed)

    if args.kills_json:
        scenarios = gen.generate_from_kills(args.kills_json)
        print(f"\nGenerated {len(scenarios)} scenarios from kills")
    else:
        scenarios = gen.generate_batch(
            n_flicks=args.n_flicks,
            n_transfers=args.n_transfers,
            n_tracking=args.n_tracking,
            n_holds=args.n_holds,
        )
        print(f"\nGenerated {len(scenarios)} scenarios")

    # Spawn bots (max needed for any scenario)
    max_bots = max(len(s.bots) for s in scenarios)
    print(f"Spawning {max_bots} bots...")
    engine._spawn_bots(max_bots)
    time.sleep(3)

    # Run all scenarios
    try:
        results = engine.run_batch(scenarios, traj_gen, n_ticks=args.n_ticks)
    except KeyboardInterrupt:
        print("\n[Engine] Interrupted by user")
        results = []
    finally:
        engine.stop()

    # Export dataset
    if results:
        exporter = DataExporter(args.output_dir)
        path = exporter.export(results)
        print(f"\nDataset saved: {path}")

        if args.real_dataset:
            merged = exporter.merge_with_real_dataset(args.real_dataset)
            print(f"Merged dataset: {merged}")


def merge_only(args):
    """Merge existing synthetic dataset with real dataset."""
    exporter = DataExporter(args.output_dir)
    if not args.real_dataset:
        print("[ERROR] --real-dataset required for merge")
        return
    merged = exporter.merge_with_real_dataset(args.real_dataset)
    print(f"Merged dataset: {merged}")


def main():
    parser = argparse.ArgumentParser(description="CS2 Synthetic Data Engine")

    parser.add_argument("--map", default="de_mirage",
                        choices=["de_mirage"],
                        help="CS2 map name")
    parser.add_argument("--output-dir", default="D:/SyntheticDataset",
                        help="Output directory for frames/audio/json")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Scenario counts
    parser.add_argument("--n-flicks", type=int, default=50)
    parser.add_argument("--n-transfers", type=int, default=20)
    parser.add_argument("--n-tracking", type=int, default=15)
    parser.add_argument("--n-holds", type=int, default=15)

    # Trajectory
    parser.add_argument("--n-ticks", type=int, default=64,
                        help="Ticks per scenario (default 64)")

    # Engine
    parser.add_argument("--timescale", type=float, default=0.05,
                        help="host_timescale for slow-mo capture")
    parser.add_argument("--tick-delay", type=float, default=0.15,
                        help="Delay between setang commands (seconds)")
    parser.add_argument("--no-audio", action="store_true",
                        help="Disable audio recording")

    # Modes
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate scenarios without CS2")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge synthetic with real dataset")
    parser.add_argument("--test-setup", action="store_true",
                        help="Test mode: connect, setup map, place 1 bot, stop")
    parser.add_argument("--real-dataset", type=str, default=None,
                        help="Path to real train_dataset.json for merging")
    parser.add_argument("--kills-json", type=str, default=None,
                        help="Path to kill_scenarios.json for real kill-based scenarios")
    parser.add_argument("--preview", action="store_true",
                        help="Fast preview: cycle through kill scenarios, place bot + aim at target")

    args = parser.parse_args()

    if args.merge_only:
        merge_only(args)
    elif args.dry_run:
        dry_run(args)
    elif args.preview:
        preview(args)
    elif args.test_setup:
        test_setup(args)
    else:
        live_run(args)


def preview(args):
    """Fast preview: cycle through kill scenarios, teleport bot + aim at target.

    Shows each scenario for 2 seconds then moves to next.
    Ctrl+C to stop.
    """
    from cs2_data_engine import CS2DataEngine
    from scenario_generator import ScenarioGenerator
    import time

    if not args.kills_json:
        print("ERROR: --kills-json required for preview")
        return

    print("=== PREVIEW MODE ===\n")

    engine = CS2DataEngine(output_dir=args.output_dir, record_audio=False)

    if not engine.connect():
        print("FAILED: CS2 not found")
        return

    engine.setup_map(args.map)
    engine._spawn_bots(1)
    time.sleep(2)

    gen = ScenarioGenerator(map_name=args.map, seed=args.seed)
    scenarios = gen.generate_from_kills(args.kills_json)
    print(f"Loaded {len(scenarios)} scenarios\n")

    try:
        for i, s in enumerate(scenarios):
            # Teleport player to attacker pos, look at target
            engine.cmd.send_batch([
                "noclip 1",
                f"setpos {s.player_pos[0]:.1f} {s.player_pos[1]:.1f} {s.player_pos[2] + 10:.1f}",
                f"setang {s.target_pitch:.2f} {s.target_yaw:.2f} 0",
            ])
            time.sleep(0.2)
            engine.cmd.send("noclip 0")

            # Teleport bot to victim pos with rotation + duck
            bot = s.bots[0]
            engine._place_bot_at(
                *bot.pos,
                yaw=bot.victim_yaw,
                pitch=bot.victim_pitch,
                ducking=bot.ducking,
            )

            print(f"[{i+1}/{len(scenarios)}] dist={math.sqrt(sum((a-b)**2 for a,b in zip(s.player_pos, s.bots[0].pos))):.0f}u "
                  f"yaw={s.target_yaw:.1f} pitch={s.target_pitch:.1f} "
                  f"delta={s.delta_yaw:.1f}")

            time.sleep(2)

    except KeyboardInterrupt:
        print("\nStopped")

    engine.cmd.send("host_timescale 1")
    print("=== PREVIEW DONE ===")


def test_setup(args):
    """Quick test: connect, setup, place bot, run 1 flick. No pauses."""
    from cs2_data_engine import CS2DataEngine
    from scenario_generator import ScenarioGenerator
    from trajectory_generator import TrajectoryGenerator
    import time

    print("=== TEST: 1 flick scenario ===\n")

    engine = CS2DataEngine(
        output_dir=args.output_dir,
        record_audio=False,
    )

    if not engine.connect():
        print("FAILED: CS2 not found")
        return

    engine.setup_map(args.map)
    engine._spawn_bots(1)
    time.sleep(2)

    gen = ScenarioGenerator(map_name=args.map, seed=42)
    traj_gen = TrajectoryGenerator(seed=42)

    if args.kills_json:
        scenarios = gen.generate_from_kills(args.kills_json)
        if not scenarios:
            print("FAILED: no scenarios from kills")
            return
        scenario = scenarios[0]
    else:
        scenario = gen.generate_flick(min_delta_deg=60, max_delta_deg=90)
    if not scenario:
        print("FAILED: no scenario")
        return

    print(f"  Player: ({scenario.player_pos[0]:.0f}, {scenario.player_pos[1]:.0f}, {scenario.player_pos[2]:.0f})")
    print(f"  Bot:    ({scenario.bots[0].pos[0]:.0f}, {scenario.bots[0].pos[1]:.0f}, {scenario.bots[0].pos[2]:.0f})")
    print(f"  Delta:  {scenario.delta_yaw:.1f} yaw, {scenario.delta_pitch:.1f} pitch")

    trajectory = traj_gen.generate(scenario, n_ticks=64)
    engine.start_capture()
    meta = engine.run_scenario(scenario, trajectory, scenario_id="test_flick")
    engine.stop()

    if meta:
        print(f"\nSaved {len(meta['states'])} frames to {meta['demo_path']}")
        exporter = DataExporter(args.output_dir)
        path = exporter.export([meta])
        print(f"Dataset JSON: {path}")
    print("=== TEST DONE ===")


if __name__ == "__main__":
    main()
