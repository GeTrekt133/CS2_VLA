"""
Build buy prediction dataset from CS2 demo files.
"""
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from demo_parser import parse_multiple_demos
from features import prepare_dataset
import pandas as pd


# CS2 demos folder
DEMOS_DIR = r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo"
OUTPUT_DIR = Path(__file__).parent / "data"


def find_all_demos(demos_dir: str) -> list:
    """Find all .dem files in directory."""
    demos_path = Path(demos_dir)
    demos = list(demos_path.glob("*.dem"))
    print(f"Found {len(demos)} demo files")
    return [str(d) for d in demos]


def build_dataset():
    """Build and save the buy prediction dataset."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Find demos
    demo_paths = find_all_demos(DEMOS_DIR)

    if not demo_paths:
        print("No demo files found!")
        return

    # Parse demos
    print(f"\nParsing {len(demo_paths)} demos...")
    raw_data = parse_multiple_demos(
        demo_paths,
        output_path=str(OUTPUT_DIR / "raw_buy_data.csv")
    )

    if len(raw_data) == 0:
        print("No data extracted!")
        return

    print(f"\nExtracted {len(raw_data)} player-round samples")

    # Engineer features
    print("\nEngineering features...")
    features, targets = prepare_dataset(raw_data)

    # Save as CSV
    features.to_csv(OUTPUT_DIR / "features.csv", index=False)
    targets.to_csv(OUTPUT_DIR / "targets.csv", index=False)

    print(f"\nDataset saved to {OUTPUT_DIR}")
    print(f"  Features: {features.shape}")
    print(f"  Targets: {targets.shape}")

    # Stats
    print(f"\nBuy type distribution:")
    print(targets['buy_type'].value_counts())

    print(f"\nFeature statistics:")
    print(features[['money', 'team_money_avg', 'round_num', 'score_diff']].describe())

    return features, targets


if __name__ == "__main__":
    build_dataset()
