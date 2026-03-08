"""
Build buy prediction dataset from CS2 demo files.

Usage:
    python build_dataset.py --demos D:/DemosCS --out ./data
"""
import argparse
from pathlib import Path

from demo_parser import parse_multiple_demos
from features import prepare_dataset


def build_dataset(demos_dir: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Find demos
    demos = sorted(Path(demos_dir).glob("*.dem"))
    print(f"Found {len(demos)} demo files in {demos_dir}")
    if not demos:
        return

    # Parse
    raw = parse_multiple_demos(
        [str(d) for d in demos],
        output_path=str(out / "raw_data.csv"),
    )
    if len(raw) == 0:
        print("No data extracted")
        return

    # Features + targets
    features, targets = prepare_dataset(raw)
    features.to_csv(out / "features.csv", index=False)
    targets.to_csv(out / "targets.csv", index=False)

    print(f"\nDataset: {len(features)} samples")
    print(f"Features: {features.shape[1]} cols")
    print(f"\nPrimary weapon distribution:")
    print(targets['primary'].value_counts().sort_index())
    print(f"\nArmor distribution:")
    print(targets['armor'].value_counts().sort_index())
    print(f"\nGrenade rates:")
    for col in ['buy_smoke', 'buy_flash', 'buy_he', 'buy_molotov', 'buy_defuser']:
        print(f"  {col}: {targets[col].mean():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--demos', default='D:/DemosCS')
    parser.add_argument('--out', default='./data')
    args = parser.parse_args()
    build_dataset(args.demos, args.out)
