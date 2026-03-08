"""
Train CS2 buy prediction model.

Usage:
    python train.py --demo-dir D:/DemosCS
    python train.py --cached-data ./data/raw_data.csv
"""
import argparse
import json
from pathlib import Path
import pandas as pd

from demo_parser import parse_multiple_demos
from features import prepare_dataset
from model import BuyPredictor


def train(
    demo_paths: list = None,
    cached_data: str = None,
    output_dir: str = "./buy_models",
    model_name: str = "buy_v2",
    num_rounds: int = 500,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "cached_data.csv"

    # Load data
    if cached_data:
        print(f"Loading cached: {cached_data}")
        raw = pd.read_csv(cached_data)
    elif cache_path.exists():
        print(f"Loading cache: {cache_path}")
        raw = pd.read_csv(cache_path)
    elif demo_paths:
        print(f"Parsing {len(demo_paths)} demos...")
        raw = parse_multiple_demos(demo_paths)
        if len(raw) > 0:
            raw.to_csv(cache_path, index=False)
    else:
        print("No data source. Use --demo-dir or --cached-data")
        return

    if len(raw) == 0:
        print("No data")
        return

    print(f"\nSamples: {len(raw)}")

    # Features
    features, targets = prepare_dataset(raw)
    print(f"Features: {features.shape}, Targets: {targets.shape}")
    print(f"\nPrimary distribution:\n{targets['primary'].value_counts().sort_index()}")

    # Train
    predictor = BuyPredictor(model_dir=output_dir)
    metrics = predictor.train(features, targets, num_boost_round=num_rounds)

    # Save
    predictor.save(model_name)

    # Feature importance
    imp = predictor.feature_importance('primary')
    if len(imp) > 0:
        imp.to_csv(out / f"{model_name}_importance.csv", index=False)
        print(f"\nTop features (primary):")
        print(imp.head(10).to_string(index=False))

    # Summary
    summary = {
        'samples': len(raw),
        'features': len(features.columns),
        'metrics': {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in metrics.items()},
    }
    with open(out / f"{model_name}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Model saved to {out / model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo-dir', type=str, help='Directory with .dem files')
    parser.add_argument('--cached-data', type=str, help='Pre-parsed CSV')
    parser.add_argument('--output-dir', type=str, default='./buy_models')
    parser.add_argument('--model-name', type=str, default='buy_v2')
    parser.add_argument('--num-rounds', type=int, default=500)
    args = parser.parse_args()

    demo_paths = []
    if args.demo_dir:
        demo_paths = sorted(str(p) for p in Path(args.demo_dir).glob("*.dem"))

    train(
        demo_paths=demo_paths or None,
        cached_data=args.cached_data,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_rounds=args.num_rounds,
    )


if __name__ == "__main__":
    main()
