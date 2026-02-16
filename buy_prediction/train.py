"""
Training script for CS2 buy prediction model.
Parses demos, engineers features, and trains the boosting model.
"""
import argparse
import json
from pathlib import Path
from typing import List, Optional
import pandas as pd

from demo_parser import parse_buy_data, parse_multiple_demos
from features import prepare_dataset
from model import BuyPredictor


def find_demo_files(demo_dir: str, pattern: str = "*.dem") -> List[str]:
    """Find all demo files in a directory."""
    demo_path = Path(demo_dir)
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo directory not found: {demo_dir}")

    demos = list(demo_path.glob(pattern))
    print(f"Found {len(demos)} demo files in {demo_dir}")
    return [str(d) for d in demos]


def load_demo_list(json_path: str) -> List[str]:
    """Load demo file paths from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'demos' in data:
        return data['demos']
    else:
        raise ValueError("JSON must be a list of paths or dict with 'demos' key")


def train_from_demos(
    demo_paths: List[str],
    output_dir: str = "./buy_models",
    model_name: str = "buy_predictor_v1",
    cache_data: bool = True,
    num_boost_round: int = 500,
) -> None:
    """
    Full training pipeline: parse demos -> engineer features -> train model.

    Args:
        demo_paths: List of paths to .dem files
        output_dir: Directory to save models
        model_name: Name for saved model
        cache_data: Whether to cache parsed data to disk
        num_boost_round: Number of boosting rounds
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cache_path = output_path / "cached_data.csv"

    # Step 1: Parse demos or load from cache
    if cache_data and cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        raw_data = pd.read_csv(cache_path)
    else:
        print(f"Parsing {len(demo_paths)} demo files...")
        raw_data = parse_multiple_demos(demo_paths)

        if cache_data and len(raw_data) > 0:
            raw_data.to_csv(cache_path, index=False)
            print(f"Cached data to {cache_path}")

    if len(raw_data) == 0:
        print("No data extracted from demos. Exiting.")
        return

    print(f"Total samples: {len(raw_data)}")

    # Step 2: Engineer features
    print("\nEngineering features...")
    features, targets = prepare_dataset(raw_data)

    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"\nBuy type distribution:")
    print(targets['buy_type'].value_counts())

    # Step 3: Train model
    print("\n" + "=" * 60)
    print("TRAINING BUY PREDICTION MODEL")
    print("=" * 60)

    predictor = BuyPredictor(model_dir=output_dir)
    metrics = predictor.train_all(
        features,
        targets,
        num_boost_round=num_boost_round,
    )

    # Step 4: Save model and feature importance
    predictor.save(model_name)

    importance = predictor.feature_importance()
    importance.to_csv(output_path / f"{model_name}_feature_importance.csv", index=False)

    # Save training summary
    summary = {
        'num_demos': len(demo_paths),
        'num_samples': len(raw_data),
        'num_features': len(features.columns),
        'metrics': metrics,
        'top_features': importance.head(20).to_dict('records'),
    }

    with open(output_path / f"{model_name}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModel saved to: {output_path / model_name}")
    print(f"Feature importance: {output_path / f'{model_name}_feature_importance.csv'}")
    print(f"Summary: {output_path / f'{model_name}_summary.json'}")


def train_from_cached(
    cache_path: str,
    output_dir: str = "./buy_models",
    model_name: str = "buy_predictor_v1",
    num_boost_round: int = 500,
) -> None:
    """Train from previously cached data."""
    print(f"Loading data from {cache_path}")
    raw_data = pd.read_csv(cache_path)

    print(f"Total samples: {len(raw_data)}")

    # Engineer features
    print("\nEngineering features...")
    features, targets = prepare_dataset(raw_data)

    # Train
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    predictor = BuyPredictor(model_dir=output_dir)
    metrics = predictor.train_all(features, targets, num_boost_round=num_boost_round)

    predictor.save(model_name)

    print("\nTraining complete!")
    print(f"Model saved to: {output_path / model_name}")


def main():
    parser = argparse.ArgumentParser(description="Train CS2 buy prediction model")

    parser.add_argument(
        '--demo-dir',
        type=str,
        help='Directory containing .dem files'
    )
    parser.add_argument(
        '--demo-list',
        type=str,
        help='JSON file with list of demo paths'
    )
    parser.add_argument(
        '--demo',
        type=str,
        action='append',
        help='Single demo file path (can be used multiple times)'
    )
    parser.add_argument(
        '--cached-data',
        type=str,
        help='Path to cached CSV data (skip parsing)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./buy_models',
        help='Output directory for models'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='buy_predictor_v1',
        help='Name for saved model'
    )
    parser.add_argument(
        '--num-rounds',
        type=int,
        default=500,
        help='Number of boosting rounds'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Do not cache parsed data'
    )

    args = parser.parse_args()

    # Collect demo paths
    demo_paths = []

    if args.cached_data:
        train_from_cached(
            args.cached_data,
            args.output_dir,
            args.model_name,
            args.num_rounds,
        )
        return

    if args.demo_dir:
        demo_paths.extend(find_demo_files(args.demo_dir))

    if args.demo_list:
        demo_paths.extend(load_demo_list(args.demo_list))

    if args.demo:
        demo_paths.extend(args.demo)

    if not demo_paths:
        print("No demo files specified. Use --demo-dir, --demo-list, or --demo")
        print("\nExample usage:")
        print("  python train.py --demo-dir /path/to/demos")
        print("  python train.py --demo /path/to/demo1.dem --demo /path/to/demo2.dem")
        print("  python train.py --demo-list demos.json")
        print("  python train.py --cached-data ./buy_models/cached_data.csv")
        return

    # Remove duplicates
    demo_paths = list(set(demo_paths))

    train_from_demos(
        demo_paths,
        args.output_dir,
        args.model_name,
        cache_data=not args.no_cache,
        num_boost_round=args.num_rounds,
    )


if __name__ == "__main__":
    main()
