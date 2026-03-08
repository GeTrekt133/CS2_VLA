"""
LightGBM model for CS2 exact buy prediction.

Predicts 7 outputs:
  - primary weapon: multiclass (none/pistol/smg/rifle/awp/scout)
  - armor: multiclass (none/vest/vesthelm)
  - smoke, flash, HE, molotov, defuser: binary each
"""
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


MULTICLASS_TARGETS = {
    'primary': 6,   # none, pistol_upgrade, smg, rifle, awp, scout
    'armor': 3,     # none, vest, vesthelm
}

BINARY_TARGETS = ['buy_smoke', 'buy_flash', 'buy_he', 'buy_molotov', 'buy_defuser']


class BuyPredictor:
    """Multi-output LightGBM for exact buy prediction."""

    def __init__(self, model_dir: str = "./buy_models"):
        self.model_dir = Path(model_dir)
        self.models: Dict[str, lgb.Booster] = {}
        self.feature_names: List[str] = []

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        exclude = {'steamid', 'demo_id', 'demo_path'}
        return [c for c in df.columns if c not in exclude]

    def train(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        num_boost_round: int = 500,
        early_stopping: int = 50,
        val_size: float = 0.2,
    ) -> Dict[str, Dict[str, float]]:
        """Train all sub-models."""
        if not HAS_LGB or not HAS_SKLEARN:
            raise ImportError("lightgbm and scikit-learn required")

        self.feature_names = self._get_feature_cols(features)
        X = features[self.feature_names]
        metrics = {}

        # Multiclass targets
        for name, num_class in MULTICLASS_TARGETS.items():
            print(f"\n{'='*50}")
            print(f"Training: {name} ({num_class} classes)")
            print(f"{'='*50}")

            y = targets[name].values
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=val_size, random_state=42, stratify=y
            )

            params = {
                'objective': 'multiclass',
                'num_class': num_class,
                'metric': 'multi_logloss',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': 42,
            }

            ds_tr = lgb.Dataset(X_tr, label=y_tr)
            ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr)

            model = lgb.train(
                params, ds_tr,
                num_boost_round=num_boost_round,
                valid_sets=[ds_tr, ds_val],
                valid_names=['train', 'val'],
                callbacks=[
                    lgb.early_stopping(early_stopping),
                    lgb.log_evaluation(100),
                ],
            )

            y_pred = np.argmax(model.predict(X_val), axis=1)
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            print(f"\n{name}: accuracy={acc:.4f}, f1={f1:.4f}")
            print(classification_report(y_val, y_pred, zero_division=0))

            self.models[name] = model
            metrics[name] = {'accuracy': acc, 'f1': f1}

        # Binary targets
        for name in BINARY_TARGETS:
            print(f"\n{'='*50}")
            print(f"Training: {name} (binary)")
            print(f"{'='*50}")

            y = targets[name].values
            pos_rate = y.mean()
            print(f"  Positive rate: {pos_rate:.3f}")

            if y.sum() < 5:
                print(f"  Skipping {name} — too few positives")
                continue

            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=val_size, random_state=42,
                stratify=y if y.sum() > 10 else None
            )

            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': 42,
                'is_unbalance': True,
            }

            ds_tr = lgb.Dataset(X_tr, label=y_tr)
            ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr)

            model = lgb.train(
                params, ds_tr,
                num_boost_round=num_boost_round,
                valid_sets=[ds_tr, ds_val],
                valid_names=['train', 'val'],
                callbacks=[
                    lgb.early_stopping(early_stopping),
                    lgb.log_evaluation(100),
                ],
            )

            y_pred = (model.predict(X_val) > 0.5).astype(int)
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            print(f"\n{name}: accuracy={acc:.4f}, f1={f1:.4f}")

            self.models[name] = model
            metrics[name] = {'accuracy': acc, 'f1': f1}

        return metrics

    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Predict all outputs. Returns dict of predictions."""
        X_feat = X[self.feature_names]
        preds = {}

        for name, num_class in MULTICLASS_TARGETS.items():
            if name in self.models:
                probs = self.models[name].predict(X_feat)
                preds[f'{name}_probs'] = probs
                preds[name] = np.argmax(probs, axis=1)

        for name in BINARY_TARGETS:
            if name in self.models:
                preds[f'{name}_prob'] = self.models[name].predict(X_feat)
                preds[name] = (preds[f'{name}_prob'] > 0.5).astype(int)

        return preds

    def feature_importance(self, model_name: str = 'primary') -> pd.DataFrame:
        if model_name not in self.models:
            return pd.DataFrame()
        imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.models[model_name].feature_importance(importance_type='gain'),
        })
        return imp.sort_values('importance', ascending=False)

    def save(self, name: str = "buy_v2"):
        save_dir = self.model_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model in self.models.items():
            model.save_model(str(save_dir / f"{model_name}.lgb"))

        meta = {'feature_names': self.feature_names}
        with open(save_dir / "metadata.json", 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"Saved {len(self.models)} models to {save_dir}")

    def load(self, name: str = "buy_v2"):
        load_dir = self.model_dir / name
        if not load_dir.exists():
            raise FileNotFoundError(f"Not found: {load_dir}")

        with open(load_dir / "metadata.json") as f:
            meta = json.load(f)
        self.feature_names = meta['feature_names']

        for model_file in load_dir.glob("*.lgb"):
            model_name = model_file.stem
            self.models[model_name] = lgb.Booster(model_file=str(model_file))

        print(f"Loaded {len(self.models)} models from {load_dir}")
