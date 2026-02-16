"""
Gradient Boosting model for CS2 buy prediction.
Uses LightGBM for fast training and inference.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import pickle

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not installed. Run: pip install lightgbm")

try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sklearn.preprocessing import LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Scikit-learn not installed. Run: pip install scikit-learn")


class BuyPredictor:
    """
    Gradient Boosting model for predicting buy decisions in CS2.

    Predicts:
    - buy_type: eco, force_buy, half_buy, full_buy (multiclass)
    - Individual item categories (binary classifiers)
    """

    def __init__(self, model_dir: str = "./buy_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.buy_type_model: Optional[lgb.Booster] = None
        self.rifle_model: Optional[lgb.Booster] = None
        self.armor_model: Optional[lgb.Booster] = None
        self.awp_model: Optional[lgb.Booster] = None

        self.feature_names: List[str] = []
        self.label_encoder = LabelEncoder()

        # Default LightGBM parameters
        self.lgb_params = {
            'objective': 'multiclass',
            'num_class': 4,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42,
        }

        self.binary_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42,
        }

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns (exclude identifiers)."""
        exclude = ['steamid', 'demo_id', 'demo_path']
        return [c for c in df.columns if c not in exclude]

    def train_buy_type(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
        val_size: float = 0.2,
    ) -> Dict[str, float]:
        """
        Train the buy type classifier.

        Args:
            X: Feature DataFrame
            y: Target Series (buy_type_encoded: 0-3)
            num_boost_round: Number of boosting rounds
            early_stopping_rounds: Early stopping patience
            val_size: Validation split size

        Returns:
            Dictionary with training metrics
        """
        if not HAS_LGB:
            raise ImportError("LightGBM is required for training")

        self.feature_names = self._get_feature_columns(X)
        X_feat = X[self.feature_names]

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_feat, y, test_size=val_size, random_state=42, stratify=y
        )

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=100),
        ]

        self.buy_type_model = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks,
        )

        # Evaluate
        y_pred = np.argmax(self.buy_type_model.predict(X_val), axis=1)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')

        print(f"\nBuy Type Model - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        print("\nClassification Report:")
        buy_type_names = ['eco', 'force_buy', 'half_buy', 'full_buy']
        unique_labels = sorted(set(y_val) | set(y_pred))
        labels_in_data = [buy_type_names[i] for i in unique_labels if i < len(buy_type_names)]
        print(classification_report(y_val, y_pred, labels=unique_labels, target_names=labels_in_data))

        return {'accuracy': accuracy, 'f1': f1}

    def train_binary_classifier(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        num_boost_round: int = 300,
        early_stopping_rounds: int = 30,
        val_size: float = 0.2,
    ) -> Tuple[lgb.Booster, Dict[str, float]]:
        """Train a binary classifier for a specific item category."""
        if not HAS_LGB:
            raise ImportError("LightGBM is required for training")

        feature_cols = self._get_feature_columns(X)
        X_feat = X[feature_cols]

        X_train, X_val, y_train, y_val = train_test_split(
            X_feat, y, test_size=val_size, random_state=42, stratify=y if y.sum() > 10 else None
        )

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=100),
        ]

        model = lgb.train(
            self.binary_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks,
        )

        # Evaluate
        y_pred = (model.predict(X_val) > 0.5).astype(int)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        print(f"\n{model_name} Model - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        return model, {'accuracy': accuracy, 'f1': f1}

    def train_all(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all models.

        Args:
            features: Feature DataFrame from features.py
            targets: Target DataFrame from features.py

        Returns:
            Dictionary with metrics for each model
        """
        metrics = {}

        # Train main buy type classifier
        print("=" * 50)
        print("Training Buy Type Classifier")
        print("=" * 50)
        metrics['buy_type'] = self.train_buy_type(
            features, targets['buy_type_encoded'], **kwargs
        )

        # Train binary classifiers
        print("\n" + "=" * 50)
        print("Training Rifle Classifier")
        print("=" * 50)
        self.rifle_model, metrics['rifle'] = self.train_binary_classifier(
            features, targets['bought_rifle'], 'Rifle', **kwargs
        )

        print("\n" + "=" * 50)
        print("Training Armor Classifier")
        print("=" * 50)
        self.armor_model, metrics['armor'] = self.train_binary_classifier(
            features, targets['bought_armor'], 'Armor', **kwargs
        )

        print("\n" + "=" * 50)
        print("Training AWP/Sniper Classifier")
        print("=" * 50)
        self.awp_model, metrics['awp'] = self.train_binary_classifier(
            features, targets['bought_sniper'], 'AWP', **kwargs
        )

        return metrics

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions for all models.

        Args:
            X: Feature DataFrame

        Returns:
            Dictionary with predictions:
            - buy_type: class indices (0-3)
            - buy_type_probs: class probabilities (N, 4)
            - rifle_prob: probability of buying rifle
            - armor_prob: probability of buying armor
            - awp_prob: probability of buying AWP
        """
        X_feat = X[self.feature_names]
        predictions = {}

        if self.buy_type_model is not None:
            probs = self.buy_type_model.predict(X_feat)
            predictions['buy_type_probs'] = probs
            predictions['buy_type'] = np.argmax(probs, axis=1)

        if self.rifle_model is not None:
            predictions['rifle_prob'] = self.rifle_model.predict(X_feat)

        if self.armor_model is not None:
            predictions['armor_prob'] = self.armor_model.predict(X_feat)

        if self.awp_model is not None:
            predictions['awp_prob'] = self.awp_model.predict(X_feat)

        return predictions

    def get_buy_recommendation(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Get buy recommendations with explanations.

        Returns list of recommendations for each sample:
        - recommended_buy: 'eco', 'force_buy', 'half_buy', 'full_buy'
        - confidence: confidence score
        - should_buy_rifle: bool
        - should_buy_armor: bool
        - suggested_items: list of suggested items
        """
        predictions = self.predict(X)
        recommendations = []

        buy_types = ['eco', 'force_buy', 'half_buy', 'full_buy']

        for i in range(len(X)):
            rec = {}

            # Buy type
            buy_idx = predictions['buy_type'][i]
            rec['recommended_buy'] = buy_types[buy_idx]
            rec['confidence'] = float(predictions['buy_type_probs'][i, buy_idx])

            # Item recommendations
            rec['should_buy_rifle'] = bool(predictions.get('rifle_prob', [0])[i] > 0.5)
            rec['should_buy_armor'] = bool(predictions.get('armor_prob', [0])[i] > 0.5)
            rec['should_buy_awp'] = bool(predictions.get('awp_prob', [0])[i] > 0.5)

            # Suggest items based on predictions and money
            money = X.iloc[i].get('money', 0)
            suggested = []

            if rec['recommended_buy'] == 'full_buy':
                if rec['should_buy_awp'] and money >= 5750:
                    suggested.extend(['awp', 'vesthelm'])
                elif rec['should_buy_rifle']:
                    if X.iloc[i].get('is_t_side', 0):
                        suggested.append('ak47')
                    else:
                        suggested.append('m4a1_silencer')
                    suggested.append('vesthelm')
                suggested.extend(['flashbang', 'smokegrenade', 'hegrenade'])

            elif rec['recommended_buy'] == 'half_buy':
                suggested.extend(['mp9' if X.iloc[i].get('is_ct_side', 0) else 'mac10', 'vest'])

            elif rec['recommended_buy'] == 'force_buy':
                suggested.extend(['p250', 'flashbang'])

            rec['suggested_items'] = suggested
            recommendations.append(rec)

        return recommendations

    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance from buy type model."""
        if self.buy_type_model is None:
            return pd.DataFrame()

        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.buy_type_model.feature_importance(importance_type='gain'),
        })
        return importance.sort_values('importance', ascending=False)

    def save(self, name: str = "buy_predictor"):
        """Save all models and metadata."""
        save_dir = self.model_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save models
        if self.buy_type_model is not None:
            self.buy_type_model.save_model(str(save_dir / "buy_type.lgb"))

        if self.rifle_model is not None:
            self.rifle_model.save_model(str(save_dir / "rifle.lgb"))

        if self.armor_model is not None:
            self.armor_model.save_model(str(save_dir / "armor.lgb"))

        if self.awp_model is not None:
            self.awp_model.save_model(str(save_dir / "awp.lgb"))

        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'lgb_params': self.lgb_params,
            'binary_params': self.binary_params,
        }
        with open(save_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Models saved to {save_dir}")

    def load(self, name: str = "buy_predictor"):
        """Load all models and metadata."""
        load_dir = self.model_dir / name

        if not load_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {load_dir}")

        # Load metadata
        with open(load_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)

        self.feature_names = metadata['feature_names']
        self.lgb_params = metadata['lgb_params']
        self.binary_params = metadata['binary_params']

        # Load models
        buy_type_path = load_dir / "buy_type.lgb"
        if buy_type_path.exists():
            self.buy_type_model = lgb.Booster(model_file=str(buy_type_path))

        rifle_path = load_dir / "rifle.lgb"
        if rifle_path.exists():
            self.rifle_model = lgb.Booster(model_file=str(rifle_path))

        armor_path = load_dir / "armor.lgb"
        if armor_path.exists():
            self.armor_model = lgb.Booster(model_file=str(armor_path))

        awp_path = load_dir / "awp.lgb"
        if awp_path.exists():
            self.awp_model = lgb.Booster(model_file=str(awp_path))

        print(f"Models loaded from {load_dir}")


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing BuyPredictor with synthetic data...")

    # Create synthetic features
    np.random.seed(42)
    n_samples = 1000

    features = pd.DataFrame({
        'money': np.random.randint(0, 16000, n_samples),
        'money_log': np.log1p(np.random.randint(0, 16000, n_samples)),
        'team_money_total': np.random.randint(0, 80000, n_samples),
        'team_money_avg': np.random.randint(0, 16000, n_samples),
        'can_full_buy': np.random.randint(0, 2, n_samples),
        'can_awp': np.random.randint(0, 2, n_samples),
        'can_half_buy': np.random.randint(0, 2, n_samples),
        'money_for_save': np.random.randint(0, 2, n_samples),
        'team_can_full_buy': np.random.randint(0, 2, n_samples),
        'team_poor': np.random.randint(0, 2, n_samples),
        'money_diff_from_team': np.random.randint(-5000, 5000, n_samples),
        'money_ratio_to_team': np.random.uniform(0.5, 2.0, n_samples),
        'teammate_money_min': np.random.randint(0, 10000, n_samples),
        'teammate_money_max': np.random.randint(0, 16000, n_samples),
        'teammate_money_std': np.random.uniform(0, 3000, n_samples),
        'teammates_can_buy': np.random.randint(0, 5, n_samples),
        'round_num': np.random.randint(1, 25, n_samples),
        'score_t': np.random.randint(0, 13, n_samples),
        'score_ct': np.random.randint(0, 13, n_samples),
        'score_diff': np.random.randint(-12, 13, n_samples),
        'total_rounds_played': np.random.randint(0, 25, n_samples),
        'is_winning': np.random.randint(0, 2, n_samples),
        'is_losing': np.random.randint(0, 2, n_samples),
        'is_tied': np.random.randint(0, 2, n_samples),
        'is_first_half': np.random.randint(0, 2, n_samples),
        'is_second_half': np.random.randint(0, 2, n_samples),
        'rounds_until_half': np.random.randint(0, 12, n_samples),
        'is_last_round_first_half': np.random.randint(0, 2, n_samples),
        'is_last_round_second_half': np.random.randint(0, 2, n_samples),
        'is_t_side': np.random.randint(0, 2, n_samples),
        'is_ct_side': np.random.randint(0, 2, n_samples),
        'round_after_pistol': np.random.randint(0, 2, n_samples),
        'is_pistol_round': np.random.randint(0, 2, n_samples),
    })

    # Create correlated targets (more realistic)
    targets = pd.DataFrame({
        'buy_type_encoded': np.where(
            features['money'] < 1500, 0,
            np.where(features['money'] < 2500, 1,
            np.where(features['money'] < 4000, 2, 3))
        ),
        'bought_rifle': (features['money'] >= 2700).astype(int),
        'bought_sniper': ((features['money'] >= 4750) & (np.random.random(n_samples) > 0.7)).astype(int),
        'bought_armor': (features['money'] >= 1000).astype(int),
    })

    # Train
    predictor = BuyPredictor(model_dir="./test_buy_models")
    metrics = predictor.train_all(features, targets, num_boost_round=100)

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print("\nMetrics:", metrics)

    # Feature importance
    print("\nTop 10 Important Features:")
    print(predictor.feature_importance().head(10))

    # Test prediction
    test_sample = features.head(5)
    recommendations = predictor.get_buy_recommendation(test_sample)
    print("\nSample Recommendations:")
    for i, rec in enumerate(recommendations):
        print(f"  Sample {i}: {rec['recommended_buy']} (conf: {rec['confidence']:.2f})")
        print(f"    Suggested items: {rec['suggested_items']}")

    # Save model
    predictor.save("test_model")
