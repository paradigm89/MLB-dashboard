"""
Batter vs. Pitcher Matchup Model: predicts xwOBA per plate appearance.

Model: LightGBM Regressor (leaf-wise growth, fast on millions of pitch rows).

Target: estimated_woba_using_speedangle from Statcast (xwOBA per contact event).
This is better than observed wOBA because:
- Contact quality adjusted (exit velocity + launch angle)
- Removes luck on batted ball outcomes
- More stable predictor of future performance

Features are Bayesian-blended in matchups.py before reaching this model.
The model learns non-linear interactions (e.g., batter exit velocity
interacting with pitcher spin rate) that Bayesian shrinkage alone can't capture.
"""
import logging
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from src.models.base import BaseMLBModel

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "models.yaml"

_NON_FEATURE_COLS = [
    "batter_id", "pitcher_id", "game_pk", "game_date",
    "xwoba_actual", "woba_actual",
]


def load_model_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f).get("batter_matchup", {})


class LGBMBatterMatchup(BaseMLBModel):
    """
    LightGBM regressor for per-PA xwOBA prediction.
    Trained on Statcast pitch-level data (one row per plate appearance).
    """

    model_type = "batter_matchup"

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        self._config = config or load_model_config()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMBatterMatchup":
        """
        Args:
            X: feature matrix (one row per plate appearance)
            y: xwOBA for each PA (from estimated_woba_using_speedangle)
        """
        params = self._config.get("params", {})
        feature_cols = [c for c in X.columns if c not in _NON_FEATURE_COLS]
        self._feature_names = feature_cols
        X_feat = X[feature_cols]

        # LightGBM handles NaN natively; no imputation needed
        y_clean = y.fillna(y.median())

        logger.info("Training batter matchup model on %d samples", len(X_feat))

        train_data = lgb.Dataset(X_feat, label=y_clean, feature_name=feature_cols)

        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "n_estimators": params.get("n_estimators", 600),
            "learning_rate": params.get("learning_rate", 0.03),
            "max_depth": params.get("max_depth", 7),
            "num_leaves": params.get("num_leaves", 63),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.8),
            "min_child_samples": params.get("min_child_samples", 20),
            "reg_alpha": params.get("reg_alpha", 0.1),
            "reg_lambda": params.get("reg_lambda", 1.0),
            "verbose": params.get("verbose", -1),
            "n_jobs": -1,
            "random_state": 42,
        }

        callbacks = [lgb.early_stopping(params.get("early_stopping_rounds", 50), verbose=False),
                     lgb.log_evaluation(period=-1)]

        self._model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[train_data],
            callbacks=callbacks,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns predicted xwOBA values, clipped to [0.0, 1.0]."""
        X_feat = self._align_features(X)
        preds = self._model.predict(X_feat)
        return np.clip(preds, 0.0, 1.0)

    def predict_single(self, features: dict) -> float:
        """Predict xwOBA for a single batter-pitcher matchup feature dict."""
        X = pd.DataFrame([features])
        return float(self.predict(X)[0])

    def save(self, path: Path) -> None:
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        # LightGBM Booster can't be saved with joblib directly; save both
        model_path = path.with_suffix(".lgb")
        self._model.save_model(str(model_path))
        joblib.dump({
            "lgb_model_path": str(model_path),
            "feature_names": self._feature_names,
        }, path)
        self._artifact_path = path

    def load(self, path: Path) -> "LGBMBatterMatchup":
        import joblib
        artifact = joblib.load(path)
        self._model = lgb.Booster(model_file=artifact["lgb_model_path"])
        self._feature_names = artifact["feature_names"]
        self._artifact_path = path
        return self
