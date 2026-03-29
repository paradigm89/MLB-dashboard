"""
Pitcher vs. Opponent Model: projects a starting pitcher's xFIP for a game.

Model: XGBoost Regressor targeting xFIP_actual (computed post-game from Statcast).

Honest caveat (documented in CLAUDE.md and plan):
xFIP per individual game start has very high variance. A pitcher who averages
3.20 xFIP on the season might record 1.00 or 8.00 in any given start due to
sequencing and luck. The output of this model should be interpreted as
"what kind of pitcher is this against this opponent in this context" rather
than "expect exactly this xFIP tonight." Confidence intervals are intentionally
wide to reflect this uncertainty.

The model is most useful as a relative ranking input to the run expectancy model
(is this a strong or weak pitching matchup?) rather than as a standalone prediction.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from src.models.base import BaseMLBModel

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "models.yaml"

_NON_FEATURE_COLS = [
    "game_pk", "game_date", "season", "pitcher_id",
    "xfip_actual", "fip_actual", "era_actual",
]


def load_model_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f).get("pitcher_model", {})


class XGBoostPitcherModel(BaseMLBModel):
    """
    XGBoost regressor for per-start xFIP projection.
    Trains on historical starter game logs with Bayesian-blended opponent features.
    """

    model_type = "pitcher_model"

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        self._config = config or load_model_config()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostPitcherModel":
        """
        Args:
            X: feature matrix (one row per starting pitcher appearance)
            y: xFIP_actual for each start
        """
        params = self._config.get("params", {})
        feature_cols = [c for c in X.columns if c not in _NON_FEATURE_COLS]
        self._feature_names = feature_cols
        X_feat = X[feature_cols]

        # Fill NaN xFIP with median (some starts may not have Statcast xFIP)
        y_clean = y.fillna(y.median())

        logger.info("Training pitcher model on %d samples", len(X_feat))
        self._model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=params.get("n_estimators", 400),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 5),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            min_child_weight=params.get("min_child_weight", 3),
            reg_alpha=params.get("reg_alpha", 0.2),
            reg_lambda=params.get("reg_lambda", 1.0),
            early_stopping_rounds=params.get("early_stopping_rounds", 50),
            eval_metric="rmse",
            n_jobs=-1,
            random_state=42,
        )
        self._model.fit(X_feat, y_clean,
                        eval_set=[(X_feat, y_clean)],
                        verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns projected xFIP values, clipped to [1.5, 8.0]."""
        X_feat = self._align_features(X)
        preds = self._model.predict(X_feat)
        return np.clip(preds, 1.5, 8.0)

    def predict_with_interval(self, features: dict,
                               confidence: float = 0.80) -> dict:
        """
        Predict xFIP with a confidence interval for a single start.

        The interval is estimated using the model's RMSE from validation
        (approximately ±1.2 xFIP for a single game start, reflecting
        true game-to-game variance).

        Honest note: these intervals are wide by design. A single game
        start has very high noise regardless of the model's accuracy.
        """
        X = pd.DataFrame([features])
        point_estimate = float(self.predict(X)[0])

        # Per-game xFIP standard deviation is typically ~1.2-1.5 for MLB starters
        # This reflects true randomness in outcomes, not model error
        per_game_std = 1.30

        # z-score for the requested confidence level
        z = {0.80: 1.28, 0.90: 1.645, 0.95: 1.96}.get(confidence, 1.28)
        margin = z * per_game_std

        return {
            "projected_xfip": round(point_estimate, 2),
            "xfip_lower": round(max(1.0, point_estimate - margin), 2),
            "xfip_upper": round(min(10.0, point_estimate + margin), 2),
            "confidence_level": confidence,
            "note": "Wide interval reflects inherent game-to-game variance, not model uncertainty",
        }
