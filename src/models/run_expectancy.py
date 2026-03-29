"""
Run Expectancy Model: predicts the Poisson rate λ for runs scored by a team.

Model: XGBoost with count:poisson objective.
The Poisson objective trains the model to output log(λ) internally and
return λ directly, making it the natural loss function for count data.

λ represents the expected number of runs — e.g., λ=4.3 means we expect
roughly a Poisson(4.3) distribution over possible run totals.

This λ feeds directly into the win probability model (Skellam distribution)
without requiring a separate binary classifier.
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

# Columns that are metadata/targets, not features
_NON_FEATURE_COLS = [
    "game_pk", "game_date", "season", "home_win",
    "runs_scored_home", "runs_scored_away",
]


def load_model_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f).get("run_expectancy", {})


class XGBoostRunModel(BaseMLBModel):
    """
    XGBoost Poisson regression for run expectancy.

    Trains two instances — one for home runs, one for away runs — because
    home/away is a strong signal and training separately keeps the feature
    space cleaner (no is_home column needed).
    """

    model_type = "run_expectancy"

    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        self._config = config or load_model_config()
        self._home_model: Optional[xgb.XGBRegressor] = None
        self._away_model: Optional[xgb.XGBRegressor] = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "XGBoostRunModel":
        """
        Train home and away run models.

        Args:
            X: feature matrix (one row per game)
            y: DataFrame with columns 'runs_scored_home' and 'runs_scored_away'
        """
        params = self._config.get("params", {})
        feature_cols = [c for c in X.columns if c not in _NON_FEATURE_COLS]
        self._feature_names = feature_cols

        X_feat = X[feature_cols]
        y_home = y["runs_scored_home"].fillna(y["runs_scored_home"].median())
        y_away = y["runs_scored_away"].fillna(y["runs_scored_away"].median())

        # XGBoost count:poisson requires non-negative targets
        y_home = y_home.clip(lower=0)
        y_away = y_away.clip(lower=0)

        logger.info("Training home run model on %d samples", len(X_feat))
        self._home_model = self._make_xgb(params)
        self._home_model.fit(X_feat, y_home,
                             eval_set=[(X_feat, y_home)],
                             verbose=False)

        logger.info("Training away run model on %d samples", len(X_feat))
        self._away_model = self._make_xgb(params)
        self._away_model.fit(X_feat, y_away,
                             eval_set=[(X_feat, y_away)],
                             verbose=False)

        self._model = (self._home_model, self._away_model)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns array of shape (N, 2) with columns [lambda_home, lambda_away].
        """
        X_feat = self._align_features(X)
        lambda_home = self._home_model.predict(X_feat)
        lambda_away = self._away_model.predict(X_feat)
        # Clip to physically meaningful range (no negative run rates)
        lambda_home = np.clip(lambda_home, 0.5, 20.0)
        lambda_away = np.clip(lambda_away, 0.5, 20.0)
        return np.column_stack([lambda_home, lambda_away])

    def predict_single(self, features: dict) -> dict:
        """
        Predict for a single game given a feature dict.
        Returns: {lambda_home, lambda_away}
        """
        X = pd.DataFrame([features])
        result = self.predict(X)[0]
        return {"lambda_home": float(result[0]), "lambda_away": float(result[1])}

    def get_feature_importance(self) -> pd.DataFrame:
        """Return SHAP-ready feature importance from the home model."""
        if self._home_model is None:
            return pd.DataFrame()
        scores = self._home_model.get_booster().get_fscore()
        return pd.DataFrame(
            {"feature": list(scores.keys()), "importance": list(scores.values())}
        ).sort_values("importance", ascending=False)

    def save(self, path: Path) -> None:
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "home_model": self._home_model,
            "away_model": self._away_model,
            "feature_names": self._feature_names,
        }, path)
        self._artifact_path = path

    def load(self, path: Path) -> "XGBoostRunModel":
        import joblib
        artifact = joblib.load(path)
        self._home_model = artifact["home_model"]
        self._away_model = artifact["away_model"]
        self._feature_names = artifact["feature_names"]
        self._model = (self._home_model, self._away_model)
        self._artifact_path = path
        return self

    @staticmethod
    def _make_xgb(params: dict) -> xgb.XGBRegressor:
        # XGBRegressor accepts count:poisson objective
        return xgb.XGBRegressor(
            objective=params.get("objective", "count:poisson"),
            n_estimators=params.get("n_estimators", 500),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 6),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            min_child_weight=params.get("min_child_weight", 5),
            reg_alpha=params.get("reg_alpha", 0.1),
            reg_lambda=params.get("reg_lambda", 1.0),
            early_stopping_rounds=params.get("early_stopping_rounds", 50),
            eval_metric=params.get("eval_metric", "poisson-nloglik"),
            n_jobs=-1,
            random_state=42,
        )
