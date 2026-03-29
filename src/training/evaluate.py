"""
Model evaluation utilities.

Provides walk-forward cross-validation, Brier score, calibration curves,
and per-metric evaluation functions for each model type.

Walk-forward CV (time-series safe):
  Train on seasons [n, n+1, ..., n+k-1], validate on season n+k.
  Repeat sliding forward one season at a time.
  This prevents data leakage — the model never sees future game outcomes.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, mean_absolute_error

from src.models.win_probability import compute_win_prob_from_lambdas

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run expectancy model evaluation
# ---------------------------------------------------------------------------

def evaluate_run_model(model, X_val: pd.DataFrame, y_val: pd.DataFrame) -> dict:
    """
    Evaluate the run expectancy model.

    Metrics:
    - MAE (mean absolute error) for home and away run predictions
    - Brier score: computed by converting Poisson lambdas → win prob → Brier vs actual outcomes
    """
    if X_val.empty or y_val.empty:
        return {}

    valid = y_val.notna().all(axis=1)
    if not valid.any():
        return {}

    X = X_val[valid]
    y = y_val[valid]

    preds = model.predict(X)  # shape (N, 2): [lambda_home, lambda_away]
    lambdas_home = preds[:, 0]
    lambdas_away = preds[:, 1]

    mae_home = float(mean_absolute_error(y["runs_scored_home"], lambdas_home))
    mae_away = float(mean_absolute_error(y["runs_scored_away"], lambdas_away))

    # Win probability from Poisson lambdas
    win_probs = np.array([
        compute_win_prob_from_lambdas(lh, la)[0]
        for lh, la in zip(lambdas_home, lambdas_away)
    ])

    # Brier score (requires actual home win outcomes)
    if "home_win" in X_val.columns:
        actuals = X_val["home_win"].loc[valid].astype(float)
        brier = float(brier_score_loss(actuals, win_probs))
    else:
        brier = None

    return {"mae_home": mae_home, "mae_away": mae_away, "brier_score": brier}


# ---------------------------------------------------------------------------
# Pitcher model evaluation
# ---------------------------------------------------------------------------

def evaluate_pitcher_model(model, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """
    Evaluate the pitcher xFIP model.

    Metrics:
    - RMSE (reflects per-game variance)
    - MAE
    - Correlation (how well relative ranking is preserved)
    """
    valid = y_val.notna()
    if not valid.any():
        return {}

    preds = model.predict(X_val[valid])
    actuals = y_val[valid].values

    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mae = float(mean_absolute_error(actuals, preds))
    corr = float(np.corrcoef(actuals, preds)[0, 1]) if len(actuals) > 1 else 0.0

    return {"rmse_xfip": rmse, "mae_xfip": mae, "correlation": corr}


# ---------------------------------------------------------------------------
# Batter matchup model evaluation
# ---------------------------------------------------------------------------

def evaluate_batter_model(model, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """
    Evaluate the batter matchup xwOBA model.

    Metrics:
    - RMSE
    - Correlation (ranking accuracy is more important than absolute accuracy)
    """
    valid = y_val.notna()
    if not valid.any() or X_val.empty:
        return {}

    preds = model.predict(X_val[valid])
    actuals = y_val[valid].values

    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    corr = float(np.corrcoef(actuals, preds)[0, 1]) if len(actuals) > 1 else 0.0

    return {"rmse": rmse, "correlation": corr}


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

def walk_forward_cv(model_factory, feature_df: pd.DataFrame,
                    target_col: str, seasons: list[int],
                    min_train_seasons: int = 2) -> pd.DataFrame:
    """
    Walk-forward cross-validation for time-series-safe model assessment.

    Args:
        model_factory: callable returning a fresh model instance
        feature_df: full feature matrix with 'season' column
        target_col: target column name
        seasons: list of seasons in chronological order
        min_train_seasons: minimum number of training seasons

    Returns:
        DataFrame with columns: val_season, mae, rmse, correlation
    """
    results = []
    for i in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:i]
        val_season = seasons[i]

        train_df = feature_df[feature_df["season"].isin(train_seasons)]
        val_df = feature_df[feature_df["season"] == val_season]

        if train_df.empty or val_df.empty:
            continue

        meta_cols = ["game_pk", "game_date", "season", "home_win",
                     "runs_scored_home", "runs_scored_away"]
        feature_cols = [c for c in train_df.columns if c not in meta_cols]

        X_train = train_df[feature_cols]
        y_train = train_df[target_col] if target_col in train_df.columns else None

        if y_train is None or y_train.isna().all():
            continue

        model = model_factory()
        model.fit(X_train, y_train)

        X_val = val_df[feature_cols]
        y_val = val_df[target_col] if target_col in val_df.columns else pd.Series()

        preds = model.predict(X_val)
        valid = y_val.notna()
        if valid.sum() < 10:
            continue

        rmse = float(np.sqrt(np.mean((preds[valid] - y_val[valid].values) ** 2)))
        mae = float(mean_absolute_error(y_val[valid], preds[valid]))
        corr = float(np.corrcoef(y_val[valid], preds[valid])[0, 1])

        results.append({
            "val_season": val_season,
            "train_seasons": train_seasons,
            "mae": mae,
            "rmse": rmse,
            "correlation": corr,
            "n_val_samples": int(valid.sum()),
        })
        logger.info("Walk-forward CV — val_season=%d, MAE=%.3f, RMSE=%.3f, corr=%.3f",
                    val_season, mae, rmse, corr)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Calibration analysis (for win probability)
# ---------------------------------------------------------------------------

def assess_calibration(win_probs: np.ndarray, actuals: np.ndarray,
                        n_bins: int = 10) -> dict:
    """
    Assess how well win probabilities are calibrated.
    Well-calibrated: when model says 70%, team wins ~70% of the time.

    Returns:
        dict with brier_score, calibration_data (fraction_of_positives vs mean_predicted_value)
    """
    brier = float(brier_score_loss(actuals, win_probs))
    fraction_pos, mean_pred = calibration_curve(actuals, win_probs, n_bins=n_bins)

    # Mean absolute calibration error
    mace = float(np.mean(np.abs(fraction_pos - mean_pred)))

    return {
        "brier_score": brier,
        "mean_abs_calibration_error": mace,
        "calibration_curve": {
            "mean_predicted": mean_pred.tolist(),
            "fraction_positive": fraction_pos.tolist(),
        },
    }


def mean_squared_error(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
