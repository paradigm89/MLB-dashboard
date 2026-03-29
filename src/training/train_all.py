"""
Training orchestrator: trains all ML models using historical data from the DB.

Reads config/models.yaml to determine which model classes to instantiate.
Saves model artifacts to data/models/ with date-stamped filenames.
Writes model version records to the `model_versions` DB table.

Usage:
    python -m src.training.train_all
    python -m src.training.train_all --train-seasons 2019 2024 --val-season 2025
"""
import argparse
import importlib
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.connection import get_db
from src.db.schema import ModelVersion
from src.pipeline.features import build_game_feature_matrix
from src.training.evaluate import evaluate_run_model, evaluate_batter_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "models.yaml"
_ARTIFACTS_DIR = Path(__file__).parents[2] / "data" / "models"

# Metadata columns that should not be used as features or targets
_META_COLS = ["game_pk", "game_date", "season", "home_win"]


def load_model_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def train_all(train_seasons: list[int], val_season: int) -> None:
    """
    Train all models registered in config/models.yaml.

    1. Build game-level feature matrix from DB
    2. Split into train/validation by season (temporal split — no leakage)
    3. Train run expectancy model
    4. Train pitcher model
    5. Train batter matchup model (uses Statcast PA-level data)
    6. Save all artifacts and record in model_versions table
    """
    cfg = load_model_config()
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")

    logger.info("Building game feature matrix for seasons %s (train) + %d (val)",
                train_seasons, val_season)
    all_seasons = train_seasons + [val_season]
    feature_df = build_game_feature_matrix(all_seasons)

    if feature_df.empty:
        logger.error("No feature data available. Run ingest.py first.")
        return

    train_df = feature_df[feature_df["season"].isin(train_seasons)].copy()
    val_df = feature_df[feature_df["season"] == val_season].copy()
    logger.info("Train: %d games, Val: %d games", len(train_df), len(val_df))

    # Cast all feature columns to float so XGBoost doesn't choke on object dtype.
    # None / non-numeric values become np.nan, which XGBoost handles natively.
    _feat_cols = [c for c in feature_df.columns if c not in _META_COLS + ["runs_scored_home", "runs_scored_away"]]
    for _df in (train_df, val_df):
        for _col in _feat_cols:
            if _col in _df.columns and _df[_col].dtype == object:
                _df[_col] = pd.to_numeric(_df[_col], errors="coerce")

    # --- 1. Run Expectancy Model -------------------------------------------
    _train_run_model(cfg, train_df, val_df, date_str)

    # --- 2. Pitcher Model --------------------------------------------------
    _train_pitcher_model(cfg, train_seasons, val_season, date_str)

    # --- 3. Batter Matchup Model ------------------------------------------
    _train_batter_model(cfg, train_seasons, val_season, date_str)

    logger.info("Training complete. All models saved to %s", _ARTIFACTS_DIR)


def _train_run_model(cfg: dict, train_df: pd.DataFrame,
                      val_df: pd.DataFrame, date_str: str) -> None:
    model_cfg = cfg.get("run_expectancy", {})
    model_class = _load_class(model_cfg["class"])
    model = model_class(config=model_cfg)

    target_cols = ["runs_scored_home", "runs_scored_away"]
    X_train = train_df.drop(columns=_META_COLS + target_cols, errors="ignore")
    y_train = train_df[target_cols]
    X_val = val_df.drop(columns=_META_COLS + target_cols, errors="ignore")
    y_val = val_df[target_cols]

    # Drop rows where either target is null
    valid_train = y_train.notna().all(axis=1)
    model.fit(X_train[valid_train], y_train[valid_train])

    # Evaluate
    metrics = evaluate_run_model(model, X_val, y_val)
    logger.info("Run model validation — MAE home: %.3f, MAE away: %.3f, Brier: %.4f",
                metrics.get("mae_home", 0), metrics.get("mae_away", 0),
                metrics.get("brier_score", 0))

    artifact_path = _ARTIFACTS_DIR / f"run_exp_{date_str}.pkl"
    model.save(artifact_path)
    _record_model_version(
        model_type="run_expectancy",
        artifact_path=str(artifact_path),
        train_seasons=train_df["season"].unique().tolist(),
        val_season=int(val_df["season"].iloc[0]) if not val_df.empty else None,
        brier_score=metrics.get("brier_score"),
        mae_runs=metrics.get("mae_home"),
    )
    logger.info("Run expectancy model saved to %s", artifact_path)


def _train_pitcher_model(cfg: dict, train_seasons: list[int],
                          val_season: int, date_str: str) -> None:
    """
    Pitcher model trains on start-by-start game appearances.
    Features are extracted per start from the game feature matrix.
    """
    model_cfg = cfg.get("pitcher_model", {})
    model_class = _load_class(model_cfg["class"])
    model = model_class(config=model_cfg)

    # Build pitcher-specific feature matrix (one row per starting pitcher appearance)
    pitcher_df = _build_pitcher_training_data(train_seasons + [val_season])
    if pitcher_df.empty:
        logger.warning("No pitcher training data found; skipping pitcher model")
        return

    train_p = pitcher_df[pitcher_df["season"].isin(train_seasons)]
    val_p = pitcher_df[pitcher_df["season"] == val_season]

    target = "xfip_actual"
    X_train = train_p.drop(columns=["game_pk", "game_date", "season", "pitcher_id", target], errors="ignore")
    y_train = train_p.get(target, pd.Series(dtype=float))
    X_val = val_p.drop(columns=["game_pk", "game_date", "season", "pitcher_id", target], errors="ignore")
    y_val = val_p.get(target, pd.Series(dtype=float))

    valid = y_train.notna()
    model.fit(X_train[valid], y_train[valid])

    if not val_p.empty and y_val.notna().any():
        val_preds = model.predict(X_val)
        rmse = float(np.sqrt(np.mean((val_preds - y_val.dropna().values) ** 2)))
        logger.info("Pitcher model validation RMSE (xFIP): %.3f", rmse)
    else:
        rmse = None

    artifact_path = _ARTIFACTS_DIR / f"pitcher_{date_str}.pkl"
    model.save(artifact_path)
    _record_model_version(
        model_type="pitcher_model",
        artifact_path=str(artifact_path),
        train_seasons=train_seasons,
        val_season=val_season,
        rmse_xfip=rmse,
    )
    logger.info("Pitcher model saved to %s", artifact_path)


def _train_batter_model(cfg: dict, train_seasons: list[int],
                         val_season: int, date_str: str) -> None:
    """
    Batter matchup model trains on PA-level Statcast data.
    Features from matchups.py; target is estimated_woba_using_speedangle.
    """
    model_cfg = cfg.get("batter_matchup", {})
    model_class = _load_class(model_cfg["class"])
    model = model_class(config=model_cfg)

    pa_df = _build_pa_training_data(train_seasons + [val_season])
    if pa_df.empty:
        logger.warning("No PA training data; skipping batter matchup model")
        return

    train_pa = pa_df[pa_df["season"].isin(train_seasons)]
    val_pa = pa_df[pa_df["season"] == val_season]

    target = "xwoba_actual"
    X_train = train_pa.drop(columns=["batter_id", "pitcher_id", "season", target], errors="ignore")
    y_train = train_pa.get(target, pd.Series(dtype=float))
    X_val = val_pa.drop(columns=["batter_id", "pitcher_id", "season", target], errors="ignore")
    y_val = val_pa.get(target, pd.Series(dtype=float))

    valid = y_train.notna()
    model.fit(X_train[valid], y_train[valid])

    metrics = evaluate_batter_model(model, X_val, y_val)
    logger.info("Batter matchup model — correlation: %.3f, RMSE: %.4f",
                metrics.get("correlation", 0), metrics.get("rmse", 0))

    artifact_path = _ARTIFACTS_DIR / f"batter_matchup_{date_str}.pkl"
    model.save(artifact_path)
    _record_model_version(
        model_type="batter_matchup",
        artifact_path=str(artifact_path),
        train_seasons=train_seasons,
        val_season=val_season,
        correlation_xwoba=metrics.get("correlation"),
    )
    logger.info("Batter matchup model saved to %s", artifact_path)


# ---------------------------------------------------------------------------
# Training data builders
# ---------------------------------------------------------------------------

def _build_pitcher_training_data(seasons: list[int]) -> pd.DataFrame:
    """
    Build a per-start DataFrame for pitcher model training.
    Joins game appearance data with season pitcher stats and Statcast features.
    """
    rows = []
    with get_db() as session:
        from sqlalchemy import text
        for season in seasons:
            result = session.execute(
                text("""
                    SELECT g.game_pk, g.game_date, g.season,
                           g.home_sp_id as pitcher_id, 'home' as side,
                           g.away_team_id as opponent_team_id,
                           ps.xfip as xfip_season, ps.k_pct, ps.bb_pct,
                           ps.avg_fastball_velo, ps.swstr_pct, ps.pct_ff,
                           ps.pct_sl, ps.pct_ch, ps.pct_cu, ps.throws,
                           pf.pf_runs as park_factor_runs,
                           -- xFIP actual computed as approximation from Statcast
                           ps.xfip as xfip_actual
                    FROM games g
                    LEFT JOIN pitching_stats ps ON ps.player_id = g.home_sp_id
                        AND ps.season = g.season
                    LEFT JOIN park_factors pf ON pf.team_id = g.home_team_id
                        AND pf.season = g.season
                    WHERE g.season = :season AND g.status = 'Final'
                      AND g.home_sp_id IS NOT NULL
                    UNION ALL
                    SELECT g.game_pk, g.game_date, g.season,
                           g.away_sp_id as pitcher_id, 'away' as side,
                           g.home_team_id as opponent_team_id,
                           ps.xfip as xfip_season, ps.k_pct, ps.bb_pct,
                           ps.avg_fastball_velo, ps.swstr_pct, ps.pct_ff,
                           ps.pct_sl, ps.pct_ch, ps.pct_cu, ps.throws,
                           pf.pf_runs as park_factor_runs,
                           ps.xfip as xfip_actual
                    FROM games g
                    LEFT JOIN pitching_stats ps ON ps.player_id = g.away_sp_id
                        AND ps.season = g.season
                    LEFT JOIN park_factors pf ON pf.team_id = g.away_team_id
                        AND pf.season = g.season
                    WHERE g.season = :season AND g.status = 'Final'
                      AND g.away_sp_id IS NOT NULL
                """),
                {"season": season},
            ).fetchall()
            rows.extend(result)

    if not rows:
        return pd.DataFrame()
    columns = ["game_pk", "game_date", "season", "pitcher_id", "side",
               "opponent_team_id", "xfip_season", "k_pct", "bb_pct",
               "avg_fastball_velo", "swstr_pct", "pct_ff", "pct_sl", "pct_ch",
               "pct_cu", "throws", "park_factor_runs", "xfip_actual"]
    df = pd.DataFrame(rows, columns=columns)
    # Encode categoricals so XGBoost gets all-numeric input
    df["is_home"] = (df["side"] == "home").astype(int)
    df["throws_l"] = (df["throws"] == "L").astype(int)
    df = df.drop(columns=["side", "throws"], errors="ignore")
    # Coerce remaining object columns to float
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _build_pa_training_data(seasons: list[int]) -> pd.DataFrame:
    """
    Build PA-level training data for the batter matchup model.
    Uses Statcast data where estimated_woba_using_speedangle is not null.
    Samples up to 500K rows per season for training efficiency.
    """
    dfs = []
    with get_db() as session:
        from sqlalchemy import text
        for season in seasons:
            result = session.execute(
                text("""
                    SELECT pitcher_id, batter_id, season,
                           stand as batter_hand, p_throws as pitcher_hand,
                           pitch_type, release_speed, release_spin_rate,
                           estimated_woba_using_speedangle as xwoba_actual
                    FROM statcast_pitches
                    WHERE season = :season
                      AND estimated_woba_using_speedangle IS NOT NULL
                      AND events IS NOT NULL
                    ORDER BY RANDOM()
                    LIMIT 500000
                """),
                {"season": season},
            ).fetchall()
            if result:
                cols = ["pitcher_id", "batter_id", "season",
                        "batter_hand", "pitcher_hand", "pitch_type",
                        "release_speed", "release_spin_rate", "xwoba_actual"]
                dfs.append(pd.DataFrame(result, columns=cols))

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    # Encode categorical features
    df["batter_hand_l"] = (df["batter_hand"] == "L").astype(int)
    df["pitcher_hand_l"] = (df["pitcher_hand"] == "L").astype(int)
    df = df.drop(columns=["batter_hand", "pitcher_hand", "pitch_type"], errors="ignore")
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_class(class_path: str):
    """Dynamically load a class from a dot-path string like 'src.models.run_expectancy.XGBoostRunModel'."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _record_model_version(model_type: str, artifact_path: str, train_seasons: list[int],
                           val_season: int | None = None, **metrics) -> None:
    """Write a model version record to the DB and deactivate older versions."""
    with get_db() as session:
        # Deactivate all previous active versions for this model type
        from sqlalchemy import text
        session.execute(
            text("UPDATE model_versions SET is_active = false WHERE model_type = :mt"),
            {"mt": model_type},
        )

        stmt = pg_insert(ModelVersion.__table__).values(
            model_type=model_type,
            artifact_path=artifact_path,
            trained_at=datetime.utcnow(),
            training_seasons="-".join(str(s) for s in sorted(train_seasons)),
            validation_season=val_season,
            is_active=True,
            brier_score=metrics.get("brier_score"),
            mae_runs=metrics.get("mae_runs"),
            rmse_xfip=metrics.get("rmse_xfip"),
            correlation_xwoba=metrics.get("correlation_xwoba"),
        )
        session.execute(stmt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all MLB prediction models")
    parser.add_argument("--train-seasons", nargs="+", type=int,
                        default=list(range(2019, 2025)),
                        help="Seasons to use for training (default: 2019-2024)")
    parser.add_argument("--val-season", type=int, default=2025,
                        help="Season to use for validation (default: 2025)")
    args = parser.parse_args()
    train_all(args.train_seasons, args.val_season)
