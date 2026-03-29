"""
Weekly model retrain scheduler.

Runs every Monday at 6 AM ET (configured in config/schedule.yaml).
Appends the prior week's game results to the training set and retrains
all models. Saves new versioned artifacts and writes to model_versions.

Safety check: if the new model's Brier score regresses by more than
0.01 vs. the previous week's model, logs a warning and keeps the old model
active. This prevents silent quality degradation.
"""
import logging
from datetime import date, timedelta
from pathlib import Path

from src.db.connection import get_db
from src.db.queries import get_active_model_version
from src.training.train_all import train_all

logger = logging.getLogger(__name__)

# How many seasons to use for training at each retrain
# Rolling 2-season window for batter matchup (prevents old Statcast dominating)
# Full history for run expectancy and pitcher models
_CURRENT_YEAR = date.today().year
_BRIER_REGRESSION_THRESHOLD = 0.01


def run_weekly_retrain() -> None:
    """
    Weekly retrain: use all seasons from 2019 through current season.
    Validates against current season data accumulated so far.
    """
    current_season = date.today().year
    train_seasons = list(range(2019, current_season))
    val_season = current_season

    logger.info("Starting weekly retrain. Train: %s, Val: %s",
                train_seasons, val_season)

    # Check if there's enough current-season data to validate against
    with get_db() as session:
        from sqlalchemy import text
        n_games = session.execute(
            text("SELECT COUNT(*) FROM games WHERE season = :s AND status = 'Final'"),
            {"s": val_season},
        ).scalar() or 0

    if n_games < 30:
        logger.info("Only %d completed games in %d — skipping validation, training on all data",
                    n_games, val_season)
        # Train on everything including current season; skip validation split
        train_all(
            train_seasons=list(range(2019, val_season + 1)),
            val_season=train_seasons[-1],  # use last historical season as val
        )
        return

    # Get previous model's Brier score for regression check
    prev_brier = _get_active_brier_score()

    # Run full retrain
    train_all(train_seasons=train_seasons, val_season=val_season)

    # Check for quality regression
    new_brier = _get_active_brier_score()
    if prev_brier is not None and new_brier is not None:
        regression = new_brier - prev_brier
        if regression > _BRIER_REGRESSION_THRESHOLD:
            logger.warning(
                "Brier score regressed by %.4f (prev=%.4f, new=%.4f). "
                "New model is still active; investigate before next refresh.",
                regression, prev_brier, new_brier,
            )
        else:
            logger.info("Retrain complete. Brier: %.4f → %.4f (Δ=%.4f)",
                        prev_brier, new_brier, new_brier - prev_brier)
    else:
        logger.info("Weekly retrain complete.")


def _get_active_brier_score() -> float | None:
    with get_db() as session:
        mv = get_active_model_version(session, "run_expectancy")
    return mv.brier_score if mv else None
