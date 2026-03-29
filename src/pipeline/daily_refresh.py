"""
Daily refresh pipeline.

Runs automatically via APScheduler (configured in config/schedule.yaml).
Also callable directly for manual refreshes or testing.

Morning refresh (9 AM ET):
  1. Fetch today's schedule with probable pitchers
  2. For each game: build projected lineups, run all models, store predictions
  3. Ingest previous day's completed game results + update Elo

Afternoon/evening refresh (3 PM and 5 PM ET):
  1. Check for confirmed lineups
  2. For newly confirmed games: re-run batter matchup + simulator → update prediction

Each refresh logs to the refresh_log table so the API /system/status
endpoint and frontend status bar always have accurate freshness data.
Errors are caught per-game so a single failure doesn't abort the whole refresh.
"""
import importlib
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import yaml
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.connection import get_db
from src.db.queries import get_active_model_version, get_lineup
from src.db.schema import Prediction, RefreshLog
from src.pipeline.features import build_prediction_features
from src.pipeline.ingest import run_daily_ingest
from src.pipeline.lineups import (
    check_and_update_confirmed_lineups,
    fetch_and_store_lineups,
)
from src.pipeline.matchups import build_lineup_matchup_features as _build_lineup_matchup
from src.models.win_probability import compute_win_probability
from src.models.simulator import MonteCarloSimulator, SimConfig
from src.pipeline.adapters.mlb_stats import MLBStatsAdapter

logger = logging.getLogger(__name__)

_ARTIFACTS_DIR = Path(__file__).parents[2] / "data" / "models"
_SCHEDULE_CONFIG = Path(__file__).parents[2] / "config" / "models.yaml"

_mlb = MLBStatsAdapter()
_simulator = MonteCarloSimulator()


# ---------------------------------------------------------------------------
# Morning refresh
# ---------------------------------------------------------------------------

def run_morning_refresh() -> None:
    """
    Morning refresh: fetch schedule, run predictions with projected lineups,
    ingest yesterday's game results.
    """
    today = str(date.today())
    yesterday = str(date.today() - timedelta(days=1))

    log_id = _start_refresh_log("morning")
    games_updated = 0
    errors = []

    try:
        # 1. Ingest yesterday's completed games + update Elo
        logger.info("Ingesting completed games for %s", yesterday)
        try:
            run_daily_ingest(yesterday)
        except Exception as exc:
            logger.error("Daily ingest failed for %s: %s", yesterday, exc)
            errors.append(f"Ingest failed: {exc}")

        # 2. Fetch today's schedule
        games = _mlb.get_schedule(today)
        logger.info("Found %d games scheduled for %s", len(games), today)

        # 3. Load models
        run_model, pitcher_model, batter_model = _load_active_models()

        for game in games:
            try:
                updated = _process_game_prediction(
                    game, today, run_model, pitcher_model, batter_model,
                    use_confirmed_lineup=False,
                )
                if updated:
                    games_updated += 1
            except Exception as exc:
                err_msg = f"Prediction failed for game_pk={game.get('game_pk')}: {exc}"
                logger.error(err_msg)
                errors.append(err_msg)

    except Exception as exc:
        logger.exception("Morning refresh failed: %s", exc)
        errors.append(str(exc))
        _finish_refresh_log(log_id, "failed", games_updated=games_updated,
                            error_message="; ".join(errors[:3]))
        return

    # Daily QC — runs after predictions are generated so all checks have data
    qc_warnings = _run_daily_qc(today, yesterday, games)
    if qc_warnings:
        errors.extend(qc_warnings)

    status = "completed" if not errors else "partial"
    _finish_refresh_log(log_id, status, games_updated=games_updated,
                        error_message="; ".join(errors[:3]) if errors else None)
    logger.info("Morning refresh complete: %d games updated, %d errors", games_updated, len(errors))


# ---------------------------------------------------------------------------
# Afternoon/evening refresh
# ---------------------------------------------------------------------------

def run_afternoon_refresh() -> None:
    """
    Afternoon refresh: check for confirmed lineups, re-run affected predictions.
    """
    today = str(date.today())
    log_id = _start_refresh_log("afternoon")
    lineups_confirmed = 0
    errors = []

    try:
        games = _mlb.get_schedule(today)
        run_model, pitcher_model, batter_model = _load_active_models()

        for game in games:
            try:
                result = check_and_update_confirmed_lineups(
                    game["game_pk"], game["home_team_id"],
                    game["away_team_id"], today,
                )
                if result["home_updated"] or result["away_updated"]:
                    lineups_confirmed += 1
                    # Re-run prediction with confirmed lineup
                    _process_game_prediction(
                        game, today, run_model, pitcher_model, batter_model,
                        use_confirmed_lineup=True,
                    )
                    logger.info("Refreshed prediction for game_pk=%s with confirmed lineup",
                                game["game_pk"])
            except Exception as exc:
                err_msg = f"Lineup refresh failed for game_pk={game.get('game_pk')}: {exc}"
                logger.error(err_msg)
                errors.append(err_msg)

    except Exception as exc:
        logger.exception("Afternoon refresh failed: %s", exc)
        _finish_refresh_log(log_id, "failed", lineups_confirmed=lineups_confirmed,
                            error_message=str(exc))
        return

    status = "completed" if not errors else "partial"
    _finish_refresh_log(log_id, status, lineups_confirmed=lineups_confirmed,
                        error_message="; ".join(errors[:3]) if errors else None)
    logger.info("Afternoon refresh: %d lineup confirmations, %d errors",
                lineups_confirmed, len(errors))


# ---------------------------------------------------------------------------
# Core prediction logic (shared by morning + afternoon)
# ---------------------------------------------------------------------------

def _process_game_prediction(game: dict, game_date: str,
                               run_model, pitcher_model, batter_model,
                               use_confirmed_lineup: bool = False) -> bool:
    """
    Run the full prediction pipeline for a single game.
    Returns True if prediction was stored successfully.
    """
    game_pk = game["game_pk"]
    home_team_id = game.get("home_team_id")
    away_team_id = game.get("away_team_id")
    venue_id = game.get("venue_id")
    season = int(game_date[:4])

    if not home_team_id or not away_team_id:
        return False

    # 1. Get or build lineups.
    # fetch_and_store_lineups returns the best available SP IDs — it prefers
    # the confirmed starter from the live boxscore (afternoon/evening) over
    # the schedule's probable pitcher (morning), which in turn was resolved
    # from a string name to an ID if needed.
    lineup_result = fetch_and_store_lineups(
        game_pk, home_team_id, away_team_id, game_date,
        game.get("home_sp_id"), game.get("away_sp_id"),
    )
    home_lineup = lineup_result["home_lineup"]
    away_lineup = lineup_result["away_lineup"]
    home_confirmed = lineup_result["home_confirmed"]
    away_confirmed = lineup_result["away_confirmed"]
    # Use the best SP ID available: boxscore confirmed > schedule probable > None
    home_sp_id = lineup_result.get("home_sp_id") or game.get("home_sp_id")
    away_sp_id = lineup_result.get("away_sp_id") or game.get("away_sp_id")

    # 2. Build game-level features
    features = build_prediction_features(
        game_pk, home_team_id, away_team_id,
        home_sp_id, away_sp_id, game_date, venue_id,
    )
    if features is None:
        logger.warning("Could not build features for game_pk=%d", game_pk)
        return False

    # 3. Run expectancy model → λ values
    lambda_result = {"lambda_home": 4.35, "lambda_away": 4.35}  # default
    if run_model is not None:
        try:
            import pandas as pd
            lambda_result = run_model.predict_single(features)
        except Exception as exc:
            logger.warning("Run model failed for game_pk=%d: %s", game_pk, exc)

    lambda_home = lambda_result.get("lambda_home", 4.35)
    lambda_away = lambda_result.get("lambda_away", 4.35)

    # 4. Pitcher projection
    pitcher_advantage_home = None
    pitcher_advantage_away = None
    home_sp_xfip = None
    away_sp_xfip = None
    if pitcher_model is not None and home_sp_id and away_sp_id:
        try:
            import pandas as pd
            import numpy as np
            home_sp_xfip = float(pitcher_model.predict(pd.DataFrame([features]))[0])
            away_sp_xfip = float(pitcher_model.predict(pd.DataFrame([features]))[0])
            pitcher_advantage_home = round(4.0 - home_sp_xfip, 2)
            pitcher_advantage_away = round(4.0 - away_sp_xfip, 2)
        except Exception as exc:
            logger.warning("Pitcher model failed for game_pk=%d: %s", game_pk, exc)

    # 5. Batter matchup scores
    home_lineup_xwoba = None
    away_lineup_xwoba = None
    home_batter_xwobas = []
    away_batter_xwobas = []
    if batter_model is not None:
        try:
            home_matchup = _build_lineup_matchup(
                home_lineup, away_sp_id, season,
                before_date=game_date, opposing_team_id=away_team_id,
            )
            away_matchup = _build_lineup_matchup(
                away_lineup, home_sp_id, season,
                before_date=game_date, opposing_team_id=home_team_id,
            )
            home_lineup_xwoba = home_matchup.get("lineup_xwoba_score")
            away_lineup_xwoba = away_matchup.get("lineup_xwoba_score")
            home_batter_xwobas = [s["xwoba"] for s in home_matchup.get("batter_matchup_scores", [])]
            away_batter_xwobas = [s["xwoba"] for s in away_matchup.get("batter_matchup_scores", [])]
        except Exception as exc:
            logger.warning("Batter matchup failed for game_pk=%d: %s", game_pk, exc)

    # 6. Monte Carlo simulation (if we have lineup xwOBA scores)
    sim_result = None
    if home_batter_xwobas and away_batter_xwobas:
        try:
            sim_result = _simulator.simulate_game(
                home_batter_xwobas, away_batter_xwobas,
                home_sp_xfip, away_sp_xfip,
            )
        except Exception as exc:
            logger.warning("Simulator failed for game_pk=%d: %s", game_pk, exc)

    # 7. Fallback to analytical Poisson win prob if simulation unavailable
    if sim_result is None:
        poisson_result = compute_win_probability(lambda_home, lambda_away)
        sim_result = poisson_result

    # 8. Get active model version ID
    with get_db() as session:
        mv = get_active_model_version(session, "run_expectancy")
        model_version_id = mv.id if mv else None

    # 9. Store prediction
    _upsert_prediction(
        game_pk=game_pk,
        game_date=game_date,
        model_version_id=model_version_id,
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        sim_result=sim_result,
        pitcher_advantage_home=pitcher_advantage_home,
        pitcher_advantage_away=pitcher_advantage_away,
        lineup_xwoba_home=home_lineup_xwoba,
        lineup_xwoba_away=away_lineup_xwoba,
        home_lineup_confirmed=home_confirmed,
        away_lineup_confirmed=away_confirmed,
    )
    return True


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _upsert_prediction(game_pk: int, game_date: str, model_version_id: Optional[int],
                        lambda_home: float, lambda_away: float, sim_result: dict,
                        **kwargs) -> None:
    from datetime import datetime as dt
    values = {
        "game_pk": game_pk,
        "game_date": game_date,
        "model_version_id": model_version_id,
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "win_prob_home": sim_result.get("win_prob_home"),
        "win_prob_away": sim_result.get("win_prob_away"),
        "median_runs_home": sim_result.get("median_runs_home"),
        "median_runs_away": sim_result.get("median_runs_away"),
        "p10_runs_home": sim_result.get("p10_runs_home"),
        "p90_runs_home": sim_result.get("p90_runs_home"),
        "p10_runs_away": sim_result.get("p10_runs_away"),
        "p90_runs_away": sim_result.get("p90_runs_away"),
        "generated_at": dt.utcnow(),
        **kwargs,
    }
    with get_db() as session:
        stmt = (
            pg_insert(Prediction.__table__)
            .values(**{k: v for k, v in values.items() if v is not None})
            .on_conflict_do_update(
                index_elements=["game_pk", "model_version_id"],
                set_={k: v for k, v in values.items() if k not in ("game_pk", "model_version_id")},
            )
        )
        session.execute(stmt)


def _start_refresh_log(refresh_type: str) -> int:
    from datetime import datetime as dt
    with get_db() as session:
        from sqlalchemy import text
        result = session.execute(
            text("""
                INSERT INTO refresh_log (refresh_type, started_at, status)
                VALUES (:rt, :sa, 'running')
                RETURNING id
            """),
            {"rt": refresh_type, "sa": dt.utcnow()},
        )
        return result.scalar()


def _finish_refresh_log(log_id: int, status: str, games_updated: int = 0,
                         lineups_confirmed: int = 0,
                         error_message: Optional[str] = None) -> None:
    from datetime import datetime as dt
    with get_db() as session:
        from sqlalchemy import text
        session.execute(
            text("""
                UPDATE refresh_log
                SET status = :status, completed_at = :ca,
                    games_updated = :gu, lineups_confirmed = :lc,
                    error_message = :em
                WHERE id = :id
            """),
            {"status": status, "ca": dt.utcnow(), "gu": games_updated,
             "lc": lineups_confirmed, "em": error_message, "id": log_id},
        )


def _run_daily_qc(today: str, yesterday: str, scheduled_games: list) -> list[str]:
    """
    Lightweight QC checks that run automatically after every morning refresh.

    Checks
    ------
    1. Yesterday's ingest  — all completed games have scores stored
    2. SP resolution       — fraction of today's games with a known starter ID
    3. Prediction coverage — a prediction exists for every scheduled game
    4. Prediction sanity   — no win probability is outside the 15–85% range
                            (extreme values suggest a feature pipeline failure)
    5. Lineup status       — log confirmed vs projected lineup counts

    Returns a list of warning strings (empty = all clear).
    Warnings are appended to the refresh_log error_message field so they
    surface in the /system/status API endpoint and frontend status bar.
    """
    warnings: list[str] = []

    # 1. Yesterday's ingest completeness
    with get_db() as session:
        from sqlalchemy import text
        missing_scores = session.execute(
            text("""
                SELECT COUNT(*) FROM games
                WHERE game_date = :d AND status = 'Final'
                  AND (home_score IS NULL OR away_score IS NULL)
            """),
            {"d": yesterday},
        ).scalar()
    if missing_scores:
        warnings.append(
            f"QC: {missing_scores} Final game(s) from {yesterday} are missing scores"
        )

    # 2. SP resolution rate for today
    if scheduled_games:
        no_home_sp = sum(1 for g in scheduled_games if not g.get("home_sp_id"))
        no_away_sp = sum(1 for g in scheduled_games if not g.get("away_sp_id"))
        total = len(scheduled_games)
        if no_home_sp > total // 2:
            warnings.append(
                f"QC: {no_home_sp}/{total} home SP IDs unresolved for {today}"
            )
        if no_away_sp > total // 2:
            warnings.append(
                f"QC: {no_away_sp}/{total} away SP IDs unresolved for {today}"
            )

    # 3. Prediction coverage
    if scheduled_games:
        scheduled_pks = {g["game_pk"] for g in scheduled_games}
        with get_db() as session:
            from sqlalchemy import text as _t
            rows = session.execute(
                _t("SELECT DISTINCT game_pk FROM predictions WHERE game_date = :d"),
                {"d": today},
            ).fetchall()
        predicted_pks = {r[0] for r in rows}
        missing_preds = scheduled_pks - predicted_pks
        if missing_preds:
            warnings.append(
                f"QC: {len(missing_preds)} game(s) scheduled for {today} "
                f"have no prediction: {sorted(missing_preds)[:3]}"
            )

    # 4. Prediction sanity — win probabilities outside 15–85% are suspicious
    with get_db() as session:
        from sqlalchemy import text as _t
        extreme = session.execute(
            _t("""
                SELECT COUNT(*) FROM predictions
                WHERE game_date = :d
                  AND (win_prob_home < 0.15 OR win_prob_home > 0.85)
            """),
            {"d": today},
        ).scalar()
    if extreme:
        warnings.append(
            f"QC: {extreme} prediction(s) for {today} have extreme win probability "
            f"(<15% or >85%) — possible feature pipeline issue"
        )

    # 5. Lineup status (informational — always log, never a warning)
    with get_db() as session:
        from sqlalchemy import text as _t
        confirmed = session.execute(
            _t("SELECT COUNT(*) FROM lineups WHERE game_date = :d AND is_confirmed = TRUE"),
            {"d": today},
        ).scalar()
        projected = session.execute(
            _t("SELECT COUNT(*) FROM lineups WHERE game_date = :d AND is_confirmed = FALSE"),
            {"d": today},
        ).scalar()
    logger.info(
        "Daily QC lineup status for %s: %d confirmed, %d projected",
        today, confirmed or 0, projected or 0,
    )

    if warnings:
        for w in warnings:
            logger.warning(w)
    else:
        logger.info("Daily QC passed for %s — all checks clean", today)

    return warnings


def _load_active_models():
    """Load the currently active model artifacts from DB + disk."""
    run_model = None
    pitcher_model = None
    batter_model = None

    with get_db() as session:
        run_mv = get_active_model_version(session, "run_expectancy")
        pitcher_mv = get_active_model_version(session, "pitcher_model")
        batter_mv = get_active_model_version(session, "batter_matchup")

    def _load(mv, class_path: str):
        if mv is None:
            return None
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            import importlib
            cls = getattr(importlib.import_module(module_path), class_name)
            return cls().load(Path(mv.artifact_path))
        except Exception as exc:
            logger.error("Failed to load model %s: %s", class_path, exc)
            return None

    run_model = _load(run_mv, "src.models.run_expectancy.XGBoostRunModel")
    pitcher_model = _load(pitcher_mv, "src.models.pitcher_model.XGBoostPitcherModel")
    batter_model = _load(batter_mv, "src.models.batter_matchup.LGBMBatterMatchup")

    return run_model, pitcher_model, batter_model
