"""
System status endpoint.
GET /api/system/status — returns refresh log, model versions, active errors.
Used by the frontend RefreshStatusBar component.
"""
import logging
from datetime import datetime

from fastapi import APIRouter

from src.api.schemas import (
    ActiveError,
    ModelVersionInfo,
    RefreshStatus,
    SystemStatus,
)
from src.db.connection import get_db
from src.db.queries import get_active_model_version, get_last_refresh

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/system")


@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Return the current system health: last refresh times, model versions, errors."""
    with get_db() as session:
        morning = get_last_refresh(session, "morning")
        afternoon = get_last_refresh(session, "afternoon")
        retrain = get_last_refresh(session, "retrain")

        run_mv = get_active_model_version(session, "run_expectancy")
        pitcher_mv = get_active_model_version(session, "pitcher_model")
        batter_mv = get_active_model_version(session, "batter_matchup")

        # DB last updated = most recent completed refresh
        from sqlalchemy import text
        db_updated = session.execute(
            text("SELECT MAX(completed_at) FROM refresh_log WHERE status IN ('completed','partial')")
        ).scalar()

    def _refresh_status(row) -> RefreshStatus:
        if row is None:
            return RefreshStatus()
        return RefreshStatus(
            ran_at=row.completed_at,
            status=row.status,
            games_updated=row.games_updated,
            lineups_confirmed=row.lineups_confirmed,
            error=row.error_message,
        )

    def _model_version(mv) -> ModelVersionInfo | None:
        if mv is None:
            return None
        return ModelVersionInfo(
            model_type=mv.model_type,
            trained_at=mv.trained_at,
            training_seasons=mv.training_seasons,
            brier_score=mv.brier_score,
            mae_runs=mv.mae_runs,
            rmse_xfip=mv.rmse_xfip,
            correlation_xwoba=mv.correlation_xwoba,
        )

    # Active errors: any refresh that failed in the last 24h
    active_errors = []
    with get_db() as session:
        from sqlalchemy import text
        failed = session.execute(
            text("""
                SELECT refresh_type, started_at, error_message
                FROM refresh_log
                WHERE status = 'failed'
                  AND started_at > NOW() - INTERVAL '24 hours'
                ORDER BY started_at DESC
                LIMIT 5
            """)
        ).fetchall()
    for row in failed:
        active_errors.append(ActiveError(
            type=row[0],
            message=row[2] or "Unknown error",
            occurred_at=row[1],
        ))

    model_versions = [v for v in [
        _model_version(run_mv),
        _model_version(pitcher_mv),
        _model_version(batter_mv),
    ] if v is not None]

    return SystemStatus(
        last_morning_refresh=_refresh_status(morning),
        last_afternoon_refresh=_refresh_status(afternoon),
        last_retrain=_refresh_status(retrain),
        db_last_updated=db_updated,
        active_model_versions=model_versions,
        active_errors=active_errors,
    )
