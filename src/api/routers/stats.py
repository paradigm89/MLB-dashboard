"""
Stats endpoints.

GET /api/stats/team/{team_id}    — rolling team stats + Elo
GET /api/stats/player/{player_id} — player season stats
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import TeamStats
from src.db.connection import get_db
from src.db.queries import get_batting_stats, get_latest_elo, get_pitching_stats, get_recent_games_for_team
from src.pipeline.features import _build_team_rolling_features

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/stats")


@router.get("/team/{team_id}", response_model=TeamStats)
async def get_team_stats(
    team_id: int,
    season: Optional[int] = Query(None, description="Season year; defaults to current season"),
):
    """Return rolling team stats and Elo rating."""
    from datetime import date
    if season is None:
        season = date.today().year
    today = str(date.today())

    rolling = _build_team_rolling_features(team_id, today)
    elo = None
    with get_db() as session:
        elo = get_latest_elo(session, team_id, today)

    # Get team abbreviation
    team_abbr = None
    with get_db() as session:
        from sqlalchemy import text
        result = session.execute(
            text("SELECT home_team_abbr FROM games WHERE home_team_id = :tid LIMIT 1"),
            {"tid": team_id},
        ).fetchone()
        if result:
            team_abbr = result[0]

    return TeamStats(
        team_id=team_id,
        team_abbr=team_abbr,
        season=season,
        elo=elo,
        runs_scored_30g=rolling.get("runs_scored_30g"),
        runs_allowed_30g=rolling.get("runs_allowed_30g"),
        era_30g=rolling.get("era_30g"),
        whip_30g=rolling.get("whip_30g"),
        pythagorean_win_pct=rolling.get("pythagorean_win_pct"),
        bullpen_era_7d=rolling.get("bullpen_era_7d"),
    )


@router.get("/player/{player_id}")
async def get_player_stats(
    player_id: int,
    season: Optional[int] = Query(None),
    stat_type: str = Query("batting", description="batting or pitching"),
):
    """Return player season stats."""
    from datetime import date
    if season is None:
        season = date.today().year

    with get_db() as session:
        if stat_type == "pitching":
            df = get_pitching_stats(session, season)
        else:
            df = get_batting_stats(session, season)

    if df.empty:
        raise HTTPException(status_code=404, detail="No stats found")

    player_row = df[df["player_id"] == player_id]
    if player_row.empty:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found in {season}")

    return player_row.iloc[0].to_dict()
