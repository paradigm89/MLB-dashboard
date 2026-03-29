"""
Reusable query helpers for common DB access patterns.
"""
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from src.db.schema import (
    BattingStats,
    EloRating,
    FieldingStats,
    Game,
    Lineup,
    ModelVersion,
    ParkFactor,
    PitchingStats,
    Prediction,
    RefreshLog,
    StatcastPitch,
)


def get_games_for_season(session: Session, season: int) -> pd.DataFrame:
    rows = session.query(Game).filter(Game.season == season, Game.status == "Final").all()
    return _to_df(rows)


def get_games_date_range(session: Session, start: str, end: str) -> pd.DataFrame:
    rows = (
        session.query(Game)
        .filter(Game.game_date >= start, Game.game_date <= end, Game.status == "Final")
        .all()
    )
    return _to_df(rows)


def get_batting_stats(session: Session, season: int) -> pd.DataFrame:
    rows = session.query(BattingStats).filter(BattingStats.season == season).all()
    return _to_df(rows)


def get_pitching_stats(session: Session, season: int) -> pd.DataFrame:
    rows = session.query(PitchingStats).filter(PitchingStats.season == season).all()
    return _to_df(rows)


def get_pitcher_stats_multi_season(session: Session, player_id: int, seasons: list[int]) -> pd.DataFrame:
    rows = (
        session.query(PitchingStats)
        .filter(PitchingStats.player_id == player_id, PitchingStats.season.in_(seasons))
        .all()
    )
    return _to_df(rows)


def get_statcast_pitcher(session: Session, pitcher_id: int, start_date: str, end_date: str) -> pd.DataFrame:
    rows = (
        session.query(StatcastPitch)
        .filter(
            StatcastPitch.pitcher_id == pitcher_id,
            StatcastPitch.game_date >= start_date,
            StatcastPitch.game_date <= end_date,
        )
        .all()
    )
    return _to_df(rows)


def get_statcast_matchup(session: Session, batter_id: int, pitcher_id: int) -> pd.DataFrame:
    """All pitch-by-pitch data for a specific batter vs pitcher matchup."""
    rows = (
        session.query(StatcastPitch)
        .filter(
            StatcastPitch.batter_id == batter_id,
            StatcastPitch.pitcher_id == pitcher_id,
        )
        .all()
    )
    return _to_df(rows)


def get_team_relievers(
    session: Session, team_id: int, season: int, min_g: int = 5
) -> pd.DataFrame:
    """
    Return FanGraphs pitching stats for relievers on a given team.
    Relievers are identified as pitchers with gs=0 (pure relievers) and at
    least min_g appearances, which filters out emergency call-ups.
    """
    rows = (
        session.query(PitchingStats)
        .filter(
            PitchingStats.team_id == team_id,
            PitchingStats.season == season,
            PitchingStats.g >= min_g,
            PitchingStats.gs == 0,
        )
        .all()
    )
    return _to_df(rows)


def get_all_reliever_ids(
    session: Session, min_season: int, min_g: int = 5
) -> list[int]:
    """
    Return all pitcher IDs known as pure relievers (gs=0) across recent seasons.
    Used to filter Statcast data to relief appearances for batter matchup blending.
    """
    rows = (
        session.query(PitchingStats.player_id)
        .filter(
            PitchingStats.season >= min_season,
            PitchingStats.g >= min_g,
            PitchingStats.gs == 0,
        )
        .distinct()
        .all()
    )
    return [r[0] for r in rows]


def get_statcast_batter_vs_relievers(
    session: Session,
    batter_id: int,
    pitcher_hand: str,
    reliever_ids: list[int],
    min_season: int,
) -> pd.DataFrame:
    """
    Return all Statcast pitches for a batter against known relievers of the
    given handedness, filtered to rows with an xwOBA value (i.e. balls in play
    or strikeouts with an estimated value).  Used for Bayesian-blended
    batter-vs-reliever xwOBA.
    """
    if not reliever_ids:
        return pd.DataFrame()
    rows = (
        session.query(StatcastPitch)
        .filter(
            StatcastPitch.batter_id == batter_id,
            StatcastPitch.pitcher_id.in_(reliever_ids),
            StatcastPitch.p_throws == pitcher_hand,
            StatcastPitch.season >= min_season,
            StatcastPitch.estimated_woba_using_speedangle.isnot(None),
        )
        .all()
    )
    return _to_df(rows)


def get_park_factor(session: Session, venue_id: int, season: int) -> Optional[ParkFactor]:
    return (
        session.query(ParkFactor)
        .filter(ParkFactor.venue_id == venue_id, ParkFactor.season == season)
        .first()
    )


def get_latest_elo(session: Session, team_id: int, before_date: str) -> Optional[float]:
    row = (
        session.query(EloRating)
        .filter(EloRating.team_id == team_id, EloRating.game_date < before_date)
        .order_by(EloRating.game_date.desc())
        .first()
    )
    return row.elo if row else 1500.0


def get_lineup(session: Session, game_pk: int, team_id: int) -> Optional[Lineup]:
    return (
        session.query(Lineup)
        .filter(Lineup.game_pk == game_pk, Lineup.team_id == team_id)
        .first()
    )


def get_recent_games_for_team(session: Session, team_id: int, before_date: str, n: int = 30) -> pd.DataFrame:
    rows = (
        session.query(Game)
        .filter(
            ((Game.home_team_id == team_id) | (Game.away_team_id == team_id)),
            Game.game_date < before_date,
            Game.status == "Final",
        )
        .order_by(Game.game_date.desc())
        .limit(n)
        .all()
    )
    return _to_df(rows)


def get_prediction(session: Session, game_pk: int) -> Optional[Prediction]:
    return (
        session.query(Prediction)
        .filter(Prediction.game_pk == game_pk)
        .order_by(Prediction.generated_at.desc())
        .first()
    )


def get_last_refresh(session: Session, refresh_type: str) -> Optional[RefreshLog]:
    return (
        session.query(RefreshLog)
        .filter(RefreshLog.refresh_type == refresh_type)
        .order_by(RefreshLog.started_at.desc())
        .first()
    )


def get_active_model_version(session: Session, model_type: str) -> Optional[ModelVersion]:
    return (
        session.query(ModelVersion)
        .filter(ModelVersion.model_type == model_type, ModelVersion.is_active == True)
        .order_by(ModelVersion.trained_at.desc())
        .first()
    )


def get_today_predictions(session: Session, game_date: str) -> list[dict]:
    rows = (
        session.query(Prediction)
        .filter(Prediction.game_date == game_date)
        .order_by(Prediction.generated_at.desc())
        .all()
    )
    seen = set()
    results = []
    for r in rows:
        if r.game_pk not in seen:
            seen.add(r.game_pk)
            results.append(r)
    return results


def _to_df(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    data = []
    for r in rows:
        row_dict = {c.name: getattr(r, c.name) for c in r.__table__.columns}
        data.append(row_dict)
    return pd.DataFrame(data)
