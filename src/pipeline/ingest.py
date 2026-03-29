"""
Historical and incremental data ingestion pipeline.

Fetches all data from pybaseball + MLB-StatsAPI and writes to PostgreSQL.
Designed to be resumable: tracks last successfully ingested date in the
DB so a network failure doesn't require starting from scratch.

Usage:
    # Full historical load (takes several hours on first run):
    python -m src.pipeline.ingest --seasons 2019 2025

    # Incremental: ingest a single date (called by daily_refresh.py):
    python -m src.pipeline.ingest --date 2026-03-28
"""
import argparse
import logging
import sys
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.connection import get_db
from src.db.schema import (
    BattingStats,
    EloRating,
    FieldingStats,
    Game,
    ParkFactor,
    PitchingStats,
    StatcastPitch,
)
from src.pipeline.adapters.mlb_stats import MLBStatsAdapter
from src.pipeline.adapters.pybaseball import PybaseballAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Earliest season with reliable Statcast data
MIN_SEASON = 2019
# Statcast chunk size: fetch in 14-day windows to avoid timeouts
STATCAST_CHUNK_DAYS = 14


def run_historical_ingest(start_season: int, end_season: int) -> None:
    """
    Full historical ingest for seasons [start_season, end_season].
    Safe to re-run: uses upsert (ON CONFLICT DO NOTHING or DO UPDATE).
    """
    pb = PybaseballAdapter()
    mlb = MLBStatsAdapter()

    for season in range(start_season, end_season + 1):
        logger.info("=== Ingesting season %d ===", season)

        # 1. Season-level player stats (fast — one request per category)
        _ingest_batting_stats(pb, season)
        _ingest_pitching_stats(pb, season)
        _ingest_fielding_stats(pb, season)
        _ingest_park_factors(pb, mlb, season)

        # 2. Game results
        _ingest_season_games(mlb, season)

        # 3. Statcast pitch-by-pitch (slow — chunked by date)
        _ingest_statcast_season(pb, season)

    logger.info("Historical ingest complete for seasons %d–%d", start_season, end_season)


def run_daily_ingest(target_date: str) -> None:
    """
    Incremental ingest for a single completed game day.
    Called by daily_refresh.py each morning.
    """
    mlb = MLBStatsAdapter()
    pb = PybaseballAdapter()

    logger.info("Running daily ingest for %s", target_date)
    completed = mlb.get_completed_games(target_date)
    _upsert_games(completed)
    _update_elo_for_date(target_date)

    # Statcast for the previous day (data is available ~12 hours after games end)
    _ingest_statcast_range(pb, target_date, target_date)
    logger.info("Daily ingest complete for %s", target_date)


# ---------------------------------------------------------------------------
# Per-data-type ingest helpers
# ---------------------------------------------------------------------------

def _ingest_batting_stats(pb: PybaseballAdapter, season: int) -> None:
    df = pb.get_season_batting_stats(season)
    if df.empty:
        logger.warning("No batting stats returned for %d", season)
        return
    _upsert_dataframe(df, BattingStats, conflict_cols=["player_id", "season"])
    logger.info("Upserted %d batting stat rows for %d", len(df), season)


def _ingest_pitching_stats(pb: PybaseballAdapter, season: int) -> None:
    df = pb.get_season_pitching_stats(season)
    if df.empty:
        logger.warning("No pitching stats returned for %d", season)
        return
    _upsert_dataframe(df, PitchingStats, conflict_cols=["player_id", "season"])
    logger.info("Upserted %d pitching stat rows for %d", len(df), season)


def _ingest_fielding_stats(pb: PybaseballAdapter, season: int) -> None:
    df = pb.get_season_fielding_stats(season)
    if df.empty:
        logger.warning("No fielding stats returned for %d", season)
        return
    _upsert_dataframe(df, FieldingStats, conflict_cols=["player_id", "season", "position"])
    logger.info("Upserted %d fielding stat rows for %d", len(df), season)


def _ingest_park_factors(pb: PybaseballAdapter, mlb: MLBStatsAdapter, season: int) -> None:
    df = pb.get_park_factors(season)
    if df.empty:
        logger.warning("No park factors returned for %d", season)
        return
    # Park factors from pybaseball don't include venue_id; we leave it null here.
    # The feature engineering layer joins on team_abbr + season instead.
    _upsert_dataframe(df, ParkFactor, conflict_cols=["venue_id", "season"])
    logger.info("Upserted park factors for %d", season)


def _ingest_season_games(mlb: MLBStatsAdapter, season: int) -> None:
    """Fetch all regular season game results for a full season."""
    start = date(season, 3, 20)  # Spring training ends ~late March
    end = date(season, 11, 15)   # Postseason ends by November
    current = start
    all_games = []
    while current <= end:
        batch_end = min(current + timedelta(days=6), end)
        games = mlb.get_completed_games(str(current))
        all_games.extend(games)
        current = batch_end + timedelta(days=1)

    _upsert_games(all_games)
    logger.info("Upserted %d game results for %d", len(all_games), season)


def _upsert_games(games: list[dict]) -> None:
    if not games:
        return
    with get_db() as session:
        for g in games:
            row = {c: g.get(c) for c in [
                "game_pk", "game_date", "season", "game_type",
                "home_team_id", "away_team_id", "home_team_abbr", "away_team_abbr",
                "home_score", "away_score", "home_win",
                "venue_id", "venue_name",
                "temperature", "wind_speed", "wind_direction", "roof_status",
                "home_sp_id", "away_sp_id", "home_plate_umpire_id", "status",
            ]}
            # Derive season from game_date if not set
            if not row.get("season") and row.get("game_date"):
                row["season"] = int(row["game_date"][:4])

            stmt = (
                pg_insert(Game.__table__)
                .values(**{k: v for k, v in row.items() if v is not None or k == "game_pk"})
                .on_conflict_do_update(
                    index_elements=["game_pk"],
                    set_={k: v for k, v in row.items() if k != "game_pk"},
                )
            )
            session.execute(stmt)


def _ingest_statcast_season(pb: PybaseballAdapter, season: int) -> None:
    """
    Fetch Statcast data for a full season in STATCAST_CHUNK_DAYS-day windows.
    Tracks progress: skips chunks already present in DB.
    """
    start = date(season, 3, 28)
    end = date(season, 11, 5)
    current = start

    with get_db() as session:
        last_date = _get_last_statcast_date(session, season)

    if last_date:
        # Resume from day after last successfully ingested date
        resume_from = date.fromisoformat(last_date) + timedelta(days=1)
        if resume_from > current:
            current = resume_from
            logger.info("Resuming Statcast ingest for %d from %s", season, current)

    while current <= end:
        chunk_end = min(current + timedelta(days=STATCAST_CHUNK_DAYS - 1), end)
        _ingest_statcast_range(pb, str(current), str(chunk_end))
        current = chunk_end + timedelta(days=1)


def _ingest_statcast_range(pb: PybaseballAdapter, start_date: str, end_date: str) -> None:
    df = pb.get_statcast_range(start_date, end_date)
    if df.empty:
        return

    # Bulk insert; Statcast rows are immutable so we skip conflicts
    with get_db() as session:
        records = df.to_dict(orient="records")
        stmt = pg_insert(StatcastPitch.__table__).values(records).on_conflict_do_nothing()
        session.execute(stmt)

    logger.info("Inserted %d Statcast pitches (%s to %s)", len(df), start_date, end_date)


def _get_last_statcast_date(session, season: int) -> Optional[str]:
    """Return the most recent game_date in statcast_pitches for this season, or None."""
    from sqlalchemy import func, text
    result = session.execute(
        text("SELECT MAX(game_date) FROM statcast_pitches WHERE season = :season"),
        {"season": season},
    ).scalar()
    return str(result) if result else None


def _update_elo_for_date(game_date: str) -> None:
    """
    Recompute Elo ratings for all teams after games on game_date.
    Uses K=20 (standard for MLB). Starting Elo = 1500.
    """
    from src.db.queries import get_games_date_range
    K = 20
    with get_db() as session:
        games = get_games_date_range(session, game_date, game_date)
        if games.empty:
            return

        for _, g in games.iterrows():
            if g["home_win"] is None:
                continue

            home_elo = _get_current_elo(session, g["home_team_id"], game_date)
            away_elo = _get_current_elo(session, g["away_team_id"], game_date)

            # Expected win probability via Elo formula
            exp_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
            exp_away = 1 - exp_home

            actual_home = 1.0 if g["home_win"] else 0.0
            actual_away = 1.0 - actual_home

            new_home = home_elo + K * (actual_home - exp_home)
            new_away = away_elo + K * (actual_away - exp_away)

            for team_id, new_elo in [(g["home_team_id"], new_home), (g["away_team_id"], new_away)]:
                stmt = pg_insert(EloRating.__table__).values(
                    team_id=int(team_id),
                    after_game_pk=int(g["game_pk"]),
                    game_date=game_date,
                    season=int(game_date[:4]),
                    elo=new_elo,
                ).on_conflict_do_nothing()
                session.execute(stmt)


def _get_current_elo(session, team_id: int, before_date: str) -> float:
    from src.db.queries import get_latest_elo
    return get_latest_elo(session, team_id, before_date) or 1500.0


# ---------------------------------------------------------------------------
# Generic upsert helper
# ---------------------------------------------------------------------------

def _upsert_dataframe(df: pd.DataFrame, model_class, conflict_cols: list[str]) -> None:
    """
    Upsert a DataFrame into a SQLAlchemy model's table.
    Uses PostgreSQL ON CONFLICT DO UPDATE on the specified conflict columns.
    Columns in the DataFrame that don't exist in the table are silently dropped.
    """
    table = model_class.__table__
    table_cols = {c.name for c in table.columns}

    with get_db() as session:
        for batch_start in range(0, len(df), 500):
            batch = df.iloc[batch_start : batch_start + 500]
            records = []
            for _, row in batch.iterrows():
                record = {k: v for k, v in row.items() if k in table_cols and pd.notna(v)}
                # Ensure conflict columns are always present
                for col in conflict_cols:
                    if col not in record and col in row:
                        record[col] = row[col]
                records.append(record)

            if not records:
                continue

            # Normalize: all records in a batch must have the same keys.
            # pg_insert().values(list) uses the first record's keys for all rows;
            # rows missing a key cause a CompileError. Fill gaps with None (SQL NULL).
            all_keys = set().union(*records)
            records = [{k: rec.get(k, None) for k in all_keys} for rec in records]

            # Build update set (all non-conflict columns)
            update_cols = [c for c in all_keys if c not in conflict_cols and c != "id"]
            stmt = (
                pg_insert(table)
                .values(records)
                .on_conflict_do_update(
                    index_elements=conflict_cols,
                    set_={col: pg_insert(table).excluded[col] for col in update_cols},
                )
            )
            session.execute(stmt)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB data ingestion pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seasons", nargs=2, type=int, metavar=("START", "END"),
                       help="Historical ingest for a range of seasons, e.g. --seasons 2019 2025")
    group.add_argument("--date", type=str, metavar="YYYY-MM-DD",
                       help="Incremental ingest for a single date")
    args = parser.parse_args()

    if args.seasons:
        run_historical_ingest(args.seasons[0], args.seasons[1])
    elif args.date:
        run_daily_ingest(args.date)
