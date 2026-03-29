"""
Lineup sourcing: official confirmed lineups and projected lineup fallback.

Lineup availability timeline:
- Probable starting pitchers: available night before or morning of game
- Official batting orders: posted by teams 3–20 minutes before first pitch
- We refresh at 9 AM (projected), 3 PM, and 5 PM ET (catching confirmed lineups)

Honest caveat: West Coast night games (first pitch 7-10 PM PT = 10 PM–1 AM ET)
often don't have confirmed lineups by our 5 PM refresh. Those predictions run
on projected lineups and are clearly labeled in the UI.
"""
import logging
from collections import Counter
from datetime import datetime
from typing import Optional

from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.db.connection import get_db
from src.db.queries import get_lineup, get_recent_games_for_team
from src.db.schema import Lineup
from src.pipeline.adapters.mlb_stats import MLBStatsAdapter

logger = logging.getLogger(__name__)

_mlb = MLBStatsAdapter()


def fetch_and_store_lineups(game_pk: int, home_team_id: int, away_team_id: int,
                             game_date: str, home_sp_id: Optional[int],
                             away_sp_id: Optional[int]) -> dict:
    """
    Try to get official lineups for a game; fall back to projected if not confirmed.
    Writes results to the `lineups` table.

    Returns:
        dict with keys: home_lineup (list of player_ids), away_lineup,
        home_confirmed (bool), away_confirmed (bool)
    """
    boxscore = _mlb.get_game_boxscore(game_pk)
    home_order = boxscore.get("home_batting_order", [])
    away_order = boxscore.get("away_batting_order", [])
    home_confirmed = boxscore.get("home_lineup_confirmed", False)
    away_confirmed = boxscore.get("away_lineup_confirmed", False)

    # If the boxscore already has the actual starter (game is live/started),
    # prefer that over the schedule's probable pitcher.
    home_sp_id = boxscore.get("home_sp_id") or home_sp_id
    away_sp_id = boxscore.get("away_sp_id") or away_sp_id

    # Fall back to projected lineup for any side not yet confirmed
    if not home_confirmed or len(home_order) != 9:
        home_order = get_projected_lineup(home_team_id, game_date)
        home_confirmed = False

    if not away_confirmed or len(away_order) != 9:
        away_order = get_projected_lineup(away_team_id, game_date)
        away_confirmed = False

    _upsert_lineup(game_pk, game_date, home_team_id, is_home=True,
                   batting_order=home_order, sp_id=home_sp_id,
                   is_confirmed=home_confirmed)
    _upsert_lineup(game_pk, game_date, away_team_id, is_home=False,
                   batting_order=away_order, sp_id=away_sp_id,
                   is_confirmed=away_confirmed)

    return {
        "home_lineup": home_order,
        "away_lineup": away_order,
        "home_confirmed": home_confirmed,
        "away_confirmed": away_confirmed,
        # Pass resolved SP IDs back so _process_game_prediction can use the
        # best available pitcher (boxscore-confirmed > schedule probable > None)
        "home_sp_id": home_sp_id,
        "away_sp_id": away_sp_id,
    }


def check_and_update_confirmed_lineups(game_pk: int, home_team_id: int,
                                        away_team_id: int, game_date: str) -> dict:
    """
    Called during afternoon/evening refresh to check if official lineups
    have been posted. Updates DB and returns whether lineups changed.

    Returns:
        dict with keys: home_updated (bool), away_updated (bool),
        home_lineup, away_lineup, home_confirmed, away_confirmed
    """
    boxscore = _mlb.get_game_boxscore(game_pk)
    home_order = boxscore.get("home_batting_order", [])
    away_order = boxscore.get("away_batting_order", [])
    home_confirmed = boxscore.get("home_lineup_confirmed", False)
    away_confirmed = boxscore.get("away_lineup_confirmed", False)

    # Actual starters from boxscore (populated once game is live)
    home_sp_id = boxscore.get("home_sp_id")
    away_sp_id = boxscore.get("away_sp_id")

    home_updated = False
    away_updated = False

    # Extract all needed attributes inside the session to avoid DetachedInstanceError
    with get_db() as session:
        existing_home = get_lineup(session, game_pk, home_team_id)
        existing_away = get_lineup(session, game_pk, away_team_id)

        prev_home_sp = existing_home.sp_id if existing_home else None
        prev_away_sp = existing_away.sp_id if existing_away else None
        prev_home_confirmed = bool(existing_home.is_confirmed) if existing_home else False
        prev_away_confirmed = bool(existing_away.is_confirmed) if existing_away else False
        prev_home_order = existing_home.batting_order if existing_home else ""
        prev_away_order = existing_away.batting_order if existing_away else ""

    # Resolve SP IDs: boxscore confirmed > previously stored probable
    if home_sp_id is None and prev_home_sp:
        home_sp_id = prev_home_sp
    if away_sp_id is None and prev_away_sp:
        away_sp_id = prev_away_sp

    # Only update if we got a confirmed lineup and it wasn't already confirmed
    if home_confirmed and len(home_order) == 9:
        if not prev_home_confirmed:
            _upsert_lineup(game_pk, game_date, home_team_id, is_home=True,
                           batting_order=home_order, sp_id=home_sp_id,
                           is_confirmed=True)
            home_updated = True

    if away_confirmed and len(away_order) == 9:
        if not prev_away_confirmed:
            _upsert_lineup(game_pk, game_date, away_team_id, is_home=False,
                           batting_order=away_order, sp_id=away_sp_id,
                           is_confirmed=True)
            away_updated = True

    return {
        "home_updated": home_updated,
        "away_updated": away_updated,
        "home_lineup": home_order if home_confirmed else (prev_home_order.split(",") if prev_home_order else []),
        "away_lineup": away_order if away_confirmed else (prev_away_order.split(",") if prev_away_order else []),
        "home_confirmed": home_confirmed,
        "away_confirmed": away_confirmed,
        "home_sp_id": home_sp_id,
        "away_sp_id": away_sp_id,
    }


def get_projected_lineup(team_id: int, game_date: str, n_games: int = 10) -> list[int]:
    """
    Build a projected batting order from the team's most common recent lineup.
    Uses the most frequent player at each batting order position over the last
    n_games games. Falls back to an empty list if no history exists.

    Honest limitation: this doesn't account for day-off/rest patterns,
    platoon situations, or roster moves. It's a reasonable pre-game estimate
    but will differ from the actual lineup for ~20-30% of positions.
    """
    with get_db() as session:
        recent = get_recent_games_for_team(session, team_id, before_date=game_date, n=n_games)

    if recent.empty:
        logger.warning("No recent game history for team_id=%d before %s", team_id, game_date)
        return []

    # Collect batting orders from stored lineups for these games
    game_pks = recent["game_pk"].tolist()
    position_votes: dict[int, Counter] = {pos: Counter() for pos in range(1, 10)}

    with get_db() as session:
        for gpk in game_pks:
            lineup_row = get_lineup(session, gpk, team_id)
            if lineup_row is None or not lineup_row.batting_order:
                continue
            player_ids = [int(p) for p in lineup_row.batting_order.split(",") if p.strip()]
            for pos, pid in enumerate(player_ids[:9], start=1):
                position_votes[pos][pid] += 1

    # Pick the most common player at each position
    projected = []
    used_players = set()
    for pos in range(1, 10):
        if not position_votes[pos]:
            continue
        # Most common player at this slot who hasn't already been placed
        for pid, _ in position_votes[pos].most_common():
            if pid not in used_players:
                projected.append(pid)
                used_players.add(pid)
                break

    return projected


def get_lineup_player_ids(game_pk: int, team_id: int) -> list[int]:
    """
    Return the stored batting order for a game/team as a list of player_ids.
    Returns empty list if no lineup is recorded.
    """
    with get_db() as session:
        row = get_lineup(session, game_pk, team_id)
    if row is None or not row.batting_order:
        return []
    return [int(p) for p in row.batting_order.split(",") if p.strip()]


# ---------------------------------------------------------------------------
# DB write helper
# ---------------------------------------------------------------------------

def _upsert_lineup(game_pk: int, game_date: str, team_id: int, is_home: bool,
                   batting_order: list[int], sp_id: Optional[int],
                   is_confirmed: bool) -> None:
    batting_order_str = ",".join(str(p) for p in batting_order)
    source = "official" if is_confirmed else "projected"

    with get_db() as session:
        stmt = (
            pg_insert(Lineup.__table__)
            .values(
                game_pk=game_pk,
                game_date=game_date,
                team_id=team_id,
                is_home=is_home,
                is_confirmed=is_confirmed,
                batting_order=batting_order_str,
                sp_id=sp_id,
                source=source,
                confirmed_at=datetime.utcnow() if is_confirmed else None,
            )
            .on_conflict_do_update(
                index_elements=["game_pk", "team_id"],
                set_={
                    "is_confirmed": is_confirmed,
                    "batting_order": batting_order_str,
                    "source": source,
                    "confirmed_at": datetime.utcnow() if is_confirmed else None,
                    "updated_at": datetime.utcnow(),
                },
            )
        )
        session.execute(stmt)
