"""
MLB-StatsAPI adapter.

Wraps the official statsapi.mlb.com API for schedules, boxscores,
rosters, and game metadata. No API key required.

Honest caveat: the MLB Stats API does not reliably surface pre-game
umpire assignments — this adapter returns None for home_plate_umpire_id
on upcoming games.
"""
import logging
from datetime import datetime
from typing import Optional

import statsapi

# Module-level cache: player name → MLB player ID.
_player_id_cache: dict[str, Optional[int]] = {}

# Static team ID → abbreviation map (MLB team IDs are stable across seasons).
_TEAM_ABBREVS: dict[int, str] = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC",  119: "LAD", 120: "WSH", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SD",  136: "SEA", 137: "SF",  138: "STL",
    139: "TB",  140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}

import pandas as pd

from src.pipeline.adapters.base import DataAdapter

logger = logging.getLogger(__name__)


class MLBStatsAdapter(DataAdapter):
    """
    Adapter for the official MLB Stats API via the MLB-StatsAPI Python wrapper.
    Handles schedules, boxscores, probable pitchers, and confirmed lineups.
    """

    # --- Schedule / game metadata -------------------------------------------

    def get_schedule(self, date: str) -> list[dict]:
        """
        Return all games scheduled for `date` (YYYY-MM-DD) including probable pitchers.
        """
        try:
            raw = statsapi.schedule(date=date, sportId=1)
        except Exception as exc:
            logger.error("MLB-StatsAPI schedule fetch failed for %s: %s", date, exc)
            return []

        games = []
        for g in raw:
            home_id = g.get("home_id")
            away_id = g.get("away_id")
            # statsapi doesn't return abbreviations — use static map as primary source
            home_abbr = (g.get("home_abbrev") or g.get("home_abbreviation")
                         or _TEAM_ABBREVS.get(home_id, ""))
            away_abbr = (g.get("away_abbrev") or g.get("away_abbreviation")
                         or _TEAM_ABBREVS.get(away_id, ""))
            home_sp_id, home_sp_name = self._extract_pitcher_info(g.get("home_probable_pitcher"))
            away_sp_id, away_sp_name = self._extract_pitcher_info(g.get("away_probable_pitcher"))
            games.append({
                "game_pk": g["game_id"],
                "game_date": date,
                "game_type": g.get("game_type", "R"),
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_team_abbr": home_abbr,
                "away_team_abbr": away_abbr,
                "venue_id": g.get("venue_id"),
                "venue_name": g.get("venue_name", ""),
                "status": g.get("status", ""),
                "home_sp_id": home_sp_id,
                "home_sp_name": home_sp_name,
                "away_sp_id": away_sp_id,
                "away_sp_name": away_sp_name,
            })
        return games

    def get_game_boxscore(self, game_pk: int) -> dict:
        """
        Parse boxscore for confirmed lineups, weather, umpire, and final score.
        """
        try:
            raw = statsapi.boxscore_data(game_pk)
        except Exception as exc:
            logger.error("Boxscore fetch failed for game_pk=%s: %s", game_pk, exc)
            return self._empty_boxscore()

        home = raw.get("home", {})
        away = raw.get("away", {})
        info = raw.get("gameInfo", {})
        weather = raw.get("weather", {})

        # Batting order: list of player IDs in order (1-9)
        home_order = [p["person"]["id"] for p in home.get("players", {}).values()
                      if p.get("battingOrder") and str(p["battingOrder"]).endswith("00")]
        away_order = [p["person"]["id"] for p in away.get("players", {}).values()
                      if p.get("battingOrder") and str(p["battingOrder"]).endswith("00")]

        # Sort by batting order position
        home_order = self._sort_batting_order(home, home_order)
        away_order = self._sort_batting_order(away, away_order)

        # Umpires
        umpire_id = None
        umpire_name = None
        for ump in raw.get("officials", []):
            if ump.get("officialType") == "Home Plate":
                umpire_id = ump.get("official", {}).get("id")
                umpire_name = ump.get("official", {}).get("fullName")
                break

        home_score = home.get("teamStats", {}).get("batting", {}).get("runs")
        away_score = away.get("teamStats", {}).get("batting", {}).get("runs")

        # Actual starting pitchers: first ID in each team's pitchers list.
        # This is only populated once the game is live/complete; pre-game it's empty.
        home_pitchers = home.get("pitchers", [])
        away_pitchers = away.get("pitchers", [])
        home_sp_id = int(home_pitchers[0]) if home_pitchers else None
        away_sp_id = int(away_pitchers[0]) if away_pitchers else None

        return {
            "home_score": home_score,
            "away_score": away_score,
            "home_win": (home_score > away_score) if (home_score is not None and away_score is not None) else None,
            "temperature": self._parse_temperature(weather.get("temp")),
            "wind_speed": self._parse_wind_speed(weather.get("wind", "")),
            "wind_direction": weather.get("wind", ""),
            "roof_status": info.get("weather", {}).get("condition", ""),
            "home_plate_umpire_id": umpire_id,
            "home_plate_umpire_name": umpire_name,
            "home_batting_order": home_order,
            "away_batting_order": away_order,
            "home_lineup_confirmed": len(home_order) == 9,
            "away_lineup_confirmed": len(away_order) == 9,
            # Confirmed starters (None pre-game; populated once game begins)
            "home_sp_id": home_sp_id,
            "away_sp_id": away_sp_id,
        }

    def get_completed_games(self, date: str) -> list[dict]:
        """
        Return completed games for a date with final scores.
        Used by morning refresh to record prior day results.
        """
        games = self.get_schedule(date)
        result = []
        for g in games:
            if g["status"] == "Final":
                boxscore = self.get_game_boxscore(g["game_pk"])
                g.update(boxscore)
                result.append(g)
        return result

    # --- Stubs for methods handled by PybaseballAdapter ---------------------
    # MLB-StatsAPI doesn't provide Statcast or FanGraphs batting/pitching stats.
    # These raise NotImplementedError to make it clear which adapter owns each method.

    def get_season_batting_stats(self, season: int) -> pd.DataFrame:
        raise NotImplementedError("Use PybaseballAdapter for batting stats")

    def get_season_pitching_stats(self, season: int) -> pd.DataFrame:
        raise NotImplementedError("Use PybaseballAdapter for pitching stats")

    def get_season_fielding_stats(self, season: int) -> pd.DataFrame:
        raise NotImplementedError("Use PybaseballAdapter for fielding stats")

    def get_statcast_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError("Use PybaseballAdapter for Statcast data")

    def get_park_factors(self, season: int) -> pd.DataFrame:
        raise NotImplementedError("Use PybaseballAdapter for park factors")

    # --- Helpers ------------------------------------------------------------

    @staticmethod
    def _extract_pitcher_info(value) -> tuple[Optional[int], Optional[str]]:
        """
        Returns (pitcher_id, pitcher_name) from statsapi probable_pitcher value.
        Value can be a dict {"id": 123, "fullName": "..."}, a plain string name,
        a numeric id, or None.
        """
        if value is None or value == "":
            return None, None
        if isinstance(value, dict):
            return value.get("id"), value.get("fullName")
        if isinstance(value, (int, float)):
            return int(value), None
        # String name — resolve to ID, keep the name as-is
        name = str(value).strip()
        return MLBStatsAdapter._lookup_player_id_by_name(name), name

    @staticmethod
    def _extract_pitcher_id(value) -> Optional[int]:
        """Kept for backward compatibility — delegates to _extract_pitcher_info."""
        return MLBStatsAdapter._extract_pitcher_info(value)[0]

    @staticmethod
    def _lookup_player_id_by_name(name: str) -> Optional[int]:
        """
        Resolve a player's full name to their MLB player ID.

        Uses a module-level cache so each unique name is only looked up once
        per process.  Returns None for genuinely new players with no entry yet
        (e.g. debut game) — the prediction pipeline handles None SP gracefully.
        """
        if not name:
            return None
        if name in _player_id_cache:
            return _player_id_cache[name]
        try:
            results = statsapi.lookup_player(name)
        except Exception as exc:
            logger.warning("Player ID lookup failed for %r: %s", name, exc)
            _player_id_cache[name] = None
            return None

        if not results:
            logger.warning("No MLB player found for name %r (may be a debut)", name)
            _player_id_cache[name] = None
            return None

        # Prefer an exact case-insensitive full-name match
        name_lower = name.lower()
        for r in results:
            if r.get("fullName", "").lower() == name_lower:
                pid = int(r["id"])
                _player_id_cache[name] = pid
                return pid

        # Multiple partial matches — take the first (active players rank higher
        # in the statsapi response, so this is usually correct)
        pid = int(results[0]["id"])
        logger.debug(
            "Ambiguous player name %r — using id=%d (%s); %d candidates",
            name, pid, results[0].get("fullName"), len(results),
        )
        _player_id_cache[name] = pid
        return pid

    def _sort_batting_order(self, team_data: dict, player_ids: list) -> list:
        """Sort player IDs by battingOrder field from the boxscore."""
        order_map = {}
        for p in team_data.get("players", {}).values():
            bo = p.get("battingOrder")
            pid = p.get("person", {}).get("id")
            if bo and pid and str(bo).endswith("00"):
                order_map[pid] = int(str(bo)) // 100
        return sorted(player_ids, key=lambda pid: order_map.get(pid, 99))

    @staticmethod
    def _parse_temperature(temp_str: Optional[str]) -> Optional[float]:
        if not temp_str:
            return None
        try:
            return float(str(temp_str).replace("°", "").strip())
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_wind_speed(wind_str: str) -> Optional[float]:
        """Extract numeric wind speed from strings like '12 mph, Out To CF'."""
        if not wind_str:
            return None
        try:
            return float(wind_str.split()[0])
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _empty_boxscore() -> dict:
        return {
            "home_score": None, "away_score": None, "home_win": None,
            "temperature": None, "wind_speed": None, "wind_direction": None,
            "roof_status": None, "home_plate_umpire_id": None,
            "home_plate_umpire_name": None,
            "home_batting_order": [], "away_batting_order": [],
            "home_lineup_confirmed": False, "away_lineup_confirmed": False,
        }
