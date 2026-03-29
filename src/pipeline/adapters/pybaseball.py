"""
pybaseball adapter.

Wraps pybaseball (Baseball Savant + FanGraphs) for:
- Season batting/pitching/fielding stats (FanGraphs)
- Pitch-by-pitch Statcast data (Baseball Savant)
- Park factors (FanGraphs)

Honest caveat on Statcast volume: 7 seasons of data is ~5–6M rows.
The initial historical load takes several hours because Baseball Savant
limits queries to 30K rows. pybaseball handles chunking internally but
it's still slow. The ingest script tracks progress so it's resumable.
"""
import logging
import time
from datetime import date, timedelta

import pandas as pd
import pybaseball

from src.pipeline.adapters.base import DataAdapter

logger = logging.getLogger(__name__)

# Rate limiting: be a good citizen with Baseball Savant
_STATCAST_SLEEP_BETWEEN_CHUNKS = 3  # seconds between chunk requests


class PybaseballAdapter(DataAdapter):
    """
    Adapter wrapping pybaseball for FanGraphs stats and Baseball Savant Statcast data.
    """

    def __init__(self):
        # Cache pybaseball's internal ID lookups to avoid repeated web requests
        pybaseball.cache.enable()

    # --- Batting stats -------------------------------------------------------

    def get_season_batting_stats(self, season: int) -> pd.DataFrame:
        """
        FanGraphs season batting stats including Statcast expected stats.
        Uses qual=0 to get all players (not just qualified).
        """
        logger.info("Fetching FanGraphs batting stats for %d", season)
        try:
            # qual=0 = all batters; ind=1 = individual season rows
            df = pybaseball.batting_stats(season, season, qual=0, ind=1)
        except Exception as exc:
            logger.error("pybaseball batting_stats failed for %d: %s", season, exc)
            return pd.DataFrame()

        return self._normalize_batting(df, season)

    def _normalize_batting(self, df: pd.DataFrame, season: int) -> pd.DataFrame:
        # Use MLBAM ID (matches MLB Stats API / statsapi) so player lookups work
        # across the whole pipeline. IDfg is FanGraphs-only and doesn't match.
        if "MLBAM" in df.columns:
            df = df.rename(columns={"MLBAM": "player_id"})
        elif "IDfg" in df.columns:
            df = df.rename(columns={"IDfg": "player_id"})
        col_map = {
            "Name": "player_name",
            "Team": "team_abbr",
            "PA": "pa",
            "AB": "ab",
            "H": "h",
            "2B": "doubles",
            "3B": "triples",
            "HR": "hr",
            "RBI": "rbi",
            "SB": "sb",
            "BB": "bb",
            "SO": "so",
            "AVG": "avg",
            "OBP": "obp",
            "SLG": "slg",
            "OPS": "ops",
            "wOBA": "woba",
            "wRC+": "wrc_plus",
            "WAR": "war",
            "BABIP": "babip",
            "ISO": "iso",
            "BB%": "bb_pct",
            "K%": "k_pct",
            "LD%": "ld_pct",
            "GB%": "gb_pct",
            "FB%": "fb_pct",
            "xBA": "xba",
            "xSLG": "xslg",
            "xwOBA": "xwoba",
            "xwOBAcon": "xwobacon",
            "HardHit%": "hard_hit_pct",
            "Barrel%": "barrel_pct",
            "EV": "avg_exit_velocity",
            "LA": "avg_launch_angle",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df["season"] = season
        # Splits (vs LHP/RHP) — FanGraphs doesn't return splits in the same call;
        # they're populated separately in matchups.py from Statcast data.
        for col in ["wrc_plus_vs_lhp", "wrc_plus_vs_rhp", "woba_vs_lhp",
                    "woba_vs_rhp", "woba_home", "woba_away", "bats", "team_id"]:
            if col not in df.columns:
                df[col] = None
        return df

    # --- Pitching stats ------------------------------------------------------

    def get_season_pitching_stats(self, season: int) -> pd.DataFrame:
        logger.info("Fetching FanGraphs pitching stats for %d", season)
        try:
            df = pybaseball.pitching_stats(season, season, qual=0, ind=1)
        except Exception as exc:
            logger.error("pybaseball pitching_stats failed for %d: %s", season, exc)
            return pd.DataFrame()

        return self._normalize_pitching(df, season)

    def _normalize_pitching(self, df: pd.DataFrame, season: int) -> pd.DataFrame:
        # Use MLBAM ID to match statsapi player IDs used everywhere else
        if "MLBAM" in df.columns:
            df = df.rename(columns={"MLBAM": "player_id"})
        elif "IDfg" in df.columns:
            df = df.rename(columns={"IDfg": "player_id"})
        col_map = {
            "Name": "player_name",
            "Team": "team_abbr",
            "W": "w",
            "L": "l",
            "ERA": "era",
            "G": "g",
            "GS": "gs",
            "IP": "ip",
            "SO": "so",
            "BB": "bb",
            "HR": "hr",
            "WHIP": "whip",
            "FIP": "fip",
            "xFIP": "xfip",
            "SIERA": "siera",
            "K/9": "k_per_9",
            "BB/9": "bb_per_9",
            "HR/9": "hr_per_9",
            "K%": "k_pct",
            "BB%": "bb_pct",
            "K-BB%": "k_minus_bb_pct",
            "LOB%": "lob_pct",
            "BABIP": "babip",
            "xERA": "xera",
            "vFA (pi)": "avg_fastball_velo",
            "Spin Rate (pi)": "avg_spin_rate",
            "SwStr%": "swstr_pct",
            "FA% (pi)": "pct_ff",
            "SI% (pi)": "pct_si",
            "CH% (pi)": "pct_ch",
            "SL% (pi)": "pct_sl",
            "CU% (pi)": "pct_cu",
            "FC% (pi)": "pct_fc",
            "FS% (pi)": "pct_fs",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df["season"] = season
        for col in ["era_vs_lhb", "era_vs_rhb", "k_pct_vs_lhb", "k_pct_vs_rhb",
                    "throws", "team_id"]:
            if col not in df.columns:
                df[col] = None
        return df

    # --- Fielding stats ------------------------------------------------------

    def get_season_fielding_stats(self, season: int) -> pd.DataFrame:
        logger.info("Fetching FanGraphs fielding stats for %d", season)
        try:
            df = pybaseball.fielding_stats(season, season, qual=0, ind=1)
        except Exception as exc:
            logger.error("pybaseball fielding_stats failed for %d: %s", season, exc)
            return pd.DataFrame()

        if "MLBAM" in df.columns:
            df = df.rename(columns={"MLBAM": "player_id"})
        elif "IDfg" in df.columns:
            df = df.rename(columns={"IDfg": "player_id"})
        col_map = {
            "Name": "player_name",
            "Team": "team_abbr",
            "Pos": "position",
            "DRS": "drs",
            "UZR": "uzr",
            "UZR/150": "uzr_150",
            "RngR": "rng_r",
            "ErrR": "err_r",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df["season"] = season
        if "team_id" not in df.columns:
            df["team_id"] = None
        return df

    # --- Statcast pitch-by-pitch ---------------------------------------------

    def get_statcast_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch Statcast pitch-by-pitch data for a date range.

        pybaseball automatically chunks requests >5 days into sub-queries
        to stay within Baseball Savant's row limit. We add a sleep between
        calls to avoid hammering the server.
        """
        logger.info("Fetching Statcast data %s to %s", start_date, end_date)
        try:
            df = pybaseball.statcast(start_dt=start_date, end_dt=end_date, parallel=False)
            time.sleep(_STATCAST_SLEEP_BETWEEN_CHUNKS)
        except Exception as exc:
            logger.error("Statcast fetch failed (%s to %s): %s", start_date, end_date, exc)
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        return self._normalize_statcast(df)

    def _normalize_statcast(self, df: pd.DataFrame) -> pd.DataFrame:
        col_map = {
            "pitcher": "pitcher_id",
            "batter": "batter_id",
            "player_name": "pitcher_name",    # pybaseball puts pitcher name here
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Derive season from game_date; normalize to plain YYYY-MM-DD string so it
        # fits the VARCHAR(10) column (pybaseball returns a pandas Timestamp which
        # psycopg2 would serialize as '2019-04-10 00:00:00' — 19 chars, too long).
        if "game_date" in df.columns:
            dt = pd.to_datetime(df["game_date"])
            df["season"] = dt.dt.year
            df["game_date"] = dt.dt.strftime("%Y-%m-%d")

        # Ensure batter_name column exists (not always present in pybaseball output)
        if "batter_name" not in df.columns:
            df["batter_name"] = None

        # Keep only the columns we store in the DB
        keep = [
            "game_pk", "game_date", "season",
            "pitcher_id", "batter_id", "pitcher_name", "batter_name",
            "inning", "inning_topbot", "balls", "strikes", "outs_when_up",
            "pitch_type", "release_speed", "release_spin_rate", "release_extension",
            "pfx_x", "pfx_z", "plate_x", "plate_z", "sz_top", "sz_bot",
            "description", "events", "type",
            "launch_speed", "launch_angle", "hit_distance_sc",
            "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
            "stand", "p_throws",
        ]
        existing = [c for c in keep if c in df.columns]
        return df[existing].copy()

    # --- Not supported by pybaseball (use MLBStatsAdapter instead) ----------

    def get_schedule(self, date: str) -> list:
        raise NotImplementedError("Use MLBStatsAdapter for schedules")

    def get_game_boxscore(self, game_pk: int) -> dict:
        raise NotImplementedError("Use MLBStatsAdapter for boxscores")

    # --- Park factors --------------------------------------------------------

    def get_park_factors(self, season: int) -> pd.DataFrame:
        logger.info("Fetching park factors for %d", season)
        df = None
        # Function name changed across pybaseball versions — try both
        for fn_name in ("park_factors", "team_park_factors"):
            fn = getattr(pybaseball, fn_name, None)
            if fn is None:
                continue
            try:
                df = fn(pos="np", season=season, league="ALL")
                break
            except TypeError:
                try:
                    df = fn(season)
                    break
                except Exception as exc:
                    logger.warning("pybaseball.%s(%d) failed: %s", fn_name, season, exc)
            except Exception as exc:
                logger.warning("pybaseball.%s failed for %d: %s", fn_name, season, exc)
        if df is None or (hasattr(df, "empty") and df.empty):
            logger.error("Park factors fetch failed for %d: no working function found", season)
            return pd.DataFrame()

        col_map = {
            "Team": "team_abbr",
            "Park Factor": "pf_runs",
            "1B": "pf_1b",
            "2B": "pf_2b",
            "3B": "pf_3b",
            "HR": "pf_hr",
            "SO": "pf_so",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df["season"] = season
        # pybaseball park factors don't include venue_id; we map from team_abbr in ingest.py
        df["venue_id"] = None
        df["venue_name"] = None
        df["team_id"] = None
        if "pf_hits" not in df.columns:
            df["pf_hits"] = df.get("pf_runs")  # rough proxy if not available
        return df
