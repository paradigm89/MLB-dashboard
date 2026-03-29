"""
Feature engineering pipeline.

Reads from the PostgreSQL DB and builds feature matrices for each ML model.
Feature definitions are driven by config/features.yaml — add/remove features
there without touching this file.

Outputs one row per game side (home and away treated as separate observations
for the run expectancy model; combined for the win probability derivation).
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from src.db.connection import get_db
from src.db.queries import (
    get_batting_stats,
    get_games_date_range,
    get_games_for_season,
    get_latest_elo,
    get_park_factor,
    get_pitching_stats,
    get_recent_games_for_team,
)

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "features.yaml"

# Bayesian shrinkage constants (from config/models.yaml)
_K_RATE = 200      # for rate stats (K%, BB%, etc.)
_K_CONTINUOUS = 100  # for continuous stats (xwOBA, etc.)
_K_ERA = 8         # for ERA/xFIP vs. opponent (in GS)


def load_feature_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_game_feature_matrix(seasons: list[int]) -> pd.DataFrame:
    """
    Build the full feature matrix for all games in the given seasons.
    Used for model training and evaluation.

    Returns a DataFrame with one row per game, columns for all features,
    and target columns: runs_scored_home, runs_scored_away.
    """
    cfg = load_feature_config()
    all_rows = []

    for season in seasons:
        logger.info("Building feature matrix for season %d", season)
        with get_db() as session:
            games = get_games_for_season(session, season)
            batting = get_batting_stats(session, season)
            pitching = get_pitching_stats(session, season)

        if games.empty:
            logger.warning("No games found for season %d", season)
            continue

        for _, game in games.iterrows():
            row = _build_game_row(game, batting, pitching, season)
            if row is not None:
                all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    logger.info("Built feature matrix: %d rows, %d columns", len(df), len(df.columns))
    return df


def build_prediction_features(game_pk: int, home_team_id: int, away_team_id: int,
                                home_sp_id: Optional[int], away_sp_id: Optional[int],
                                game_date: str, venue_id: Optional[int]) -> Optional[dict]:
    """
    Build features for a single upcoming game (for live prediction).
    Returns a dict of feature values (or None if insufficient data).
    """
    season = int(game_date[:4])
    with get_db() as session:
        batting = get_batting_stats(session, season)
        pitching = get_pitching_stats(session, season)

    # Build a synthetic game row for feature extraction
    game = pd.Series({
        "game_pk": game_pk,
        "game_date": game_date,
        "season": season,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "venue_id": venue_id,
        "home_sp_id": home_sp_id,
        "away_sp_id": away_sp_id,
        "home_score": None,  # unknown for upcoming game
        "away_score": None,
        # Weather/umpire unknown pre-game; will fill as None
        "temperature": None,
        "wind_speed": None,
        "wind_direction": None,
        "roof_status": None,
        "home_plate_umpire_id": None,
    })

    return _build_game_row(game, batting, pitching, season)


# ---------------------------------------------------------------------------
# Core feature construction
# ---------------------------------------------------------------------------

def _build_game_row(game: pd.Series, batting: pd.DataFrame,
                     pitching: pd.DataFrame, season: int) -> Optional[dict]:
    """Build a single feature row for a game."""
    game_date = str(game["game_date"])
    home_id = int(game["home_team_id"])
    away_id = int(game["away_team_id"])
    home_sp_id = game.get("home_sp_id")
    away_sp_id = game.get("away_sp_id")

    row = {}

    # Context features
    row["game_pk"] = game["game_pk"]
    row["game_date"] = game_date
    row["season"] = season

    # Park factors
    pf = _get_park_features(game.get("venue_id"), season)
    row.update(pf)

    # Elo ratings
    row["elo_home"] = _get_elo(home_id, game_date)
    row["elo_away"] = _get_elo(away_id, game_date)

    # Weather / context (null-safe; XGBoost handles missing values)
    row["temperature"] = _to_float(game.get("temperature"))
    row["wind_speed"] = _to_float(game.get("wind_speed"))
    row["wind_direction_in"] = _wind_direction(game.get("wind_direction"), "in")
    row["wind_direction_out"] = _wind_direction(game.get("wind_direction"), "out")
    row["roof_closed"] = _roof_closed(game.get("roof_status"))
    row["umpire_called_strike_delta"] = _get_umpire_delta(game.get("home_plate_umpire_id"))

    # Season game number
    row["season_game_number"] = _get_season_game_number(home_id, game_date)
    row["is_doubleheader"] = 0  # TODO: detect from schedule

    # Team rolling stats (home and away)
    home_rolling = _build_team_rolling_features(home_id, game_date)
    away_rolling = _build_team_rolling_features(away_id, game_date)
    for k, v in home_rolling.items():
        row[f"{k}_home"] = v
    for k, v in away_rolling.items():
        row[f"{k}_away"] = v

    # Starting pitcher features
    home_sp = _build_pitcher_features(home_sp_id, away_id, game_date, pitching, prefix="home_sp")
    away_sp = _build_pitcher_features(away_sp_id, home_id, game_date, pitching, prefix="away_sp")
    row.update(home_sp)
    row.update(away_sp)

    # Target variables (None for upcoming games)
    row["runs_scored_home"] = _to_float(game.get("home_score"))
    row["runs_scored_away"] = _to_float(game.get("away_score"))
    row["home_win"] = game.get("home_win")

    return row


def _build_team_rolling_features(team_id: int, before_date: str) -> dict:
    """Compute rolling stats for a team from their last N games."""
    features = {}
    with get_db() as session:
        games_30 = get_recent_games_for_team(session, team_id, before_date, n=30)
        games_7 = games_30.head(7)

    if games_30.empty:
        return {k: None for k in [
            "runs_scored_7g", "runs_scored_30g", "runs_allowed_7g", "runs_allowed_30g",
            "ops_30g", "era_30g", "whip_30g", "fip_30g", "wrc_plus_30g",
            "pythagorean_win_pct", "run_diff_slope_10g",
            "bullpen_era_7d", "bullpen_ip_3d", "team_drs_30g", "lineup_xwoba_score",
        ]}

    def runs_for(df, col_home, col_away):
        return pd.concat([
            df.loc[df["home_team_id"] == team_id, col_home],
            df.loc[df["away_team_id"] == team_id, col_away],
        ]).mean()

    features["runs_scored_7g"] = runs_for(games_7, "home_score", "away_score")
    features["runs_scored_30g"] = runs_for(games_30, "home_score", "away_score")
    features["runs_allowed_7g"] = runs_for(games_7, "away_score", "home_score")
    features["runs_allowed_30g"] = runs_for(games_30, "away_score", "home_score")

    # Pythagorean win% (Bill James formula, exponent 1.83)
    rs = features["runs_scored_30g"] or 0
    ra = features["runs_allowed_30g"] or 1
    features["pythagorean_win_pct"] = rs ** 1.83 / (rs ** 1.83 + ra ** 1.83) if rs > 0 else 0.5

    # Run differential slope (momentum) — linear regression over last 10 games
    games_10 = games_30.head(10)
    if len(games_10) >= 3:
        run_diffs = _compute_run_diffs(games_10, team_id)
        if len(run_diffs) >= 2:
            x = np.arange(len(run_diffs))
            features["run_diff_slope_10g"] = float(np.polyfit(x, run_diffs, 1)[0])
        else:
            features["run_diff_slope_10g"] = 0.0
    else:
        features["run_diff_slope_10g"] = 0.0

    # Aggregate batting/pitching rate stats require joining per-game box scores
    # which aren't stored at that granularity; leave as None for now.
    for k in ["ops_30g", "era_30g", "whip_30g", "fip_30g", "wrc_plus_30g",
              "team_drs_30g", "lineup_xwoba_score"]:
        features[k] = None

    # Bullpen features — computed from FanGraphs pitching stats (gs=0 relievers)
    # and recent Statcast pitch counts for fatigue estimation.
    season = int(before_date[:4])
    from src.pipeline.matchups import get_team_bullpen_features
    bp = get_team_bullpen_features(team_id, season, before_date)
    features["bullpen_xfip_season"] = bp.get("bullpen_xfip_season")
    features["bullpen_k_pct_season"] = bp.get("bullpen_k_pct_season")
    features["bullpen_pct_lhp"] = bp.get("bullpen_pct_lhp")
    features["bullpen_era_7d"] = bp.get("bullpen_xfip_season")  # xFIP is a better ERA proxy
    features["bullpen_ip_3d"] = bp.get("bullpen_ip_3d")

    return features


def _compute_run_diffs(games: pd.DataFrame, team_id: int) -> list[float]:
    diffs = []
    for _, g in games.iterrows():
        if g["home_team_id"] == team_id:
            diffs.append((g["home_score"] or 0) - (g["away_score"] or 0))
        else:
            diffs.append((g["away_score"] or 0) - (g["home_score"] or 0))
    return diffs


def _build_pitcher_features(sp_id: Optional[int], opponent_team_id: int,
                              game_date: str, pitching: pd.DataFrame,
                              prefix: str) -> dict:
    """Build pitcher feature columns for a starting pitcher."""
    null_row = {f"{prefix}_{k}": None for k in [
        "xfip_season", "fip_season", "siera_season", "k_pct_season", "bb_pct_season",
        "k_minus_bb_season", "hr_per_9_season",
        "xfip_rolling5", "k_pct_rolling5", "ip_rolling5",
        "fastball_velo_rolling3", "velo_delta_rolling3", "swstr_pct_season",
        "pct_ff", "pct_sl", "pct_ch", "pct_cu", "throws_l",
        "vs_opp_xfip", "vs_opp_k_pct", "vs_opp_hr_per_9", "vs_opp_sample_gs",
        "opp_k_pct_vs_hand", "opp_ops_vs_hand",
    ]}

    if sp_id is None or pitching.empty:
        return null_row

    sp_stats = pitching[pitching["player_id"] == sp_id]
    if sp_stats.empty:
        return null_row

    sp = sp_stats.iloc[0]
    features = {}
    p = prefix

    features[f"{p}_xfip_season"] = _to_float(sp.get("xfip"))
    features[f"{p}_fip_season"] = _to_float(sp.get("fip"))
    features[f"{p}_siera_season"] = _to_float(sp.get("siera"))
    features[f"{p}_k_pct_season"] = _to_float(sp.get("k_pct"))
    features[f"{p}_bb_pct_season"] = _to_float(sp.get("bb_pct"))
    features[f"{p}_k_minus_bb_season"] = _to_float(sp.get("k_minus_bb_pct"))
    features[f"{p}_hr_per_9_season"] = _to_float(sp.get("hr_per_9"))
    features[f"{p}_swstr_pct_season"] = _to_float(sp.get("swstr_pct"))
    features[f"{p}_pct_ff"] = _to_float(sp.get("pct_ff"))
    features[f"{p}_pct_sl"] = _to_float(sp.get("pct_sl"))
    features[f"{p}_pct_ch"] = _to_float(sp.get("pct_ch"))
    features[f"{p}_pct_cu"] = _to_float(sp.get("pct_cu"))
    features[f"{p}_throws_l"] = 1 if str(sp.get("throws", "R")).upper() == "L" else 0

    # Rolling last 5 starts — requires start-by-start game log (not available from
    # season aggregate stats; computed from statcast data in matchups.py and cached)
    for k in ["xfip_rolling5", "k_pct_rolling5", "ip_rolling5",
              "fastball_velo_rolling3", "velo_delta_rolling3"]:
        features[f"{p}_{k}"] = None  # populated by matchups.py

    # Opponent-specific features (Bayesian-blended; populated by matchups.py)
    for k in ["vs_opp_xfip", "vs_opp_k_pct", "vs_opp_hr_per_9", "vs_opp_sample_gs",
              "opp_k_pct_vs_hand", "opp_ops_vs_hand"]:
        features[f"{p}_{k}"] = None

    return features


# ---------------------------------------------------------------------------
# Helper lookups
# ---------------------------------------------------------------------------

def _get_park_features(venue_id: Optional[int], season: int) -> dict:
    default = {"park_factor_runs": 100.0, "park_factor_hr": 100.0, "park_factor_hits": 100.0}
    if not venue_id:
        return default
    with get_db() as session:
        pf = get_park_factor(session, venue_id, season)
    if pf is None:
        return default
    return {
        "park_factor_runs": pf.pf_runs or 100.0,
        "park_factor_hr": pf.pf_hr or 100.0,
        "park_factor_hits": pf.pf_hits or 100.0,
    }


def _get_elo(team_id: int, game_date: str) -> float:
    with get_db() as session:
        return get_latest_elo(session, team_id, game_date) or 1500.0


def _get_season_game_number(team_id: int, game_date: str) -> int:
    """Count how many games this team has played so far this season."""
    season = int(game_date[:4])
    season_start = f"{season}-03-01"
    with get_db() as session:
        games = get_games_date_range(session, season_start, game_date)
    if games.empty:
        return 0
    mask = (games["home_team_id"] == team_id) | (games["away_team_id"] == team_id)
    return int(mask.sum())


def _get_umpire_delta(umpire_id: Optional[int]) -> float:
    """Return the umpire's called strike rate delta vs. league avg. 0.0 if unknown."""
    if not umpire_id:
        return 0.0
    from sqlalchemy import text
    with get_db() as session:
        result = session.execute(
            text("SELECT called_strike_rate_delta FROM umpire_stats WHERE umpire_id = :uid ORDER BY season DESC LIMIT 1"),
            {"uid": umpire_id},
        ).scalar()
    return float(result) if result is not None else 0.0


def _wind_direction(wind_str: Optional[str], direction: str) -> int:
    if not wind_str:
        return 0
    wind_lower = str(wind_str).lower()
    if direction == "in" and ("in" in wind_lower):
        return 1
    if direction == "out" and ("out" in wind_lower):
        return 1
    return 0


def _roof_closed(roof_status: Optional[str]) -> int:
    if not roof_status:
        return 0
    return 1 if "closed" in str(roof_status).lower() or "dome" in str(roof_status).lower() else 0


def _to_float(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
