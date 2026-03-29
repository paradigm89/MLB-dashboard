"""
Batter vs. pitcher matchup aggregation with Bayesian shrinkage.

Reads from statcast_pitches to compute per-matchup xwOBA and pitch-type
specific stats. Applies Bayesian shrinkage so small-sample matchups
(e.g., 3 PA between a batter and pitcher) are blended toward the
appropriate population prior rather than taken at face value.

Shrinkage formula: blended = (n * observed + k * prior) / (n + k)
where k is chosen per stat type (from config/models.yaml).

Also computes:
- Pitcher rolling stats (last 5 starts)
- Pitcher vs. specific opponent team stats
- Lineup-level xwOBA score (weighted by batting order position)
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from src.db.connection import get_db
from src.db.queries import get_batting_stats, get_pitching_stats, get_statcast_matchup

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "models.yaml"

# Batting order position weights (leadoff batters see more PAs)
# Approximate relative PA weight by lineup slot
_BATTING_ORDER_WEIGHTS = {1: 1.15, 2: 1.12, 3: 1.10, 4: 1.08, 5: 1.05,
                           6: 1.00, 7: 0.97, 8: 0.94, 9: 0.91}


def load_shrinkage_config() -> dict:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("bayesian_shrinkage", {})


# ---------------------------------------------------------------------------
# Batter vs. pitcher matchup features
# ---------------------------------------------------------------------------

def get_batter_matchup_features(batter_id: int, pitcher_id: int,
                                  batting_order_pos: int = 5) -> dict:
    """
    Return Bayesian-blended matchup features for a single batter vs. pitcher.
    Used by both the batter_matchup model and the Monte Carlo simulator.
    """
    cfg = load_shrinkage_config()
    k_continuous = cfg.get("k_continuous", 100)
    k_rate = cfg.get("k_rate_stats", 200)

    # Pull all historical PAs between this pair
    with get_db() as session:
        matchup_df = get_statcast_matchup(session, batter_id, pitcher_id)

    n_pa = len(matchup_df) if not matchup_df.empty else 0

    # Population priors: season stats for this batter and pitcher
    season = _current_season()
    with get_db() as session:
        batter_season = _get_batter_season_stats(session, batter_id, season)
        pitcher_season = _get_pitcher_season_stats(session, pitcher_id, season)

    features = {
        "batter_id": batter_id,
        "pitcher_id": pitcher_id,
        "n_pa_matchup": n_pa,
        "batting_order_position": batting_order_pos,
    }

    # Batter season-level features (no shrinkage needed — full season sample)
    features["batter_xwoba_season"] = batter_season.get("xwoba")
    features["batter_wrc_plus_vs_hand"] = _batter_wrc_vs_hand(batter_season, pitcher_season)
    features["batter_chase_rate"] = batter_season.get("o_swing_pct")  # FanGraphs O-Swing%
    features["batter_z_contact"] = batter_season.get("z_contact_pct")
    features["batter_barrel_pct"] = batter_season.get("barrel_pct")
    features["batter_hard_hit_pct"] = batter_season.get("hard_hit_pct")
    features["batter_hand_l"] = 1 if str(batter_season.get("bats", "R")).upper() == "L" else 0
    features["pitcher_hand_l"] = 1 if str(pitcher_season.get("throws", "R")).upper() == "L" else 0

    # Pitcher season-level features
    features["pitcher_zone_pct"] = pitcher_season.get("zone_pct")
    features["pitcher_edge_pct"] = pitcher_season.get("edge_pct")

    # Pitch-type specific features: blended from matchup + pitcher's overall rates
    for pitch_type, feature_key in [("FF", "ff"), ("SL", "sl"), ("CH", "ch"), ("CU", "cu")]:
        # Batter xwOBA vs. this pitch type
        batter_vs_pt = _batter_xwoba_vs_pitch_type(
            matchup_df, batter_id, pitch_type, n_pa, k_continuous,
            prior=batter_season.get("xwoba", 0.320),
        )
        features[f"batter_xwoba_vs_{feature_key}"] = batter_vs_pt

        # Pitcher whiff rate on this pitch type vs. batter's handedness
        batter_hand = "L" if features["batter_hand_l"] else "R"
        features[f"pitcher_whiff_{feature_key}"] = _pitcher_whiff_rate(
            pitcher_season, pitch_type, batter_hand
        )

    return features


def build_lineup_matchup_features(lineup_player_ids: list[int], pitcher_id: int,
                                    season: int) -> dict:
    """
    Aggregate individual batter matchup scores into a lineup-level xwOBA score.
    Weights by batting order position (leadoff sees more PAs).

    Returns:
        dict with lineup_xwoba_score and individual batter scores
    """
    if not lineup_player_ids:
        return {"lineup_xwoba_score": None, "batter_matchup_scores": []}

    scores = []
    total_weight = 0.0

    for pos, batter_id in enumerate(lineup_player_ids[:9], start=1):
        features = get_batter_matchup_features(batter_id, pitcher_id, batting_order_pos=pos)
        xwoba = features.get("batter_xwoba_season")  # fallback to season if matchup is sparse

        # Use blended matchup xwOBA if we have reasonable data
        blended = _weighted_blend(features)
        if blended is not None:
            xwoba = blended

        weight = _BATTING_ORDER_WEIGHTS.get(pos, 1.0)
        if xwoba is not None:
            scores.append({"pos": pos, "batter_id": batter_id, "xwoba": xwoba, "weight": weight})
            total_weight += weight

    if not scores:
        return {"lineup_xwoba_score": None, "batter_matchup_scores": []}

    lineup_xwoba = sum(s["xwoba"] * s["weight"] for s in scores) / total_weight
    return {
        "lineup_xwoba_score": lineup_xwoba,
        "batter_matchup_scores": scores,
    }


def get_pitcher_rolling_stats(pitcher_id: int, before_date: str, n_starts: int = 5) -> dict:
    """
    Compute rolling stats over last n_starts for a pitcher from Statcast data.
    Returns: xfip_rolling, k_pct_rolling, ip_rolling, avg_velo, velo_delta
    """
    season = int(before_date[:4])
    with get_db() as session:
        from sqlalchemy import text
        rows = session.execute(
            text("""
                SELECT game_date, game_pk,
                       AVG(release_speed) as avg_velo,
                       COUNT(*) as n_pitches,
                       SUM(CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) as k,
                       SUM(CASE WHEN inning_topbot IS NOT NULL THEN 1 ELSE 0 END) as batters_faced
                FROM statcast_pitches
                WHERE pitcher_id = :pid
                  AND game_date < :before_date
                  AND season = :season
                  AND pitch_type IS NOT NULL
                GROUP BY game_date, game_pk
                ORDER BY game_date DESC
                LIMIT :n_starts
            """),
            {"pid": pitcher_id, "before_date": before_date,
             "season": season, "n_starts": n_starts},
        ).fetchall()

    if not rows:
        return {"xfip_rolling5": None, "k_pct_rolling5": None, "ip_rolling5": None,
                "fastball_velo_rolling3": None, "velo_delta_rolling3": None}

    df = pd.DataFrame(rows, columns=["game_date", "game_pk", "avg_velo", "n_pitches", "k", "batters_faced"])

    # Rough IP estimate: n_pitches / 15 (15 pitches per inning approximation)
    df["est_ip"] = df["n_pitches"] / 15.0
    df["k_pct"] = df["k"] / df["batters_faced"].replace(0, np.nan)

    velo_last3 = df.head(3)["avg_velo"].mean() if len(df) >= 3 else df["avg_velo"].mean()
    velo_season = df["avg_velo"].mean()

    return {
        "xfip_rolling5": None,  # xFIP requires HR and BB data; simplified here
        "k_pct_rolling5": float(df["k_pct"].mean()) if not df["k_pct"].isna().all() else None,
        "ip_rolling5": float(df["est_ip"].mean()),
        "fastball_velo_rolling3": float(velo_last3) if pd.notna(velo_last3) else None,
        "velo_delta_rolling3": float(velo_last3 - velo_season) if pd.notna(velo_last3) else None,
    }


def get_pitcher_vs_opponent(pitcher_id: int, opponent_team_id: int,
                              before_date: str) -> dict:
    """
    Compute Bayesian-blended pitcher stats vs. a specific opponent team.
    Returns: vs_opp_xfip, vs_opp_k_pct, vs_opp_hr_per_9, vs_opp_sample_gs
    """
    cfg = load_shrinkage_config()
    k_era = cfg.get("k_era_stats", 8)

    season = int(before_date[:4])
    with get_db() as session:
        from sqlalchemy import text
        rows = session.execute(
            text("""
                SELECT sp.xfip, sp.k_pct, sp.hr_per_9
                FROM pitching_stats sp
                WHERE sp.player_id = :pid AND sp.season = :season
                LIMIT 1
            """),
            {"pid": pitcher_id, "season": season},
        ).fetchone()

    if rows is None:
        return {"vs_opp_xfip": None, "vs_opp_k_pct": None,
                "vs_opp_hr_per_9": None, "vs_opp_sample_gs": 0}

    season_xfip = rows[0]
    season_k_pct = rows[1]
    season_hr9 = rows[2]

    # Count GS vs. this opponent (from games table + pitching appearance)
    with get_db() as session:
        from sqlalchemy import text as sql_text
        opp_rows = session.execute(
            sql_text("""
                SELECT COUNT(DISTINCT g.game_pk) as gs
                FROM games g
                WHERE (g.home_sp_id = :pid OR g.away_sp_id = :pid)
                  AND (g.home_team_id = :opp OR g.away_team_id = :opp)
                  AND g.game_date < :before_date
                  AND g.season >= :min_season
                  AND g.status = 'Final'
            """),
            {"pid": pitcher_id, "opp": opponent_team_id,
             "before_date": before_date, "min_season": season - 2},
        ).fetchone()

    n_gs = opp_rows[0] if opp_rows else 0

    # Bayesian blend: when n_gs < k_era, shrink heavily toward season stats
    w = n_gs / (n_gs + k_era)

    # For opponent-specific stats we only have aggregate data here;
    # in practice you'd query game-by-game Statcast to compute opp-specific xFIP.
    # Using season stats as both prior and observed for now (conservative).
    return {
        "vs_opp_xfip": season_xfip,   # TODO: replace with actual vs-opp stats when available
        "vs_opp_k_pct": season_k_pct,
        "vs_opp_hr_per_9": season_hr9,
        "vs_opp_sample_gs": n_gs,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _batter_xwoba_vs_pitch_type(matchup_df: pd.DataFrame, batter_id: int,
                                  pitch_type: str, n_pa: int,
                                  k: float, prior: float) -> Optional[float]:
    """
    Compute Bayesian-blended xwOBA for a batter against a specific pitch type.
    """
    if matchup_df.empty:
        return prior

    pt_df = matchup_df[matchup_df["pitch_type"] == pitch_type]
    if pt_df.empty:
        return prior

    in_play = pt_df[pt_df["estimated_woba_using_speedangle"].notna()]
    n_contact = len(in_play)

    if n_contact == 0:
        return prior

    observed = float(in_play["estimated_woba_using_speedangle"].mean())
    blended = (n_contact * observed + k * prior) / (n_contact + k)
    return blended


def _pitcher_whiff_rate(pitcher_stats: dict, pitch_type: str, batter_hand: str) -> Optional[float]:
    """
    Return pitcher's whiff rate for a specific pitch type vs. batter handedness.
    Simplified: uses overall swstr_pct when pitch-type-specific data unavailable.
    """
    return pitcher_stats.get("swstr_pct")


def _batter_wrc_vs_hand(batter_stats: dict, pitcher_stats: dict) -> Optional[float]:
    """Return batter's wRC+ split for the pitcher's handedness."""
    throws = str(pitcher_stats.get("throws", "R")).upper()
    if throws == "L":
        return batter_stats.get("wrc_plus_vs_lhp")
    return batter_stats.get("wrc_plus_vs_rhp")


def _weighted_blend(features: dict) -> Optional[float]:
    """
    Derive a single blended xwOBA estimate from pitch-type matchup features.
    Weights by typical pitcher's pitch mix usage.
    """
    weights = {"ff": 0.35, "sl": 0.25, "ch": 0.20, "cu": 0.15}
    total_weight = 0.0
    total = 0.0
    for pt, w in weights.items():
        val = features.get(f"batter_xwoba_vs_{pt}")
        if val is not None:
            total += val * w
            total_weight += w
    if total_weight < 0.1:
        return None
    return total / total_weight


def _get_batter_season_stats(session, batter_id: int, season: int) -> dict:
    from src.db.schema import BattingStats
    row = session.query(BattingStats).filter(
        BattingStats.player_id == batter_id,
        BattingStats.season == season,
    ).first()
    if row is None:
        return {}
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}


def _get_pitcher_season_stats(session, pitcher_id: int, season: int) -> dict:
    from src.db.schema import PitchingStats
    row = session.query(PitchingStats).filter(
        PitchingStats.player_id == pitcher_id,
        PitchingStats.season == season,
    ).first()
    if row is None:
        return {}
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}


def _current_season() -> int:
    from datetime import date
    return date.today().year
