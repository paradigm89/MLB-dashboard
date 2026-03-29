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
from src.db.queries import (
    get_batting_stats,
    get_pitching_stats,
    get_statcast_matchup,
    get_all_reliever_ids,
    get_statcast_batter_vs_relievers,
    get_team_relievers,
)

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "models.yaml"

# Batting order position weights (leadoff batters see more PAs)
_BATTING_ORDER_WEIGHTS = {1: 1.15, 2: 1.12, 3: 1.10, 4: 1.08, 5: 1.05,
                           6: 1.00, 7: 0.97, 8: 0.94, 9: 0.91}

# League-average xwOBA vs. relievers is slightly lower than vs. starters
# (~.310 vs. ~.320) because relievers tend to throw harder and shorter outings.
_RELIEF_PRIOR_XWOBA = 0.310

# Module-level cache: reliever player ID sets, keyed by min_season.
# Populated once per process (re-queried only when a new season key is needed).
_reliever_id_cache: dict[int, list[int]] = {}


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


def build_lineup_matchup_features(
    lineup_player_ids: list[int],
    pitcher_id: int,
    season: int,
    before_date: Optional[str] = None,
    opposing_team_id: Optional[int] = None,
) -> dict:
    """
    Aggregate individual batter matchup scores into a lineup-level xwOBA score,
    blending the starter component and bullpen component by expected innings.

    Blend logic
    -----------
    Modern MLB starters average ~5.5 IP.  Each batter's final xwOBA is:

        xwoba = sp_weight * xwoba_vs_sp  +  relief_weight * xwoba_vs_relief

    where sp_weight = min(expected_sp_ip / 9, 0.85) — capped so the bullpen
    always contributes at least 15% of the expected value.

    The relief component is itself a blend of the batter's historical xwOBA
    vs. RH and LH relievers, weighted by the opposing bullpen's handedness
    split (defaults to league average 70% RHP / 30% LHP if unknown).

    Returns
    -------
    dict with lineup_xwoba_score, sp_weight, relief_weight, and per-batter scores
    (each score now includes xwoba_vs_sp and xwoba_vs_relief breakdowns).
    """
    if not lineup_player_ids:
        return {"lineup_xwoba_score": None, "batter_matchup_scores": []}

    if before_date is None:
        from datetime import date as _date
        before_date = str(_date.today())

    # --- Starter/bullpen blend weights ---
    sp_rolling = get_pitcher_rolling_stats(pitcher_id, before_date)
    expected_sp_ip = sp_rolling.get("ip_rolling5") or 5.5
    sp_weight = float(np.clip(expected_sp_ip / 9.0, 0.50, 0.85))
    relief_weight = 1.0 - sp_weight

    # --- Opposing bullpen handedness split ---
    if opposing_team_id:
        bullpen_feats = get_team_bullpen_features(opposing_team_id, season, before_date)
        bullpen_pct_lhp = bullpen_feats.get("bullpen_pct_lhp", 0.30)
    else:
        bullpen_pct_lhp = 0.30  # league average
    bullpen_pct_rhp = 1.0 - bullpen_pct_lhp

    scores = []
    total_weight = 0.0

    for pos, batter_id in enumerate(lineup_player_ids[:9], start=1):
        # Starter matchup (existing logic)
        features = get_batter_matchup_features(batter_id, pitcher_id, batting_order_pos=pos)
        xwoba_vs_sp = _weighted_blend(features) or features.get("batter_xwoba_season")

        # Bullpen matchup (new)
        xwoba_vs_rhp_rel = get_batter_vs_relief_xwoba(batter_id, "R", season)
        xwoba_vs_lhp_rel = get_batter_vs_relief_xwoba(batter_id, "L", season)
        xwoba_vs_relief = (
            bullpen_pct_rhp * xwoba_vs_rhp_rel
            + bullpen_pct_lhp * xwoba_vs_lhp_rel
        )

        # Final blend
        if xwoba_vs_sp is not None:
            xwoba = sp_weight * xwoba_vs_sp + relief_weight * xwoba_vs_relief
        else:
            xwoba = xwoba_vs_relief

        weight = _BATTING_ORDER_WEIGHTS.get(pos, 1.0)
        scores.append({
            "pos": pos,
            "batter_id": batter_id,
            "xwoba": xwoba,
            "xwoba_vs_sp": xwoba_vs_sp,
            "xwoba_vs_rhp_relief": xwoba_vs_rhp_rel,
            "xwoba_vs_lhp_relief": xwoba_vs_lhp_rel,
            "weight": weight,
        })
        total_weight += weight

    if not scores:
        return {"lineup_xwoba_score": None, "batter_matchup_scores": []}

    lineup_xwoba = sum(s["xwoba"] * s["weight"] for s in scores) / total_weight
    return {
        "lineup_xwoba_score": lineup_xwoba,
        "sp_weight": round(sp_weight, 3),
        "relief_weight": round(relief_weight, 3),
        "expected_sp_ip": round(expected_sp_ip, 1),
        "batter_matchup_scores": scores,
    }


def get_batter_vs_relief_xwoba(
    batter_id: int,
    pitcher_hand: str,
    season: int,
    k: int = 150,
) -> float:
    """
    Bayesian-blended xwOBA for a batter against RH or LH relievers.

    Uses the full historical Statcast record for this batter against any pitcher
    identified as a pure reliever (gs=0 in FanGraphs) over the last three seasons.
    Blends toward the league-average reliever xwOBA (_RELIEF_PRIOR_XWOBA = .310)
    when the sample is small — typical for debut or rarely-used platoon players.

    k=150 means a batter needs ~150 tracked PA vs. relievers before their
    observed rate is weighted 50/50 with the league prior.
    """
    min_season = max(season - 3, 2019)

    # Populate the module-level cache once per min_season value
    if min_season not in _reliever_id_cache:
        with get_db() as session:
            _reliever_id_cache[min_season] = get_all_reliever_ids(session, min_season)
    reliever_ids = _reliever_id_cache[min_season]

    if not reliever_ids:
        return _RELIEF_PRIOR_XWOBA

    with get_db() as session:
        df = get_statcast_batter_vs_relievers(
            session, batter_id, pitcher_hand, reliever_ids, min_season
        )

    n = len(df)
    if n == 0:
        return _RELIEF_PRIOR_XWOBA

    observed = float(df["estimated_woba_using_speedangle"].mean())
    return (n * observed + k * _RELIEF_PRIOR_XWOBA) / (n + k)


def get_team_bullpen_features(
    team_id: int,
    season: int,
    before_date: str,
) -> dict:
    """
    Compute team bullpen quality features for use as model inputs.

    bullpen_xfip_season  -- IP-weighted xFIP of the team's pure relievers
    bullpen_k_pct_season -- IP-weighted K% of the team's pure relievers
    bullpen_pct_lhp      -- fraction of bullpen IP from LH arms (handedness split)
    bullpen_ip_3d        -- estimated innings used by the bullpen in the last 3 days
                           (proxy for fatigue/availability; derived from pitch counts)

    Relievers are identified as pitchers with gs=0 and ≥5 appearances in the
    FanGraphs season stats, which filters out emergency starters while including
    all true bullpen arms.
    """
    null_result = {
        "bullpen_xfip_season": None,
        "bullpen_k_pct_season": None,
        "bullpen_pct_lhp": 0.30,
        "bullpen_ip_3d": None,
    }

    with get_db() as session:
        rel_df = get_team_relievers(session, team_id, season)

    if rel_df.empty:
        return null_result

    total_ip = rel_df["ip"].fillna(0).sum()
    if total_ip == 0:
        return null_result

    # IP-weighted xFIP
    valid = rel_df[rel_df["xfip"].notna() & (rel_df["ip"].fillna(0) > 0)]
    bullpen_xfip = (
        float((valid["xfip"] * valid["ip"]).sum() / valid["ip"].sum())
        if not valid.empty else None
    )

    # IP-weighted K%
    valid_k = rel_df[rel_df["k_pct"].notna() & (rel_df["ip"].fillna(0) > 0)]
    bullpen_kpct = (
        float((valid_k["k_pct"] * valid_k["ip"]).sum() / valid_k["ip"].sum())
        if not valid_k.empty else None
    )

    # Handedness split
    lhp_ip = rel_df[rel_df["throws"] == "L"]["ip"].fillna(0).sum()
    bullpen_pct_lhp = float(lhp_ip / total_ip) if total_ip > 0 else 0.30

    # Bullpen IP used in last 3 days (fatigue / availability signal)
    reliever_ids = rel_df["player_id"].dropna().astype(int).tolist()
    cutoff = (pd.Timestamp(before_date) - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    bullpen_ip_3d = 0.0
    if reliever_ids:
        with get_db() as session:
            from sqlalchemy import text
            row = session.execute(
                text("""
                    SELECT COUNT(*) AS n_pitches
                    FROM statcast_pitches
                    WHERE pitcher_id = ANY(:pids)
                      AND game_date >= :cutoff
                      AND game_date < :before_date
                """),
                {"pids": reliever_ids, "cutoff": cutoff, "before_date": before_date},
            ).fetchone()
        # ~15 pitches per inning is the standard approximation
        bullpen_ip_3d = float(row[0]) / 15.0 if row and row[0] else 0.0

    return {
        "bullpen_xfip_season": bullpen_xfip,
        "bullpen_k_pct_season": bullpen_kpct,
        "bullpen_pct_lhp": bullpen_pct_lhp,
        "bullpen_ip_3d": bullpen_ip_3d,
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
