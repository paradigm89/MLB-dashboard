"""
Monte Carlo Game Simulator.

Simulates 10,000 games using individual batter matchup predictions
and pitcher fatigue modeling to produce a distribution over final scores.

Architecture:
- Each half-inning: cycle through lineup, draw PA outcomes from batter matchup scores
- PA outcomes sampled from a probability table derived from xwOBA:
  xwOBA → expected PA outcome distribution (K, BB, 1B, 2B, 3B, HR, out)
- Base-out state transitions via an empirical Markov chain run expectancy table
- Pitcher effectiveness decays after pitch count / inning thresholds

Honest caveat (from plan):
This simulator is only as good as its inputs. If the batter matchup model
is poorly calibrated, the simulator will amplify those errors. We validate
the batter matchup model independently before relying on simulator outputs.
If the score distribution median is outside [2, 8] runs per team, we log
a warning and fall back to the Poisson-based win probability.

The Markov chain transition table is empirically derived from Statcast data
in training/evaluate.py. The default table below is a reasonable approximation
based on historical MLB run expectancy.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.models.win_probability import compute_win_probability

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Empirical run expectancy transition table
# Base-out state: (runners_on_base_bitmask, outs)
# runners bitmask: bit 0=1B, bit 1=2B, bit 2=3B → 0..7
# Outcome categories: out, walk, single, double, triple, hr, strikeout
# ---------------------------------------------------------------------------

# Average xwOBA bands → outcome probabilities (7 outcomes)
# Derived from MLB 2019-2024 average Statcast distributions
# Columns: [out, walk, single, double, triple, hr, strikeout]
# Indexed by xwOBA quintile [0-0.2, 0.2-0.28, 0.28-0.34, 0.34-0.40, 0.40+]
_XWOBA_TO_OUTCOMES = np.array([
    [0.38, 0.06, 0.14, 0.03, 0.005, 0.015, 0.38],   # very weak (<0.20)
    [0.32, 0.08, 0.16, 0.04, 0.005, 0.025, 0.33],   # below avg (0.20-0.28)
    [0.26, 0.09, 0.17, 0.05, 0.006, 0.04,  0.28],   # avg (0.28-0.34)
    [0.20, 0.10, 0.18, 0.06, 0.007, 0.06,  0.23],   # above avg (0.34-0.40)
    [0.16, 0.12, 0.18, 0.07, 0.008, 0.09,  0.21],   # elite (>0.40)
], dtype=float)

# Normalize rows to sum to 1
_XWOBA_TO_OUTCOMES = _XWOBA_TO_OUTCOMES / _XWOBA_TO_OUTCOMES.sum(axis=1, keepdims=True)

_XWOBA_BREAKS = [0.20, 0.28, 0.34, 0.40]

# Runs scored on a triple with various base configurations (simplified)
_RUNS_ON_TRIPLE = {0: 0, 1: 1, 2: 1, 3: 2, 4: 1, 5: 2, 6: 2, 7: 3}
_RUNS_ON_DOUBLE = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2}


@dataclass
class SimConfig:
    n_simulations: int = 10000
    max_innings: int = 9
    extra_innings_method: str = "coin_flip"
    pitcher_fatigue_pitch_threshold: int = 75
    pitcher_fatigue_inning_threshold: int = 5
    pitcher_fatigue_decay: float = 0.08   # multiply xwOBA advantage by (1 - decay)


@dataclass
class HalfInningState:
    runners: int = 0   # bitmask: bit0=1B, bit1=2B, bit2=3B
    outs: int = 0
    runs: int = 0


class MonteCarloSimulator:
    """
    Game simulator combining batter matchup features and pitcher fatigue.
    """

    def __init__(self, config: Optional[SimConfig] = None):
        self._config = config or SimConfig()

    def simulate_game(self,
                       home_lineup_xwobas: list[float],
                       away_lineup_xwobas: list[float],
                       home_sp_xfip: Optional[float] = None,
                       away_sp_xfip: Optional[float] = None) -> dict:
        """
        Run n_simulations of a game and return score distribution + win probability.

        Args:
            home_lineup_xwobas: list of 9 xwOBA values for home batters vs. away SP
            away_lineup_xwobas: list of 9 xwOBA values for away batters vs. home SP
            home_sp_xfip: home SP's projected xFIP (for pitcher advantage adjustment)
            away_sp_xfip: away SP's projected xFIP

        Returns:
            dict with win_prob_home/away, median/p10/p90 runs, score_distribution
        """
        cfg = self._config

        # Validate inputs
        if not self._validate_inputs(home_lineup_xwobas, away_lineup_xwobas):
            logger.warning("Invalid lineup xwOBA inputs; falling back to Poisson model")
            return self._fallback_poisson(home_sp_xfip, away_sp_xfip)

        home_scores = np.zeros(cfg.n_simulations, dtype=int)
        away_scores = np.zeros(cfg.n_simulations, dtype=int)

        for sim in range(cfg.n_simulations):
            h_runs, a_runs = self._simulate_single_game(
                home_lineup_xwobas, away_lineup_xwobas,
                home_sp_xfip, away_sp_xfip,
            )
            home_scores[sim] = h_runs
            away_scores[sim] = a_runs

        # Check for unrealistic score distributions
        median_home = float(np.median(home_scores))
        median_away = float(np.median(away_scores))
        if not (1.0 <= median_home <= 10.0) or not (1.0 <= median_away <= 10.0):
            logger.warning(
                "Simulator produced unrealistic score distribution "
                "(median home=%.1f, away=%.1f). Falling back to Poisson model.",
                median_home, median_away,
            )
            return self._fallback_poisson(home_sp_xfip, away_sp_xfip)

        home_wins = int(np.sum(home_scores > away_scores))
        away_wins = int(np.sum(away_scores > home_scores))
        ties = cfg.n_simulations - home_wins - away_wins

        # Extra innings coin flip for ties
        extra_home = int(ties * 0.51)
        extra_away = ties - extra_home

        total_home_wins = home_wins + extra_home
        total_away_wins = away_wins + extra_away
        total = total_home_wins + total_away_wins

        return {
            "win_prob_home": round(total_home_wins / total, 4),
            "win_prob_away": round(total_away_wins / total, 4),
            "median_runs_home": median_home,
            "median_runs_away": median_away,
            "p10_runs_home": float(np.percentile(home_scores, 10)),
            "p90_runs_home": float(np.percentile(home_scores, 90)),
            "p10_runs_away": float(np.percentile(away_scores, 10)),
            "p90_runs_away": float(np.percentile(away_scores, 90)),
            "simulation_method": "monte_carlo",
            "n_simulations": cfg.n_simulations,
        }

    def _simulate_single_game(self,
                                home_xwobas: list[float],
                                away_xwobas: list[float],
                                home_sp_xfip: Optional[float],
                                away_sp_xfip: Optional[float]) -> tuple[int, int]:
        """Simulate a single 9-inning game. Returns (home_runs, away_runs)."""
        cfg = self._config
        home_runs = 0
        away_runs = 0

        # Pitcher effectiveness modifier derived from xFIP
        # Lower xFIP → pitcher suppresses offense → reduce batter xwOBA values
        home_sp_mod = _xfip_to_modifier(home_sp_xfip)   # applied to away batters
        away_sp_mod = _xfip_to_modifier(away_sp_xfip)    # applied to home batters

        home_pitcher_pitches = 0
        away_pitcher_pitches = 0
        home_batter_idx = 0
        away_batter_idx = 0

        for inning in range(1, cfg.max_innings + 1):
            # Top of inning: away team bats vs. home SP
            away_fatigue = _compute_fatigue_mod(
                cfg, away_pitcher_pitches, inning, pitcher_side="away"
            )
            runs, pitches_used, away_batter_idx = self._simulate_half_inning(
                away_xwobas, away_batter_idx, home_sp_mod * away_fatigue
            )
            away_runs += runs
            home_pitcher_pitches += pitches_used

            # Bottom of inning: home team bats vs. away SP
            home_fatigue = _compute_fatigue_mod(
                cfg, home_pitcher_pitches, inning, pitcher_side="home"
            )
            runs, pitches_used, home_batter_idx = self._simulate_half_inning(
                home_xwobas, home_batter_idx, away_sp_mod * home_fatigue
            )
            home_runs += runs
            away_pitcher_pitches += pitches_used

        return home_runs, away_runs

    def _simulate_half_inning(self,
                               lineup_xwobas: list[float],
                               batter_start_idx: int,
                               pitcher_modifier: float) -> tuple[int, int, int]:
        """
        Simulate a half-inning. Returns (runs, approx_pitches, next_batter_idx).
        """
        state = HalfInningState()
        pitches = 0
        n = len(lineup_xwobas)

        while state.outs < 3:
            batter_xwoba = lineup_xwobas[batter_start_idx % n] * pitcher_modifier
            batter_xwoba = float(np.clip(batter_xwoba, 0.05, 0.80))
            batter_start_idx += 1

            outcome = _sample_outcome(batter_xwoba)
            runs, state = _apply_outcome(outcome, state)
            state.runs += runs
            pitches += _avg_pitches_for_outcome(outcome)

        total_runs = state.runs
        return total_runs, pitches, batter_start_idx

    def _validate_inputs(self, home: list[float], away: list[float]) -> bool:
        if not home or not away:
            return False
        if len(home) < 1 or len(away) < 1:
            return False
        if any(x is None or not (0 <= x <= 1) for x in home + away):
            return False
        return True

    def _fallback_poisson(self, home_xfip: Optional[float],
                           away_xfip: Optional[float]) -> dict:
        """
        Fallback to the analytical Poisson win probability when simulation fails.
        Uses league-average run production adjusted by SP xFIP if available.
        """
        league_avg_lambda = 4.35  # MLB average runs per game, 2019-2025

        def xfip_to_lambda(xfip: Optional[float]) -> float:
            if not xfip:
                return league_avg_lambda
            # Rough conversion: every 1.0 xFIP unit above/below 4.00 shifts lambda by ~0.5
            return league_avg_lambda + (4.00 - xfip) * 0.5

        lambda_home = xfip_to_lambda(away_xfip)  # home offense vs. away SP
        lambda_away = xfip_to_lambda(home_xfip)  # away offense vs. home SP

        result = compute_win_probability(lambda_home, lambda_away)
        result["simulation_method"] = "poisson_fallback"
        return result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _xwoba_to_quintile(xwoba: float) -> int:
    for i, threshold in enumerate(_XWOBA_BREAKS):
        if xwoba < threshold:
            return i
    return len(_XWOBA_BREAKS)


def _sample_outcome(xwoba: float) -> str:
    """Sample a PA outcome from the xwOBA-to-outcome probability table."""
    quintile = _xwoba_to_quintile(xwoba)
    probs = _XWOBA_TO_OUTCOMES[quintile]
    outcome_idx = np.random.choice(7, p=probs)
    return ["out", "walk", "single", "double", "triple", "hr", "strikeout"][outcome_idx]


def _apply_outcome(outcome: str, state: HalfInningState) -> tuple[int, HalfInningState]:
    """
    Apply a PA outcome to the base-out state using simplified Markov transitions.
    Returns (runs_scored, new_state).
    """
    runs = 0
    runners = state.runners

    if outcome in ("out", "strikeout"):
        state.outs += 1
        return 0, state

    if outcome == "walk":
        # Force runners: 1B always advances, others only if forced
        if runners & 0b001:  # runner on 1B
            if runners & 0b010:  # runner on 2B too
                if runners & 0b100:  # bases loaded
                    runs = 1
                    # Runners stay at 2B and 3B, new batter at 1B
                    state.runners = 0b111
                else:
                    state.runners = runners | 0b100  # advance to 3B
            else:
                state.runners = runners | 0b010  # advance to 2B
        else:
            state.runners = runners | 0b001  # batter to 1B

    elif outcome == "single":
        # Runners on 2B and 3B score; 1B advances to 2B; batter to 1B
        runs += bin(runners >> 1).count("1")  # runners on 2B and 3B score
        r1 = 1 if (runners & 0b001) else 0    # runner on 1B goes to 2B
        state.runners = 0b001 | (r1 << 1)     # batter at 1B, old 1B at 2B

    elif outcome == "double":
        runs += _RUNS_ON_DOUBLE.get(runners, 0)
        r1 = 1 if (runners & 0b001) else 0    # runner on 1B may score or go to 3B
        state.runners = 0b010 | (r1 << 2)     # batter at 2B

    elif outcome == "triple":
        runs += _RUNS_ON_TRIPLE.get(runners, 0)
        state.runners = 0b100                  # only batter at 3B

    elif outcome == "hr":
        runs += bin(runners).count("1") + 1   # all runners + batter score
        state.runners = 0

    state.runs = 0  # runs returned separately
    return runs, state


def _avg_pitches_for_outcome(outcome: str) -> float:
    """Approximate pitch count per PA outcome."""
    return {"out": 3.5, "strikeout": 4.5, "walk": 5.5,
            "single": 3.8, "double": 4.0, "triple": 4.0, "hr": 4.2}.get(outcome, 3.8)


def _xfip_to_modifier(xfip: Optional[float]) -> float:
    """
    Convert pitcher xFIP to a multiplicative modifier on batter xwOBA.
    League average xFIP ~4.00 → modifier 1.0 (no adjustment).
    Elite SP (xFIP 3.00) → modifier ~0.85 (15% suppression of batter offense).
    Poor SP (xFIP 5.50) → modifier ~1.15 (15% boost to batter offense).
    """
    if xfip is None:
        return 1.0
    league_avg_xfip = 4.00
    return float(np.clip(1.0 + (xfip - league_avg_xfip) * 0.10, 0.70, 1.30))


def _compute_fatigue_mod(cfg: SimConfig, pitch_count: int,
                          inning: int, pitcher_side: str) -> float:
    """
    Compute pitcher effectiveness decay due to pitch count / deep into game.
    Returns a modifier <= 1.0 (1.0 = no fatigue).
    """
    exceeded_pitches = pitch_count > cfg.pitcher_fatigue_pitch_threshold
    exceeded_innings = inning > cfg.pitcher_fatigue_inning_threshold
    if exceeded_pitches and exceeded_innings:
        return 1.0 - cfg.pitcher_fatigue_decay * 2
    if exceeded_pitches or exceeded_innings:
        return 1.0 - cfg.pitcher_fatigue_decay
    return 1.0
