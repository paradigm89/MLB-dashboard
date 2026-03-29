"""
Win Probability Model: analytical derivation from Poisson run rates.

Given lambda_home and lambda_away from the run expectancy model, computes
win probability using the Skellam distribution:

  P(home wins) = P(X - Y > 0) where X ~ Poisson(λ_h), Y ~ Poisson(λ_a)
               = Σ_{k=1}^{max_runs} P(X=k) * P(Y < k)

This approach has several advantages over a binary classifier:
1. Naturally calibrated — no Platt scaling needed
2. Forces consistency: win prob and run distribution are from the same model
3. Produces a full score distribution, not just a point probability
4. Used by FanGraphs, Baseball Prospectus, and most sabermetric systems

Extra innings: handled with a coin-flip approximation (50/50 once tied
after 9 innings). This is imprecise but rarely affects the final probability
by more than 1-2 percentage points.
"""
import logging

import numpy as np
from scipy.stats import poisson, skellam

logger = logging.getLogger(__name__)

# Maximum run total to consider in the summation (above this, P is negligible)
_MAX_RUNS = 30


def compute_win_probability(lambda_home: float, lambda_away: float) -> dict:
    """
    Compute win/loss/tie probabilities and score distribution from Poisson rates.

    Args:
        lambda_home: expected runs for home team (XGBoost run model output)
        lambda_away: expected runs for away team

    Returns:
        dict with:
          win_prob_home, win_prob_away: probability each team wins
          p_tie_regulation: probability of tie after 9 innings
          median_runs_home, median_runs_away: median of Poisson distribution
          p10_runs_home, p90_runs_home: 10th and 90th percentile run totals (home)
          p10_runs_away, p90_runs_away: (away)
          score_distribution: list of (home_runs, away_runs, probability) for the 25 most likely scores
    """
    if lambda_home <= 0:
        lambda_home = 0.5
    if lambda_away <= 0:
        lambda_away = 0.5

    # --- Win probability via Skellam CDF ------------------------------------
    # The Skellam distribution is the difference of two independent Poisson RVs.
    # P(home wins) = P(X - Y > 0) = 1 - P(X - Y <= 0)
    # We use scipy.stats.skellam for accuracy.

    # P(X > Y) = P(X - Y >= 1)
    p_home_wins_reg = float(1 - skellam.cdf(0, mu1=lambda_home, mu2=lambda_away))
    # P(Y > X) = P(X - Y <= -1)
    p_away_wins_reg = float(skellam.cdf(-1, mu1=lambda_home, mu2=lambda_away))
    # P(X == Y) = P(X - Y == 0)
    p_tie = max(0.0, 1.0 - p_home_wins_reg - p_away_wins_reg)

    # Extra innings: assume 50/50 split of tied games (coin flip)
    # This is a simplification; in reality extra-inning win% is close to 50%
    # but slightly home-team-favored (~51%) due to last-licks advantage.
    extra_innings_home_advantage = 0.51
    p_home_wins = p_home_wins_reg + p_tie * extra_innings_home_advantage
    p_away_wins = p_away_wins_reg + p_tie * (1 - extra_innings_home_advantage)

    # Normalize to sum to 1.0 (floating point safety)
    total = p_home_wins + p_away_wins
    p_home_wins /= total
    p_away_wins /= total

    # --- Score distribution (most likely scores) ----------------------------
    run_range = np.arange(0, _MAX_RUNS + 1)
    pmf_home = poisson.pmf(run_range, lambda_home)
    pmf_away = poisson.pmf(run_range, lambda_away)

    scores = []
    for h in range(0, 20):
        for a in range(0, 20):
            prob = float(pmf_home[h] * pmf_away[a])
            if prob > 0.0001:
                scores.append({"home": h, "away": a, "probability": prob})

    scores.sort(key=lambda x: x["probability"], reverse=True)
    top_scores = scores[:25]

    # --- Run distribution percentiles ---------------------------------------
    cdf_home = np.cumsum(pmf_home)
    cdf_away = np.cumsum(pmf_away)

    return {
        "win_prob_home": round(p_home_wins, 4),
        "win_prob_away": round(p_away_wins, 4),
        "p_tie_regulation": round(p_tie, 4),
        "median_runs_home": float(np.searchsorted(cdf_home, 0.5)),
        "median_runs_away": float(np.searchsorted(cdf_away, 0.5)),
        "p10_runs_home": float(np.searchsorted(cdf_home, 0.10)),
        "p90_runs_home": float(np.searchsorted(cdf_home, 0.90)),
        "p10_runs_away": float(np.searchsorted(cdf_away, 0.10)),
        "p90_runs_away": float(np.searchsorted(cdf_away, 0.90)),
        "score_distribution": top_scores,
    }


def compute_win_prob_from_lambdas(lambda_home: float, lambda_away: float) -> tuple[float, float]:
    """
    Convenience function returning just (win_prob_home, win_prob_away).
    """
    result = compute_win_probability(lambda_home, lambda_away)
    return result["win_prob_home"], result["win_prob_away"]
