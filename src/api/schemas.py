"""
Pydantic schemas for all API request/response models.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class TeamInfo(BaseModel):
    team_id: int
    team_abbr: str


class PitcherInfo(BaseModel):
    pitcher_id: int
    pitcher_name: Optional[str] = None
    throws: Optional[str] = None


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

class RunDistribution(BaseModel):
    median: float
    p10: float
    p90: float


class PitcherMatchupScore(BaseModel):
    projected_xfip: Optional[float] = None
    xfip_lower: Optional[float] = None
    xfip_upper: Optional[float] = None
    projected_k_rate: Optional[float] = None
    projected_bb_rate: Optional[float] = None
    vs_opponent_sample_gs: Optional[int] = None
    confidence_note: str = "Wide intervals reflect inherent per-game variance"


class GamePrediction(BaseModel):
    game_pk: int
    game_date: str
    home_team: TeamInfo
    away_team: TeamInfo
    home_sp: Optional[PitcherInfo] = None
    away_sp: Optional[PitcherInfo] = None

    # Win probability
    win_prob_home: Optional[float] = Field(None, ge=0.0, le=1.0)
    win_prob_away: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Run distribution
    runs_home: Optional[RunDistribution] = None
    runs_away: Optional[RunDistribution] = None

    # Sub-model scores
    home_pitcher_matchup: Optional[PitcherMatchupScore] = None
    away_pitcher_matchup: Optional[PitcherMatchupScore] = None
    home_lineup_xwoba: Optional[float] = None
    away_lineup_xwoba: Optional[float] = None
    lineup_edge: Optional[float] = Field(None, description="home lineup xwOBA minus away lineup xwOBA")

    # Per-batter matchup scores (populated on game detail page requests)
    home_batter_matchups: Optional[list] = None
    away_batter_matchups: Optional[list] = None

    # Data quality indicators
    home_lineup_confirmed: bool = False
    away_lineup_confirmed: bool = False
    simulation_method: str = "monte_carlo"
    prediction_generated_at: Optional[datetime] = None


class PitcherPrediction(BaseModel):
    pitcher_id: int
    opponent_team_id: Optional[int] = None
    projected_xfip: Optional[float] = None
    xfip_lower: Optional[float] = None
    xfip_upper: Optional[float] = None
    projected_k_rate: Optional[float] = None
    projected_bb_rate: Optional[float] = None
    confidence_level: float = 0.80
    opponent_sample_gs: Optional[int] = None
    note: str = "Wide confidence interval reflects per-game variance, not model uncertainty"


class BatterMatchupPrediction(BaseModel):
    batter_id: int
    pitcher_id: int
    projected_xwoba: Optional[float] = None
    matchup_score: Optional[float] = None
    sample_size_pa: Optional[int] = None
    bayesian_blend_weight: Optional[float] = Field(
        None,
        description="0=full prior, 1=full observed matchup data"
    )
    batter_xwoba_vs_pitch_types: Optional[dict] = None


# ---------------------------------------------------------------------------
# Schedule / today's games
# ---------------------------------------------------------------------------

class GameScheduleItem(BaseModel):
    game_pk: int
    game_date: str
    time_et: Optional[str] = None
    venue_name: Optional[str] = None
    home_team: TeamInfo
    away_team: TeamInfo
    home_sp: Optional[PitcherInfo] = None
    away_sp: Optional[PitcherInfo] = None
    home_lineup_confirmed: bool = False
    away_lineup_confirmed: bool = False
    predictions: Optional[GamePrediction] = None
    data_as_of: Optional[datetime] = None


class TodayScheduleResponse(BaseModel):
    date: str
    games: list[GameScheduleItem]
    total_games: int
    lineups_confirmed: int
    lineups_projected: int
    last_refreshed: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TeamStats(BaseModel):
    team_id: int
    team_abbr: Optional[str] = None
    season: int
    elo: Optional[float] = None
    runs_scored_30g: Optional[float] = None
    runs_allowed_30g: Optional[float] = None
    ops_30g: Optional[float] = None
    era_30g: Optional[float] = None
    whip_30g: Optional[float] = None
    fip_30g: Optional[float] = None
    wrc_plus_30g: Optional[float] = None
    pythagorean_win_pct: Optional[float] = None
    bullpen_era_7d: Optional[float] = None


# ---------------------------------------------------------------------------
# System status
# ---------------------------------------------------------------------------

class RefreshStatus(BaseModel):
    ran_at: Optional[datetime] = None
    status: Optional[str] = None
    games_updated: Optional[int] = None
    lineups_confirmed: Optional[int] = None
    error: Optional[str] = None


class ModelVersionInfo(BaseModel):
    model_type: str
    trained_at: Optional[datetime] = None
    training_seasons: Optional[str] = None
    brier_score: Optional[float] = None
    mae_runs: Optional[float] = None
    rmse_xfip: Optional[float] = None
    correlation_xwoba: Optional[float] = None


class ActiveError(BaseModel):
    type: str
    message: str
    occurred_at: Optional[datetime] = None


class SystemStatus(BaseModel):
    last_morning_refresh: RefreshStatus
    last_afternoon_refresh: RefreshStatus
    last_retrain: RefreshStatus
    db_last_updated: Optional[datetime] = None
    active_model_versions: list[ModelVersionInfo] = []
    active_errors: list[ActiveError] = []
