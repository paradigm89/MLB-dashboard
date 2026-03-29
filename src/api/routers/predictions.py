"""
Prediction endpoints.

GET /api/predict/game       — full game prediction (win prob + runs + matchup scores)
GET /api/predict/pitcher    — pitcher projection for a specific game/opponent
GET /api/predict/batter     — batter vs. pitcher matchup prediction
GET /api/schedule/today     — today's games with embedded predictions
"""
import logging
from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import (
    BatterMatchupPrediction,
    GamePrediction,
    GameScheduleItem,
    PitcherInfo,
    PitcherMatchupScore,
    PitcherPrediction,
    RunDistribution,
    TeamInfo,
    TodayScheduleResponse,
)
from src.db.connection import get_db
from src.db.queries import get_last_refresh, get_prediction, get_today_predictions
from src.pipeline.adapters.mlb_stats import MLBStatsAdapter
from src.pipeline.daily_refresh import _load_active_models
from src.pipeline.features import build_prediction_features
from src.pipeline.lineups import fetch_and_store_lineups
from src.pipeline.matchups import get_batter_matchup_features, get_pitcher_vs_opponent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

_mlb = MLBStatsAdapter()


@router.get("/predict/game", response_model=GamePrediction)
async def predict_game(
    home_team_id: int = Query(..., description="Home team ID"),
    away_team_id: int = Query(..., description="Away team ID"),
    game_pk: Optional[int] = Query(None, description="MLB game_pk (preferred over team IDs)"),
    date_str: Optional[str] = Query(None, alias="date", description="Game date YYYY-MM-DD"),
):
    """
    Return the full prediction for a single game.
    Reads from the `predictions` table (populated by daily_refresh).
    Falls back to a live prediction if no stored prediction exists.
    """
    if date_str is None:
        date_str = str(date.today())

    # Try to serve from stored prediction first (fast path)
    if game_pk:
        with get_db() as session:
            stored = get_prediction(session, game_pk)
        if stored:
            return _prediction_row_to_schema(stored, home_team_id, away_team_id)

    # Fall back to on-demand prediction
    try:
        run_model, pitcher_model, _ = _load_active_models()
        features = build_prediction_features(
            game_pk or 0, home_team_id, away_team_id,
            None, None, date_str, None,
        )
        if features is None:
            raise HTTPException(status_code=404, detail="Insufficient data for prediction")

        from src.models.win_probability import compute_win_probability
        lambda_home = 4.35
        lambda_away = 4.35
        if run_model:
            lr = run_model.predict_single(features)
            lambda_home = lr["lambda_home"]
            lambda_away = lr["lambda_away"]

        win_result = compute_win_probability(lambda_home, lambda_away)
        return GamePrediction(
            game_pk=game_pk or 0,
            game_date=date_str,
            home_team=TeamInfo(team_id=home_team_id, team_abbr=""),
            away_team=TeamInfo(team_id=away_team_id, team_abbr=""),
            win_prob_home=win_result["win_prob_home"],
            win_prob_away=win_result["win_prob_away"],
            runs_home=RunDistribution(
                median=win_result["median_runs_home"],
                p10=win_result["p10_runs_home"],
                p90=win_result["p90_runs_home"],
            ),
            runs_away=RunDistribution(
                median=win_result["median_runs_away"],
                p10=win_result["p10_runs_away"],
                p90=win_result["p90_runs_away"],
            ),
            simulation_method="poisson_on_demand",
            prediction_generated_at=datetime.utcnow(),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("On-demand prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction generation failed")


@router.get("/predict/pitcher/{pitcher_id}", response_model=PitcherPrediction)
async def predict_pitcher(
    pitcher_id: int,
    opponent_team_id: Optional[int] = Query(None),
    game_date: Optional[str] = Query(None),
):
    """Project a pitcher's performance for an upcoming start."""
    gdate = game_date or str(date.today())
    opp_stats = {}
    if opponent_team_id:
        opp_stats = get_pitcher_vs_opponent(pitcher_id, opponent_team_id, gdate)

    _, pitcher_model, _ = _load_active_models()
    if pitcher_model is None:
        raise HTTPException(status_code=503, detail="Pitcher model not yet trained")

    # Build a minimal feature dict for prediction
    features = build_prediction_features(0, 0, 0, pitcher_id, None, gdate, None) or {}
    interval = pitcher_model.predict_with_interval(features)

    return PitcherPrediction(
        pitcher_id=pitcher_id,
        opponent_team_id=opponent_team_id,
        projected_xfip=interval.get("projected_xfip"),
        xfip_lower=interval.get("xfip_lower"),
        xfip_upper=interval.get("xfip_upper"),
        opponent_sample_gs=opp_stats.get("vs_opp_sample_gs"),
        note=interval.get("note", ""),
    )


@router.get("/predict/batter/{batter_id}", response_model=BatterMatchupPrediction)
async def predict_batter(
    batter_id: int,
    pitcher_id: int = Query(...),
    batting_order_pos: int = Query(5, ge=1, le=9),
):
    """Return xwOBA prediction for a batter vs. a specific pitcher."""
    features = get_batter_matchup_features(batter_id, pitcher_id, batting_order_pos)
    n_pa = features.get("n_pa_matchup", 0)

    # Bayesian blend weight: how much the matchup data contributes vs. prior
    from src.pipeline.matchups import load_shrinkage_config
    k = load_shrinkage_config().get("k_continuous", 100)
    blend_weight = round(n_pa / (n_pa + k), 3) if n_pa > 0 else 0.0

    xwoba = features.get("batter_xwoba_season")
    return BatterMatchupPrediction(
        batter_id=batter_id,
        pitcher_id=pitcher_id,
        projected_xwoba=xwoba,
        matchup_score=xwoba,
        sample_size_pa=n_pa,
        bayesian_blend_weight=blend_weight,
        batter_xwoba_vs_pitch_types={
            "4seam": features.get("batter_xwoba_vs_ff"),
            "slider": features.get("batter_xwoba_vs_sl"),
            "changeup": features.get("batter_xwoba_vs_ch"),
            "curveball": features.get("batter_xwoba_vs_cu"),
        },
    )


@router.get("/schedule/today", response_model=TodayScheduleResponse)
async def get_today_schedule():
    """Return today's games with embedded predictions and lineup confirmation status."""
    today = str(date.today())
    games_raw = _mlb.get_schedule(today)

    lineup_confirmed_count = 0
    games_out = []

    with get_db() as session:
        predictions = {p.game_pk: p for p in get_today_predictions(session, today)}
        last_refresh = get_last_refresh(session, "morning")

        # Build all output objects inside the session so ORM attributes are accessible
        for g in games_raw:
            pred = predictions.get(g["game_pk"])
            home_confirmed = bool(pred.home_lineup_confirmed) if pred else False
            away_confirmed = bool(pred.away_lineup_confirmed) if pred else False
            generated_at = pred.generated_at if pred else None
            pred_schema = _prediction_row_to_schema(pred, g["home_team_id"], g["away_team_id"]) if pred else None

            game_item = GameScheduleItem(
                game_pk=g["game_pk"],
                game_date=today,
                venue_name=g.get("venue_name"),
                home_team=TeamInfo(team_id=g["home_team_id"], team_abbr=g.get("home_team_abbr", "")),
                away_team=TeamInfo(team_id=g["away_team_id"], team_abbr=g.get("away_team_abbr", "")),
                home_sp=PitcherInfo(pitcher_id=g["home_sp_id"], pitcher_name=g.get("home_sp_name")) if g.get("home_sp_id") else None,
                away_sp=PitcherInfo(pitcher_id=g["away_sp_id"], pitcher_name=g.get("away_sp_name")) if g.get("away_sp_id") else None,
                home_lineup_confirmed=home_confirmed,
                away_lineup_confirmed=away_confirmed,
                predictions=pred_schema,
                data_as_of=generated_at,
            )
            if home_confirmed and away_confirmed:
                lineup_confirmed_count += 1
            games_out.append(game_item)

        last_refreshed = last_refresh.completed_at if last_refresh else None

    return TodayScheduleResponse(
        date=today,
        games=games_out,
        total_games=len(games_out),
        lineups_confirmed=lineup_confirmed_count,
        lineups_projected=len(games_out) - lineup_confirmed_count,
        last_refreshed=last_refreshed,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prediction_row_to_schema(pred, home_team_id: int, away_team_id: int) -> GamePrediction:
    return GamePrediction(
        game_pk=pred.game_pk,
        game_date=pred.game_date,
        home_team=TeamInfo(team_id=home_team_id, team_abbr=""),
        away_team=TeamInfo(team_id=away_team_id, team_abbr=""),
        win_prob_home=pred.win_prob_home,
        win_prob_away=pred.win_prob_away,
        runs_home=RunDistribution(
            median=pred.median_runs_home or 0,
            p10=pred.p10_runs_home or 0,
            p90=pred.p90_runs_home or 0,
        ) if pred.median_runs_home else None,
        runs_away=RunDistribution(
            median=pred.median_runs_away or 0,
            p10=pred.p10_runs_away or 0,
            p90=pred.p90_runs_away or 0,
        ) if pred.median_runs_away else None,
        home_pitcher_matchup=PitcherMatchupScore(
            projected_xfip=None,
            vs_opponent_sample_gs=None,
        ),
        home_lineup_xwoba=pred.lineup_xwoba_home,
        away_lineup_xwoba=pred.lineup_xwoba_away,
        lineup_edge=pred.lineup_edge,
        home_lineup_confirmed=pred.home_lineup_confirmed or False,
        away_lineup_confirmed=pred.away_lineup_confirmed or False,
        prediction_generated_at=pred.generated_at,
    )
