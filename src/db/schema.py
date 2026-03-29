"""
SQLAlchemy table definitions for the MLB prediction database.
All tables include created_at / updated_at for audit trails.
"""
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from src.db.connection import DATABASE_URL


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True)
    game_pk = Column(Integer, unique=True, nullable=False, index=True)
    game_date = Column(String(10), nullable=False, index=True)  # YYYY-MM-DD
    season = Column(Integer, nullable=False, index=True)
    game_type = Column(String(4))  # R=Regular, P=Postseason, S=Spring

    home_team_id = Column(Integer, nullable=False, index=True)
    away_team_id = Column(Integer, nullable=False, index=True)
    home_team_abbr = Column(String(5))
    away_team_abbr = Column(String(5))

    home_score = Column(Integer)
    away_score = Column(Integer)
    home_win = Column(Boolean)

    venue_id = Column(Integer)
    venue_name = Column(String(100))

    # Context
    temperature = Column(Float)
    wind_speed = Column(Float)
    wind_direction = Column(String(30))
    roof_status = Column(String(20))

    home_sp_id = Column(Integer)
    away_sp_id = Column(Integer)
    home_plate_umpire_id = Column(Integer)

    status = Column(String(20))  # Final, Postponed, etc.

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ---------------------------------------------------------------------------
# Player stats (season aggregates)
# ---------------------------------------------------------------------------

class BattingStats(Base):
    __tablename__ = "batting_stats"
    __table_args__ = (UniqueConstraint("player_id", "season", name="uq_batting_player_season"),)

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, nullable=False, index=True)
    player_name = Column(String(100))
    team_id = Column(Integer)
    team_abbr = Column(String(5))
    season = Column(Integer, nullable=False, index=True)

    # Standard
    pa = Column(Integer)
    ab = Column(Integer)
    h = Column(Integer)
    doubles = Column(Integer)
    triples = Column(Integer)
    hr = Column(Integer)
    rbi = Column(Integer)
    sb = Column(Integer)
    bb = Column(Integer)
    so = Column(Integer)
    avg = Column(Float)
    obp = Column(Float)
    slg = Column(Float)
    ops = Column(Float)

    # Advanced
    woba = Column(Float)
    wrc_plus = Column(Float)
    war = Column(Float)
    babip = Column(Float)
    iso = Column(Float)
    bb_pct = Column(Float)
    k_pct = Column(Float)
    ld_pct = Column(Float)
    gb_pct = Column(Float)
    fb_pct = Column(Float)

    # Statcast / expected stats
    xba = Column(Float)
    xslg = Column(Float)
    xwoba = Column(Float)
    xwobacon = Column(Float)
    hard_hit_pct = Column(Float)
    barrel_pct = Column(Float)
    avg_exit_velocity = Column(Float)
    avg_launch_angle = Column(Float)

    # Splits (vs LHP / RHP stored as JSON-encoded floats)
    wrc_plus_vs_lhp = Column(Float)
    wrc_plus_vs_rhp = Column(Float)
    woba_vs_lhp = Column(Float)
    woba_vs_rhp = Column(Float)
    woba_home = Column(Float)
    woba_away = Column(Float)

    bats = Column(String(1))  # L/R/S

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PitchingStats(Base):
    __tablename__ = "pitching_stats"
    __table_args__ = (UniqueConstraint("player_id", "season", name="uq_pitching_player_season"),)

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, nullable=False, index=True)
    player_name = Column(String(100))
    team_id = Column(Integer)
    team_abbr = Column(String(5))
    season = Column(Integer, nullable=False, index=True)

    # Standard
    w = Column(Integer)
    l = Column(Integer)
    era = Column(Float)
    g = Column(Integer)
    gs = Column(Integer)
    ip = Column(Float)
    so = Column(Integer)
    bb = Column(Integer)
    hr = Column(Integer)
    whip = Column(Float)

    # Advanced
    fip = Column(Float)
    xfip = Column(Float)
    siera = Column(Float)
    k_per_9 = Column(Float)
    bb_per_9 = Column(Float)
    hr_per_9 = Column(Float)
    k_pct = Column(Float)
    bb_pct = Column(Float)
    k_minus_bb_pct = Column(Float)
    lob_pct = Column(Float)
    babip = Column(Float)

    # Statcast
    xera = Column(Float)
    avg_fastball_velo = Column(Float)
    avg_spin_rate = Column(Float)
    swstr_pct = Column(Float)  # swinging strike %

    # Pitch mix percentages
    pct_ff = Column(Float)  # 4-seam fastball
    pct_si = Column(Float)  # sinker
    pct_ch = Column(Float)  # changeup
    pct_sl = Column(Float)  # slider
    pct_cu = Column(Float)  # curveball
    pct_fc = Column(Float)  # cutter
    pct_fs = Column(Float)  # splitter

    # Splits
    era_vs_lhb = Column(Float)
    era_vs_rhb = Column(Float)
    k_pct_vs_lhb = Column(Float)
    k_pct_vs_rhb = Column(Float)

    throws = Column(String(1))  # L/R

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class FieldingStats(Base):
    __tablename__ = "fielding_stats"
    __table_args__ = (UniqueConstraint("player_id", "season", "position", name="uq_fielding_player_season_pos"),)

    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, nullable=False, index=True)
    player_name = Column(String(100))
    team_id = Column(Integer)
    season = Column(Integer, nullable=False, index=True)
    position = Column(String(3), nullable=False)

    drs = Column(Float)   # Defensive Runs Saved
    uzr = Column(Float)   # Ultimate Zone Rating
    uzr_150 = Column(Float)
    rng_r = Column(Float)  # Range Runs
    err_r = Column(Float)  # Error Runs

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ---------------------------------------------------------------------------
# Statcast pitch-by-pitch
# ---------------------------------------------------------------------------

class StatcastPitch(Base):
    __tablename__ = "statcast_pitches"

    id = Column(Integer, primary_key=True)
    game_pk = Column(Integer, nullable=False, index=True)
    game_date = Column(String(10), nullable=False, index=True)
    season = Column(Integer, nullable=False, index=True)

    pitcher_id = Column(Integer, nullable=False, index=True)
    batter_id = Column(Integer, nullable=False, index=True)
    pitcher_name = Column(String(100))
    batter_name = Column(String(100))

    # Count / game state
    inning = Column(Integer)
    inning_topbot = Column(String(3))
    balls = Column(Integer)
    strikes = Column(Integer)
    outs_when_up = Column(Integer)

    # Pitch details
    pitch_type = Column(String(3), index=True)  # FF, SL, CH, CU, etc.
    release_speed = Column(Float)
    release_spin_rate = Column(Float)
    release_extension = Column(Float)
    pfx_x = Column(Float)
    pfx_z = Column(Float)
    plate_x = Column(Float)
    plate_z = Column(Float)
    sz_top = Column(Float)
    sz_bot = Column(Float)

    # Outcome
    description = Column(String(50))  # called_strike, swinging_strike, ball, hit_into_play
    events = Column(String(50), index=True)  # single, home_run, strikeout, walk, etc.
    type = Column(String(1))  # S=strike, B=ball, X=in-play

    # Hit data (when applicable)
    launch_speed = Column(Float)
    launch_angle = Column(Float)
    hit_distance_sc = Column(Float)
    estimated_ba_using_speedangle = Column(Float)
    estimated_woba_using_speedangle = Column(Float)  # xwOBA per contact

    # Handedness
    stand = Column(String(1))   # batter: L/R
    p_throws = Column(String(1))  # pitcher: L/R

    created_at = Column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Park factors
# ---------------------------------------------------------------------------

class ParkFactor(Base):
    __tablename__ = "park_factors"
    __table_args__ = (UniqueConstraint("venue_id", "season", name="uq_park_season"),)

    id = Column(Integer, primary_key=True)
    venue_id = Column(Integer, nullable=False, index=True)
    venue_name = Column(String(100))
    team_id = Column(Integer)
    season = Column(Integer, nullable=False, index=True)

    pf_runs = Column(Float)  # 100 = neutral; >100 = hitter-friendly
    pf_hr = Column(Float)
    pf_hits = Column(Float)
    pf_1b = Column(Float)
    pf_2b = Column(Float)
    pf_3b = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ---------------------------------------------------------------------------
# Elo ratings
# ---------------------------------------------------------------------------

class EloRating(Base):
    __tablename__ = "elo_ratings"
    __table_args__ = (UniqueConstraint("team_id", "after_game_pk", name="uq_elo_team_game"),)

    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, nullable=False, index=True)
    team_abbr = Column(String(5))
    after_game_pk = Column(Integer, nullable=False)
    game_date = Column(String(10), nullable=False, index=True)
    season = Column(Integer, nullable=False)
    elo = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Lineups
# ---------------------------------------------------------------------------

class Lineup(Base):
    __tablename__ = "lineups"
    __table_args__ = (UniqueConstraint("game_pk", "team_id", name="uq_lineup_game_team"),)

    id = Column(Integer, primary_key=True)
    game_pk = Column(Integer, nullable=False, index=True)
    game_date = Column(String(10), nullable=False, index=True)
    team_id = Column(Integer, nullable=False, index=True)
    is_home = Column(Boolean, nullable=False)
    is_confirmed = Column(Boolean, default=False, nullable=False)

    # Batting order: player_ids as comma-separated string (positions 1-9)
    batting_order = Column(Text)  # "656305,592450,..."

    # Starting pitcher
    sp_id = Column(Integer)
    sp_name = Column(String(100))

    source = Column(String(20))  # "official" | "projected"
    confirmed_at = Column(DateTime)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ---------------------------------------------------------------------------
# Model predictions
# ---------------------------------------------------------------------------

class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (UniqueConstraint("game_pk", "model_version_id", name="uq_prediction_game_model"),)

    id = Column(Integer, primary_key=True)
    game_pk = Column(Integer, nullable=False, index=True)
    game_date = Column(String(10), nullable=False, index=True)
    model_version_id = Column(Integer, index=True)

    # Run expectancy
    lambda_home = Column(Float)  # XGBoost predicted Poisson rate
    lambda_away = Column(Float)
    median_runs_home = Column(Float)
    median_runs_away = Column(Float)
    p10_runs_home = Column(Float)
    p90_runs_home = Column(Float)
    p10_runs_away = Column(Float)
    p90_runs_away = Column(Float)

    # Win probability (Skellam)
    win_prob_home = Column(Float)
    win_prob_away = Column(Float)

    # Sub-model scores
    pitcher_advantage_home = Column(Float)  # projected xFIP delta
    pitcher_advantage_away = Column(Float)
    lineup_xwoba_home = Column(Float)  # Monte Carlo lineup score
    lineup_xwoba_away = Column(Float)
    lineup_edge = Column(Float)         # home - away lineup xwOBA

    # Lineup state at prediction time
    home_lineup_confirmed = Column(Boolean, default=False)
    away_lineup_confirmed = Column(Boolean, default=False)

    # Actual results (filled after game)
    actual_home_score = Column(Integer)
    actual_away_score = Column(Integer)
    home_won = Column(Boolean)

    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ---------------------------------------------------------------------------
# System / operational tables
# ---------------------------------------------------------------------------

class RefreshLog(Base):
    __tablename__ = "refresh_log"

    id = Column(Integer, primary_key=True)
    refresh_type = Column(String(20), nullable=False)  # morning | afternoon | retrain
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    status = Column(String(20), nullable=False, default="running")  # running | completed | failed
    games_updated = Column(Integer)
    lineups_confirmed = Column(Integer)
    error_message = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True)
    model_type = Column(String(50), nullable=False)  # run_expectancy | win_prob | pitcher | batter | ensemble
    artifact_path = Column(String(255), nullable=False)
    trained_at = Column(DateTime, nullable=False)
    training_seasons = Column(String(50))  # "2019-2025"
    validation_season = Column(Integer)

    # Metrics
    brier_score = Column(Float)
    mae_runs = Column(Float)
    rmse_xfip = Column(Float)
    correlation_xwoba = Column(Float)

    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Umpire stats (career call tendencies)
# ---------------------------------------------------------------------------

class UmpireStats(Base):
    __tablename__ = "umpire_stats"
    __table_args__ = (UniqueConstraint("umpire_id", "season", name="uq_umpire_season"),)

    id = Column(Integer, primary_key=True)
    umpire_id = Column(Integer, nullable=False, index=True)
    umpire_name = Column(String(100))
    season = Column(Integer, nullable=False)

    games_umpired = Column(Integer)
    called_strike_rate = Column(Float)       # vs. league avg
    called_strike_rate_delta = Column(Float)  # positive = more strikes than avg

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ---------------------------------------------------------------------------
# Engine / session factory helpers
# ---------------------------------------------------------------------------

def get_engine(url: str | None = None):
    url = url or DATABASE_URL
    return create_engine(url, pool_pre_ping=True)


def get_session_factory(url: str | None = None):
    engine = get_engine(url)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def create_all_tables(url: str | None = None):
    engine = get_engine(url)
    Base.metadata.create_all(engine)
