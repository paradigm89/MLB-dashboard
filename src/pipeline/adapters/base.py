"""
Abstract base class for all data source adapters.

To add a new data source: subclass DataAdapter, implement all abstract methods,
and register the adapter in src/pipeline/ingest.py. No other files need changes.
"""
from abc import ABC, abstractmethod

import pandas as pd


class DataAdapter(ABC):
    """
    Common interface for all external data sources.

    Every adapter returns data as a pandas DataFrame with a documented schema.
    Callers should not depend on any adapter-specific internals.
    """

    @abstractmethod
    def get_season_batting_stats(self, season: int) -> pd.DataFrame:
        """
        Season-level batting statistics for all qualified batters.

        Required columns: player_id, player_name, team_id, team_abbr, season,
        pa, ab, h, doubles, triples, hr, rbi, sb, bb, so, avg, obp, slg, ops,
        woba, wrc_plus, war, babip, iso, bb_pct, k_pct, ld_pct, gb_pct, fb_pct,
        xba, xslg, xwoba, xwobacon, hard_hit_pct, barrel_pct,
        avg_exit_velocity, avg_launch_angle,
        wrc_plus_vs_lhp, wrc_plus_vs_rhp, woba_vs_lhp, woba_vs_rhp,
        woba_home, woba_away, bats
        """
        ...

    @abstractmethod
    def get_season_pitching_stats(self, season: int) -> pd.DataFrame:
        """
        Season-level pitching statistics for all qualified starters + relievers.

        Required columns: player_id, player_name, team_id, team_abbr, season,
        w, l, era, g, gs, ip, so, bb, hr, whip,
        fip, xfip, siera, k_per_9, bb_per_9, hr_per_9, k_pct, bb_pct,
        k_minus_bb_pct, lob_pct, babip,
        xera, avg_fastball_velo, avg_spin_rate, swstr_pct,
        pct_ff, pct_si, pct_ch, pct_sl, pct_cu, pct_fc, pct_fs,
        era_vs_lhb, era_vs_rhb, k_pct_vs_lhb, k_pct_vs_rhb, throws
        """
        ...

    @abstractmethod
    def get_season_fielding_stats(self, season: int) -> pd.DataFrame:
        """
        Season-level fielding statistics.

        Required columns: player_id, player_name, team_id, season, position,
        drs, uzr, uzr_150, rng_r, err_r
        """
        ...

    @abstractmethod
    def get_statcast_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Pitch-by-pitch Statcast data for a date range.

        Required columns: game_pk, game_date, season,
        pitcher_id (mapped from pitcher), batter_id (mapped from batter),
        pitcher_name, batter_name,
        inning, inning_topbot, balls, strikes, outs_when_up,
        pitch_type, release_speed, release_spin_rate, release_extension,
        pfx_x, pfx_z, plate_x, plate_z, sz_top, sz_bot,
        description, events, type,
        launch_speed, launch_angle, hit_distance_sc,
        estimated_ba_using_speedangle, estimated_woba_using_speedangle,
        stand, p_throws
        """
        ...

    @abstractmethod
    def get_schedule(self, date: str) -> list[dict]:
        """
        Games scheduled for a given date (YYYY-MM-DD).

        Each dict contains: game_pk, game_date, game_type,
        home_team_id, away_team_id, home_team_abbr, away_team_abbr,
        venue_id, venue_name, status,
        home_sp_id (probable pitcher, may be None), away_sp_id
        """
        ...

    @abstractmethod
    def get_game_boxscore(self, game_pk: int) -> dict:
        """
        Boxscore for a completed or in-progress game.

        Returns a dict with keys: home_score, away_score, home_win,
        temperature, wind_speed, wind_direction, roof_status,
        home_plate_umpire_id, home_plate_umpire_name,
        home_batting_order (list of player_ids, empty if not yet confirmed),
        away_batting_order (list of player_ids, empty if not yet confirmed),
        home_lineup_confirmed (bool), away_lineup_confirmed (bool)
        """
        ...

    @abstractmethod
    def get_park_factors(self, season: int) -> pd.DataFrame:
        """
        Park factors by venue and season.

        Required columns: venue_id, venue_name, team_id, season,
        pf_runs, pf_hr, pf_hits, pf_1b, pf_2b, pf_3b
        """
        ...
