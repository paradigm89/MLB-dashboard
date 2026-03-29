"""
src/pipeline/ingest_qc.py

Quality-control checks for the historical data ingest.

Compares the local database against a sampling of data points fetched
directly from Baseball Savant / FanGraphs / MLB Stats API to verify the
ingest was complete and accurate.

Checks performed
----------------
1. Coverage       -- every expected 14-day chunk in each season has Statcast rows
2. Boundaries     -- first and last pitch of each chunk have the correct game_date
3. Stat counts    -- batting/pitching/fielding row counts are within plausible ranges
4. Value sanity   -- flags impossible Statcast values (speed, spin, null game_pk)
5. Game counts    -- games table has the expected number of Final games per season
6. Spot-check     -- re-fetches N random chunks per season from Baseball Savant;
                     compares row counts and a random sample of 10 individual
                     pitches (game_pk, pitcher_id, pitch_type, release_speed)
7. Game results   -- re-fetches 10 random game scores from the MLB Stats API and
                     compares home/away score exactly against the DB
8. Pitcher stats  -- re-fetches 3 random qualified pitchers from FanGraphs and
                     compares ERA (±0.20) and K% (±0.5 pp) against the DB
9. Batter stats   -- re-fetches 3 random qualified batters from FanGraphs and
                     compares AVG (±0.003) and OPS (±0.010) against the DB

Usage (inside Docker)
---------------------
    # Fast checks only (no source re-fetch, ~30 seconds):
    docker compose exec backend python -m src.pipeline.ingest_qc --seasons 2019 2025 --spot-checks 0

    # Full check — re-fetches from source for all network checks (~15 min):
    docker compose exec backend python -m src.pipeline.ingest_qc --seasons 2019 2025

    # More thorough Statcast spot-check (2 chunks per season):
    docker compose exec backend python -m src.pipeline.ingest_qc --seasons 2019 2025 --spot-checks 2
"""

import argparse
import logging
import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy import func

from src.db.connection import get_db
from src.db.schema import BattingStats, FieldingStats, Game, PitchingStats, StatcastPitch

# Expected range of Final regular-season games per season (home + away = 1 game).
# Tolerance of ±30 covers typical postponements / makeup games.
# 2020 was a 60-game COVID season; 2025 may be mid-season depending on run date.
EXPECTED_GAME_RANGE: dict[int, tuple[int, int]] = {
    2019: (2400, 2460),
    2020: (870,  930),
    2021: (2400, 2460),
    2022: (2400, 2460),
    2023: (2400, 2460),
    2024: (2400, 2460),
    2025: (50,   2460),  # open upper bound — season may still be in progress
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Season calendar
# ---------------------------------------------------------------------------

# Approximate MLB regular-season start/end per year.
# These bound the coverage and boundary checks; exact game dates vary slightly.
SEASON_DATES: dict[int, tuple[date, date]] = {
    2019: (date(2019, 3, 28), date(2019, 9, 29)),
    2020: (date(2020, 7, 23), date(2020, 9, 27)),  # COVID-shortened
    2021: (date(2021, 4, 1),  date(2021, 10, 3)),
    2022: (date(2022, 4, 7),  date(2022, 10, 5)),
    2023: (date(2023, 3, 30), date(2023, 10, 1)),
    2024: (date(2024, 3, 20), date(2024, 9, 29)),
    2025: (date(2025, 3, 27), date(2025, 9, 28)),
}

CHUNK_DAYS = 14  # must match the chunk size used in ingest.py


def _season_chunks(season: int) -> list[tuple[date, date]]:
    """Return (chunk_start, chunk_end) pairs covering the regular season."""
    if season not in SEASON_DATES:
        return []
    start, end = SEASON_DATES[season]
    chunks = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS - 1), end)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

# Severity ordering for max() comparisons
_SEVERITY = {PASS: 0, WARN: 1, FAIL: 2}


@dataclass
class CheckResult:
    name: str
    status: str   # PASS | WARN | FAIL
    detail: str = ""


@dataclass
class SeasonReport:
    season: int
    checks: list = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(CheckResult(name, status, detail))

    @property
    def overall(self) -> str:
        if not self.checks:
            return WARN
        return max((c.status for c in self.checks), key=lambda s: _SEVERITY[s])


# ---------------------------------------------------------------------------
# Check 1: Coverage
# ---------------------------------------------------------------------------

def check_coverage(season: int) -> CheckResult:
    """Verify every expected 14-day chunk has at least one Statcast row in the DB."""
    chunks = _season_chunks(season)
    if not chunks:
        return CheckResult("coverage", WARN, f"No calendar entry for season {season}")

    missing: list[str] = []
    with get_db() as session:
        for chunk_start, chunk_end in chunks:
            count = (
                session.query(func.count(StatcastPitch.id))
                .filter(
                    StatcastPitch.season == season,
                    StatcastPitch.game_date >= str(chunk_start),
                    StatcastPitch.game_date <= str(chunk_end),
                )
                .scalar()
            )
            if count == 0:
                missing.append(f"{chunk_start}…{chunk_end}")

    if not missing:
        return CheckResult("coverage", PASS, f"All {len(chunks)} chunks present")

    pct_missing = len(missing) / len(chunks) * 100
    status = FAIL if pct_missing > 20 else WARN
    sample = ", ".join(missing[:4]) + (" …" if len(missing) > 4 else "")
    return CheckResult(
        "coverage", status,
        f"{len(missing)}/{len(chunks)} chunks empty: {sample}",
    )


# ---------------------------------------------------------------------------
# Check 2: Boundary rows
# ---------------------------------------------------------------------------

def check_boundaries(season: int) -> CheckResult:
    """Confirm the first and last pitch date of every chunk falls within that chunk."""
    chunks = _season_chunks(season)
    issues: list[str] = []

    with get_db() as session:
        for chunk_start, chunk_end in chunks:
            base = (
                session.query(StatcastPitch.game_date)
                .filter(
                    StatcastPitch.season == season,
                    StatcastPitch.game_date >= str(chunk_start),
                    StatcastPitch.game_date <= str(chunk_end),
                )
            )
            first_row = base.order_by(StatcastPitch.game_date.asc()).first()
            if first_row is None:
                # Already flagged by coverage check; skip
                continue
            last_row = base.order_by(StatcastPitch.game_date.desc()).first()

            first_date = first_row[0]
            last_date = last_row[0]

            if not (str(chunk_start) <= first_date <= str(chunk_end)):
                issues.append(
                    f"chunk {chunk_start}: first row date {first_date!r} out of range"
                )
            if not (str(chunk_start) <= last_date <= str(chunk_end)):
                issues.append(
                    f"chunk {chunk_end}: last row date {last_date!r} out of range"
                )

    if not issues:
        return CheckResult("boundaries", PASS, f"All {len(chunks)} chunk boundaries valid")
    sample = "; ".join(issues[:3]) + (" …" if len(issues) > 3 else "")
    return CheckResult("boundaries", WARN, sample)


# ---------------------------------------------------------------------------
# Check 3: Stat table counts
# ---------------------------------------------------------------------------

def check_stat_counts(season: int) -> list[CheckResult]:
    """Verify batting/pitching/fielding row counts are within plausible ranges."""
    # COVID-shortened 2020 has far fewer players/games
    is_short = season == 2020
    thresholds = {
        "batting_stats":  (BattingStats,  200 if is_short else 500),
        "pitching_stats": (PitchingStats, 100 if is_short else 300),
        "fielding_stats": (FieldingStats, 100 if is_short else 400),
    }
    results = []
    with get_db() as session:
        for label, (model, minimum) in thresholds.items():
            count = (
                session.query(func.count(model.id))
                .filter(model.season == season)
                .scalar()
            )
            if count == 0:
                results.append(CheckResult(label, FAIL, "0 rows — not ingested"))
            elif count < minimum:
                results.append(CheckResult(label, WARN, f"{count:,} rows (expected ≥{minimum:,})"))
            else:
                results.append(CheckResult(label, PASS, f"{count:,} rows"))
    return results


# ---------------------------------------------------------------------------
# Check 4: Value sanity
# ---------------------------------------------------------------------------

def check_value_sanity(season: int) -> CheckResult:
    """Flag rows whose Statcast values fall outside physically possible ranges."""
    issues: list[str] = []
    with get_db() as session:
        # Release speed: valid range 40–110 mph
        bad_speed = (
            session.query(func.count(StatcastPitch.id))
            .filter(
                StatcastPitch.season == season,
                StatcastPitch.release_speed.isnot(None),
                (StatcastPitch.release_speed < 40) | (StatcastPitch.release_speed > 110),
            )
            .scalar()
        )
        if bad_speed:
            issues.append(f"{bad_speed:,} pitches with release_speed outside 40–110 mph")

        # Spin rate: valid range 500–4,000 rpm
        bad_spin = (
            session.query(func.count(StatcastPitch.id))
            .filter(
                StatcastPitch.season == season,
                StatcastPitch.release_spin_rate.isnot(None),
                (StatcastPitch.release_spin_rate < 500) | (StatcastPitch.release_spin_rate > 4000),
            )
            .scalar()
        )
        if bad_spin:
            issues.append(f"{bad_spin:,} pitches with spin_rate outside 500–4,000 rpm")

        # Null game_pk should never happen
        null_gp = (
            session.query(func.count(StatcastPitch.id))
            .filter(
                StatcastPitch.season == season,
                StatcastPitch.game_pk.is_(None),
            )
            .scalar()
        )
        if null_gp:
            issues.append(f"{null_gp:,} rows with null game_pk")

    if not issues:
        return CheckResult("value_sanity", PASS, "No impossible values detected")
    return CheckResult("value_sanity", WARN, "; ".join(issues))


# ---------------------------------------------------------------------------
# Check 5: Spot-check (re-fetch from source)
# ---------------------------------------------------------------------------

def spot_check_chunk(
    season: int,
    chunk_start: date,
    chunk_end: date,
    pb,
) -> CheckResult:
    """
    Re-fetch one 14-day chunk from Baseball Savant and compare against the DB.

    Row count tolerance: ±5% (small differences are normal due to Statcast
    data corrections Baseball Savant applies after the fact).

    Field-level: randomly samples up to 10 pitches from the source and
    looks each one up in the DB by (game_pk, pitcher_id, pitch_type),
    then checks release_speed matches within ±0.5 mph.
    """
    name = f"spot_{chunk_start}"
    logger.info("Spot-check: re-fetching %s..%s from Baseball Savant", chunk_start, chunk_end)

    try:
        source_df = pb.get_statcast_range(str(chunk_start), str(chunk_end))
    except Exception as exc:
        return CheckResult(name, WARN, f"Source fetch failed: {exc}")

    if source_df.empty:
        return CheckResult(name, WARN, "Source returned 0 rows (possible off-season period)")

    source_count = len(source_df)

    with get_db() as session:
        db_count = (
            session.query(func.count(StatcastPitch.id))
            .filter(
                StatcastPitch.season == season,
                StatcastPitch.game_date >= str(chunk_start),
                StatcastPitch.game_date <= str(chunk_end),
            )
            .scalar()
        )

    diff = db_count - source_count
    pct_diff = abs(diff) / source_count * 100 if source_count else 0
    count_status = PASS if pct_diff <= 5 else (WARN if pct_diff <= 15 else FAIL)
    count_detail = f"DB={db_count:,} source={source_count:,} (Δ{diff:+,}, {pct_diff:.1f}%)"

    # Field-level spot check: sample up to 10 pitches from the source batch
    sample_size = min(10, len(source_df))
    sample = source_df.sample(n=sample_size, random_state=42)

    mismatches: list[str] = []
    with get_db() as session:
        for _, src_row in sample.iterrows():
            # pybaseball may use 'pitcher' or 'pitcher_id' for the pitcher column
            gp = src_row.get("game_pk") or src_row.get("gamePk")
            pid = src_row.get("pitcher") or src_row.get("pitcher_id")
            ptype = src_row.get("pitch_type")
            src_spd = src_row.get("release_speed")

            if not gp or not pid:
                continue

            db_row = (
                session.query(StatcastPitch)
                .filter(
                    StatcastPitch.game_pk == int(gp),
                    StatcastPitch.pitcher_id == int(pid),
                    StatcastPitch.pitch_type == ptype,
                )
                .first()
            )

            if db_row is None:
                mismatches.append(
                    f"game_pk={int(gp)} pitcher={int(pid)} pitch_type={ptype}: not in DB"
                )
            elif src_spd and db_row.release_speed:
                delta = abs(float(src_spd) - float(db_row.release_speed))
                if delta > 0.5:
                    mismatches.append(
                        f"game_pk={int(gp)}: release_speed "
                        f"source={src_spd} db={db_row.release_speed}"
                    )

    matched = sample_size - len(mismatches)
    field_status = PASS if not mismatches else (WARN if len(mismatches) <= 2 else FAIL)
    field_detail = (
        f"{matched}/{sample_size} sampled pitches match"
        if not mismatches
        else f"{len(mismatches)}/{sample_size} mismatches: "
             + "; ".join(mismatches[:2])
             + (" …" if len(mismatches) > 2 else "")
    )

    overall = max(count_status, field_status, key=lambda s: _SEVERITY[s])
    return CheckResult(name, overall, f"Counts: {count_detail} | Fields: {field_detail}")


# ---------------------------------------------------------------------------
# Check 5: Game counts (offline)
# ---------------------------------------------------------------------------

def check_game_counts(season: int) -> CheckResult:
    """
    Verify the games table has a plausible number of completed games.

    Checks:
    - Total Final games is within the expected range for the season format.
    - No Final games are missing a score (home_score or away_score is null).
    - home_win is set for all Final games (training target is populated).
    """
    issues: list[str] = []

    with get_db() as session:
        total_final = (
            session.query(func.count(Game.id))
            .filter(Game.season == season, Game.status == "Final")
            .scalar()
        )
        missing_score = (
            session.query(func.count(Game.id))
            .filter(
                Game.season == season,
                Game.status == "Final",
                (Game.home_score.is_(None)) | (Game.away_score.is_(None)),
            )
            .scalar()
        )
        missing_outcome = (
            session.query(func.count(Game.id))
            .filter(
                Game.season == season,
                Game.status == "Final",
                Game.home_win.is_(None),
            )
            .scalar()
        )

    lo, hi = EXPECTED_GAME_RANGE.get(season, (50, 2460))
    if total_final == 0:
        return CheckResult("game_counts", FAIL, "0 Final games — games table not ingested")
    if not (lo <= total_final <= hi):
        issues.append(f"{total_final:,} Final games (expected {lo:,}–{hi:,})")
    if missing_score:
        issues.append(f"{missing_score:,} Final games missing a score")
    if missing_outcome:
        issues.append(f"{missing_outcome:,} Final games missing home_win (training target)")

    if not issues:
        return CheckResult("game_counts", PASS, f"{total_final:,} Final games")
    status = FAIL if total_final == 0 or missing_score else WARN
    return CheckResult("game_counts", status, "; ".join(issues))


# ---------------------------------------------------------------------------
# Check 6: Statcast spot-check (re-fetch from Baseball Savant)
# (moved up from original check 5 — logic unchanged)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Check 7: Game result accuracy (re-fetch scores from MLB Stats API)
# ---------------------------------------------------------------------------

def spot_check_game_results(season: int, n: int = 10) -> CheckResult:
    """
    Re-fetch final scores for N random completed games from the MLB Stats API
    and compare home_score / away_score exactly against the DB.

    Exact score matches are required — even a 1-run difference means the
    game result stored in the DB is wrong and would corrupt model training.
    """
    import statsapi

    with get_db() as session:
        rows = (
            session.query(Game.game_pk, Game.home_score, Game.away_score,
                          Game.game_date, Game.home_team_abbr, Game.away_team_abbr)
            .filter(
                Game.season == season,
                Game.status == "Final",
                Game.home_score.isnot(None),
                Game.away_score.isnot(None),
            )
            .all()
        )

    if not rows:
        return CheckResult("game_result_spot", WARN, "No Final games in DB to check")

    sample = random.sample(rows, min(n, len(rows)))
    mismatches: list[str] = []
    errors: list[str] = []

    for row in sample:
        game_pk, db_home, db_away, game_date, home_abbr, away_abbr = row
        try:
            raw = statsapi.boxscore_data(game_pk)
            api_home = raw.get("home", {}).get("teamStats", {}).get("batting", {}).get("runs")
            api_away = raw.get("away", {}).get("teamStats", {}).get("batting", {}).get("runs")
        except Exception as exc:
            errors.append(f"game_pk={game_pk}: API error ({exc})")
            continue

        if api_home is None or api_away is None:
            errors.append(f"game_pk={game_pk}: API returned no score")
            continue

        if int(api_home) != int(db_home) or int(api_away) != int(db_away):
            mismatches.append(
                f"{away_abbr}@{home_abbr} {game_date} "
                f"DB={db_away}-{db_home} API={api_away}-{api_home}"
            )

    checked = len(sample) - len(errors)
    if mismatches:
        return CheckResult(
            "game_result_spot", FAIL,
            f"{len(mismatches)}/{checked} scores wrong: " + "; ".join(mismatches[:3]),
        )
    if errors:
        detail = f"{checked}/{len(sample)} checked (API errors on {len(errors)})"
        return CheckResult("game_result_spot", WARN, detail)
    return CheckResult("game_result_spot", PASS, f"All {checked} sampled scores match")


# ---------------------------------------------------------------------------
# Check 8: Pitcher stat spot-check (re-fetch from FanGraphs)
# ---------------------------------------------------------------------------

def spot_check_pitcher_stats(season: int, n: int = 3) -> CheckResult:
    """
    Re-fetch season stats for N random qualified starters from FanGraphs and
    compare ERA (±0.20) and K% (±0.5 percentage points) against the DB.

    Uses qualified starters (ip ≥ 100) so small-sample relievers don't produce
    noisy comparisons.
    """
    import pybaseball

    with get_db() as session:
        rows = (
            session.query(
                PitchingStats.player_id, PitchingStats.player_name,
                PitchingStats.era, PitchingStats.k_pct, PitchingStats.ip,
            )
            .filter(
                PitchingStats.season == season,
                PitchingStats.ip >= 100,
                PitchingStats.era.isnot(None),
                PitchingStats.k_pct.isnot(None),
            )
            .all()
        )

    if not rows:
        return CheckResult("pitcher_stat_spot", WARN, "No qualified starters in DB to check")

    sample = random.sample(rows, min(n, len(rows)))

    try:
        fg = pybaseball.pitching_stats(season, qual=0)
    except Exception as exc:
        return CheckResult("pitcher_stat_spot", WARN, f"FanGraphs fetch failed: {exc}")

    # FanGraphs uses IDfg as the player identifier
    fg_by_id = {int(row["IDfg"]): row for _, row in fg.iterrows() if "IDfg" in fg.columns}

    mismatches: list[str] = []
    matched = 0

    for player_id, name, db_era, db_kpct, db_ip in sample:
        src = fg_by_id.get(int(player_id))
        if src is None:
            mismatches.append(f"{name}: not found in FanGraphs source")
            continue

        src_era = src.get("ERA")
        src_kpct = src.get("K%")

        era_ok = src_era is None or db_era is None or abs(float(src_era) - float(db_era)) <= 0.20
        kpct_ok = src_kpct is None or db_kpct is None or abs(float(src_kpct) - float(db_kpct)) <= 0.005

        if not era_ok:
            mismatches.append(f"{name}: ERA DB={db_era:.2f} source={src_era:.2f}")
        elif not kpct_ok:
            mismatches.append(f"{name}: K% DB={db_kpct:.3f} source={src_kpct:.3f}")
        else:
            matched += 1

    if not mismatches:
        return CheckResult("pitcher_stat_spot", PASS, f"All {matched} sampled pitchers match")
    status = FAIL if len(mismatches) >= n else WARN
    return CheckResult(
        "pitcher_stat_spot", status,
        f"{matched}/{len(sample)} match; issues: " + "; ".join(mismatches[:2]),
    )


# ---------------------------------------------------------------------------
# Check 9: Batter stat spot-check (re-fetch from FanGraphs)
# ---------------------------------------------------------------------------

def spot_check_batter_stats(season: int, n: int = 3) -> CheckResult:
    """
    Re-fetch season stats for N random qualified batters from FanGraphs and
    compare AVG (±0.003) and OPS (±0.010) against the DB.

    Uses qualified batters (pa ≥ 300) for stable comparisons.
    """
    import pybaseball

    with get_db() as session:
        rows = (
            session.query(
                BattingStats.player_id, BattingStats.player_name,
                BattingStats.avg, BattingStats.ops, BattingStats.pa,
            )
            .filter(
                BattingStats.season == season,
                BattingStats.pa >= 300,
                BattingStats.avg.isnot(None),
                BattingStats.ops.isnot(None),
            )
            .all()
        )

    if not rows:
        return CheckResult("batter_stat_spot", WARN, "No qualified batters in DB to check")

    sample = random.sample(rows, min(n, len(rows)))

    try:
        fg = pybaseball.batting_stats(season, qual=0)
    except Exception as exc:
        return CheckResult("batter_stat_spot", WARN, f"FanGraphs fetch failed: {exc}")

    fg_by_id = {int(row["IDfg"]): row for _, row in fg.iterrows() if "IDfg" in fg.columns}

    mismatches: list[str] = []
    matched = 0

    for player_id, name, db_avg, db_ops, db_pa in sample:
        src = fg_by_id.get(int(player_id))
        if src is None:
            mismatches.append(f"{name}: not found in FanGraphs source")
            continue

        src_avg = src.get("AVG")
        src_ops = src.get("OPS")

        avg_ok = src_avg is None or db_avg is None or abs(float(src_avg) - float(db_avg)) <= 0.003
        ops_ok = src_ops is None or db_ops is None or abs(float(src_ops) - float(db_ops)) <= 0.010

        if not avg_ok:
            mismatches.append(f"{name}: AVG DB={db_avg:.3f} source={src_avg:.3f}")
        elif not ops_ok:
            mismatches.append(f"{name}: OPS DB={db_ops:.3f} source={src_ops:.3f}")
        else:
            matched += 1

    if not mismatches:
        return CheckResult("batter_stat_spot", PASS, f"All {matched} sampled batters match")
    status = FAIL if len(mismatches) >= n else WARN
    return CheckResult(
        "batter_stat_spot", status,
        f"{matched}/{len(sample)} match; issues: " + "; ".join(mismatches[:2]),
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_qc(
    season_start: int = 2019,
    season_end: int = 2025,
    n_spot_checks: int = 1,
    seed: int = 42,
) -> dict[int, SeasonReport]:
    """
    Run all QC checks for each season in [season_start, season_end].

    Parameters
    ----------
    season_start, season_end
        Inclusive range of seasons to check.
    n_spot_checks
        Number of random 14-day chunks per season to re-fetch from Baseball
        Savant for comparison.  Set to 0 to run offline checks only (fast).
    seed
        Random seed used when picking which chunks to spot-check, so results
        are reproducible.
    """
    random.seed(seed)

    pb = None
    if n_spot_checks > 0:
        from src.pipeline.adapters.pybaseball import PybaseballAdapter
        pb = PybaseballAdapter()

    reports: dict[int, SeasonReport] = {}

    for season in range(season_start, season_end + 1):
        logger.info("── QC season %d ──", season)
        report = SeasonReport(season)

        # Offline checks (always run)
        report.checks.append(check_coverage(season))
        report.checks.append(check_boundaries(season))
        report.checks.extend(check_stat_counts(season))
        report.checks.append(check_value_sanity(season))
        report.checks.append(check_game_counts(season))

        # Network checks (run when n_spot_checks > 0)
        if pb and n_spot_checks > 0:
            # Statcast chunk re-fetch
            chunks = _season_chunks(season)
            eligible = chunks[1:-1] if len(chunks) > 2 else chunks
            selected = random.sample(eligible, min(n_spot_checks, len(eligible)))
            for chunk_start, chunk_end in selected:
                report.checks.append(spot_check_chunk(season, chunk_start, chunk_end, pb))

            # Game score accuracy
            report.checks.append(spot_check_game_results(season))

            # Player stat accuracy
            report.checks.append(spot_check_pitcher_stats(season))
            report.checks.append(spot_check_batter_stats(season))

        reports[season] = report

    return reports


def print_report(reports: dict[int, SeasonReport]) -> None:
    ICON = {PASS: "✓", WARN: "~", FAIL: "✗"}
    pass_count = warn_count = fail_count = 0

    print("\n" + "=" * 72)
    print("  MLB Ingest QC Report")
    print("=" * 72)

    for season, report in sorted(reports.items()):
        print(f"\nSeason {season}  [{ICON[report.overall]} {report.overall}]")
        for c in report.checks:
            print(f"  {ICON[c.status]}  {c.name:<28} {c.detail}")

        if report.overall == PASS:
            pass_count += 1
        elif report.overall == WARN:
            warn_count += 1
        else:
            fail_count += 1

    total = pass_count + warn_count + fail_count
    print("\n" + "─" * 72)
    print(
        f"  Summary: {pass_count}/{total} PASS  "
        f"{warn_count}/{total} WARN  "
        f"{fail_count}/{total} FAIL"
    )
    if fail_count:
        print("  Action: Re-run the ingest for FAIL seasons, then re-run QC.")
    elif warn_count:
        print("  Action: Review WARN items above — may be off-season gaps or minor drift.")
    else:
        print("  All seasons look good.")
    print("─" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB ingest QC checker")
    parser.add_argument(
        "--seasons", nargs=2, type=int, metavar=("START", "END"),
        default=[2019, 2025],
        help="Inclusive season range to check (default: 2019 2025)",
    )
    parser.add_argument(
        "--spot-checks", type=int, default=1, dest="spot_checks",
        help=(
            "Number of random 14-day chunks per season to re-fetch from "
            "Baseball Savant for field-level comparison. "
            "Use 0 for fast offline-only checks (default: 1)."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible chunk selection (default: 42)",
    )
    args = parser.parse_args()

    reports = run_qc(
        season_start=args.seasons[0],
        season_end=args.seasons[1],
        n_spot_checks=args.spot_checks,
        seed=args.seed,
    )
    print_report(reports)
