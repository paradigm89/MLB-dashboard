"""
src/pipeline/ingest_qc.py

Quality-control checks for the historical data ingest.

Compares the local database against a sampling of data points fetched
directly from Baseball Savant / FanGraphs to verify the ingest was complete
and accurate.

Checks performed
----------------
1. Coverage    -- every expected 14-day chunk in each season has Statcast rows
2. Boundaries  -- first and last pitch of each chunk have the correct game_date
3. Stat counts -- batting/pitching/fielding row counts are within plausible ranges
4. Value sanity-- flags impossible Statcast values (speed, spin, null game_pk)
5. Spot-check  -- re-fetches N random chunks per season from Baseball Savant;
                  compares row counts and a random sample of 10 individual
                  pitches (game_pk, pitcher_id, pitch_type, release_speed)

Usage (inside Docker)
---------------------
    # Fast checks only (no source re-fetch):
    docker compose exec backend python -m src.pipeline.ingest_qc --seasons 2019 2025 --spot-checks 0

    # Full check — re-fetches 1 random chunk per season from Baseball Savant:
    docker compose exec backend python -m src.pipeline.ingest_qc --seasons 2019 2025

    # More thorough — re-fetches 2 chunks per season:
    docker compose exec backend python -m src.pipeline.ingest_qc --seasons 2019 2025 --spot-checks 2
"""

import argparse
import logging
import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy import func

from src.db.connection import get_db
from src.db.schema import BattingStats, FieldingStats, PitchingStats, StatcastPitch

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

        report.checks.append(check_coverage(season))
        report.checks.append(check_boundaries(season))
        report.checks.extend(check_stat_counts(season))
        report.checks.append(check_value_sanity(season))

        if pb and n_spot_checks > 0:
            chunks = _season_chunks(season)
            # Prefer mid-season chunks; skip first and last (may be partial weeks)
            eligible = chunks[1:-1] if len(chunks) > 2 else chunks
            selected = random.sample(eligible, min(n_spot_checks, len(eligible)))
            for chunk_start, chunk_end in selected:
                report.checks.append(spot_check_chunk(season, chunk_start, chunk_end, pb))

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
