"""
FastAPI application entry point.

Mounts all routers and starts APScheduler jobs at startup.
Scheduler job timing is driven by config/schedule.yaml.
"""
import importlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import predictions, stats, system
from src.db.schema import create_all_tables

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_SCHEDULE_CONFIG = Path(__file__).parents[2] / "config" / "schedule.yaml"

scheduler = AsyncIOScheduler()


def load_schedule_config() -> dict:
    with open(_SCHEDULE_CONFIG) as f:
        return yaml.safe_load(f)


def register_scheduler_jobs() -> None:
    """
    Load all scheduled jobs from config/schedule.yaml.
    To add a new job: add an entry to config/schedule.yaml — no code changes needed.
    """
    cfg = load_schedule_config()
    tz = cfg.get("timezone", "America/New_York")

    for job_name, job_cfg in cfg.get("jobs", {}).items():
        cron_str = job_cfg["cron"]
        func_path = job_cfg["function"]

        try:
            module_path, func_name = func_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
        except (ImportError, AttributeError) as exc:
            logger.error("Failed to load scheduler function %s: %s", func_path, exc)
            continue

        # Parse cron string "m h dom mon dow"
        parts = cron_str.split()
        trigger = CronTrigger(
            minute=parts[0], hour=parts[1],
            day=parts[2], month=parts[3], day_of_week=parts[4],
            timezone=tz,
        )
        scheduler.add_job(func, trigger, id=job_name, replace_existing=True)
        logger.info("Registered scheduled job: %s → %s @ %s ET", job_name, func_path, cron_str)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start scheduler on app startup; shut down on exit."""
    # Ensure all DB tables exist (idempotent)
    try:
        create_all_tables()
        logger.info("DB tables verified/created")
    except Exception as exc:
        logger.warning("DB table creation failed (may be expected before first migration): %s", exc)

    register_scheduler_jobs()
    scheduler.start()
    logger.info("APScheduler started with %d jobs", len(scheduler.get_jobs()))

    yield

    scheduler.shutdown()
    logger.info("APScheduler shut down")


app = FastAPI(
    title="MLB Prediction Dashboard API",
    description=(
        "Machine learning predictions for MLB games: win probability, "
        "run expectancy, pitcher matchups, and batter vs. pitcher scores."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: allow the Next.js frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(predictions.router, tags=["Predictions"])
app.include_router(stats.router, tags=["Stats"])
app.include_router(system.router, tags=["System"])


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/refresh/trigger/{refresh_type}")
async def trigger_refresh_manually(refresh_type: str):
    """
    Manually trigger a refresh (for testing or manual runs).
    refresh_type: 'morning' | 'afternoon' | 'retrain'
    """
    from src.pipeline.daily_refresh import run_morning_refresh, run_afternoon_refresh
    from src.pipeline.retrain_scheduler import run_weekly_retrain
    import asyncio

    dispatch = {
        "morning": run_morning_refresh,
        "afternoon": run_afternoon_refresh,
        "retrain": run_weekly_retrain,
    }
    fn = dispatch.get(refresh_type)
    if fn is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Unknown refresh_type: {refresh_type}")

    # Run in background so the request returns immediately
    asyncio.get_event_loop().run_in_executor(None, fn)
    return {"status": "triggered", "refresh_type": refresh_type}
