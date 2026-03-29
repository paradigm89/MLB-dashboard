# MLB Prediction Dashboard — Claude Session Memory

This file is read automatically by Claude Code at the start of every session.
It captures the current implementation state so work can resume seamlessly after any interruption.

---

## What This Project Is

A full-stack MLB game prediction system that produces:
- **Win probability** for any matchup (home vs. away)
- **Expected runs** (with P10–P90 distribution) per team
- **Pitcher vs. opponent** projections (xFIP, K%, BB%)
- **Batter vs. pitcher** matchup scores (xwOBA per PA, Bayesian-blended)

**Stack**: Python 3.11 (FastAPI + ML pipeline) · PostgreSQL · React/Next.js frontend
**Data**: pybaseball (Statcast + FanGraphs) + MLB-StatsAPI · Historical: 2019–2025, current season: live
**Deployment target**: Railway (or Docker Compose on a VPS)

---

## Key Design Decisions

- **Models implement `BaseMLBModel`** (`src/models/base.py`) — swappable via `config/models.yaml`
- **Features defined in `config/features.yaml`** — add/remove features without code changes
- **Scheduler jobs defined in `config/schedule.yaml`** — change refresh times without code changes
- **Data sources behind adapter classes** (`src/pipeline/adapters/`) — swappable without touching the pipeline
- **Run expectancy uses XGBoost with `count:poisson` objective** → outputs λ → Skellam win prob (no classifier needed)
- **Batter matchup uses LightGBM** (speed on millions of pitch rows) targeting xwOBA per PA
- **Monte Carlo simulator** (10,000 games) derives final win prob and score distribution; no direct binary classifier
- **Bayesian shrinkage** throughout: small-sample matchup stats blend toward population priors using `n / (n + k)`
- **Weather/umpire features** are included as feature slots; they fill as `None`/`0.0` pre-game (XGBoost handles NaN natively)

---

## Project Structure

```
mlb-dashboard/
├── config/
│   ├── features.yaml       ✅ done
│   ├── models.yaml         ✅ done
│   └── schedule.yaml       ✅ done
├── data/models/            ✅ created (trained model artifacts stored here)
├── src/
│   ├── pipeline/
│   │   ├── adapters/
│   │   │   ├── base.py     ⬜ todo
│   │   │   ├── mlb_stats.py ⬜ todo
│   │   │   └── pybaseball.py ⬜ todo
│   │   ├── ingest.py       ⬜ todo
│   │   ├── features.py     ⬜ todo
│   │   ├── matchups.py     ⬜ todo
│   │   ├── lineups.py      ⬜ todo
│   │   ├── daily_refresh.py ⬜ todo
│   │   └── retrain_scheduler.py ⬜ todo
│   ├── db/
│   │   ├── schema.py       ✅ done
│   │   ├── connection.py   ✅ done
│   │   └── queries.py      ✅ done
│   ├── models/
│   │   ├── base.py         ⬜ todo
│   │   ├── run_expectancy.py ⬜ todo
│   │   ├── win_probability.py ⬜ todo
│   │   ├── pitcher_model.py ⬜ todo
│   │   ├── batter_matchup.py ⬜ todo
│   │   └── simulator.py    ⬜ todo
│   ├── training/
│   │   ├── train_all.py    ⬜ todo
│   │   └── evaluate.py     ⬜ todo
│   └── api/
│       ├── main.py         ⬜ todo
│       ├── routers/
│       │   ├── predictions.py ⬜ todo
│       │   ├── stats.py    ⬜ todo
│       │   └── system.py   ⬜ todo
│       └── schemas.py      ⬜ todo
├── frontend/               ⬜ todo (Next.js)
├── notebooks/              ⬜ todo (EDA stubs)
├── docker-compose.yml      ✅ done
├── requirements.txt        ✅ done
├── .env.example            ✅ done
└── CLAUDE.md               ✅ this file
```

---

## Implementation Status

### ✅ Completed
- `config/features.yaml` — all feature definitions for all models
- `config/models.yaml` — XGBoost/LightGBM hyperparameters, Bayesian shrinkage constants, simulator settings
- `config/schedule.yaml` — cron timing for all refresh/retrain jobs
- `src/db/schema.py` — all SQLAlchemy table definitions (games, batting/pitching/fielding stats, statcast_pitches, park_factors, elo_ratings, lineups, predictions, refresh_log, model_versions, umpire_stats)
- `src/db/connection.py` — DB connection management via DATABASE_URL env var
- `src/db/queries.py` — reusable query helpers
- `docker-compose.yml` — PostgreSQL + backend + frontend services
- `requirements.txt` — all Python dependencies
- `.env.example` — environment variable template
- `src/Dockerfile` — fixed (build context = project root)

### ⬜ Next Up
1. **DB migrations**: Run `alembic init alembic` inside the project, configure `alembic.ini` and `env.py`, generate initial migration from `schema.py`, commit
2. **Pipeline adapters**: `src/pipeline/adapters/base.py`, `mlb_stats.py`, `pybaseball.py`
3. **Ingest pipeline**: `src/pipeline/ingest.py` with resumable historical fetch (2019–2025)
4. **Feature engineering**: `src/pipeline/features.py` reading from DB, driven by `config/features.yaml`
5. **Matchup aggregation**: `src/pipeline/matchups.py` — Bayesian batter vs. pitcher xwOBA
6. **Lineup logic**: `src/pipeline/lineups.py`
7. **ML models**: `src/models/base.py` → `run_expectancy.py` → `win_probability.py` → `pitcher_model.py` → `batter_matchup.py` → `simulator.py`
8. **Training**: `src/training/train_all.py` + `evaluate.py`
9. **Refresh/retrain**: `src/pipeline/daily_refresh.py` + `retrain_scheduler.py`
10. **FastAPI**: `src/api/main.py` + all routers + schemas
11. **Frontend**: Next.js dashboard

---

## Known Limitations (from design review)

1. **xFIP per game has high variance** — outputs framed as distributions, not point estimates; confidence intervals are intentionally wide
2. **Official lineup timing** — available 3–20 minutes before game; 3 PM and 5 PM refreshes cover most games; predictions always show whether lineup is confirmed or projected
3. **Monte Carlo simulator built incrementally** — validate batter matchup model first, then plug into simulator; if score distributions are unrealistic, investigate and fix before surfacing to UI
4. **Weather features** — present as feature slots but typically null pre-game; XGBoost handles missing values natively; flagged in UI
5. **Statcast initial load** — ~5–6M rows, takes several hours; ingest script is resumable via `last_ingested_date` in DB
6. **Umpire pre-game** — umpire ID filled post-game; pre-game uses 0.0 (league average delta)

---

## Running Locally

```bash
cp .env.example .env          # fill in DB_PASSWORD
docker compose up -d          # starts PostgreSQL, backend, frontend
# Initial historical ingest (one-time, takes several hours):
docker compose exec backend python -m src.pipeline.ingest --seasons 2019 2025
# Run training after ingest completes:
docker compose exec backend python -m src.training.train_all
```

---

## Branch
`claude/mlb-prediction-model-RsCvq`
