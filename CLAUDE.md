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
│   │   │   ├── base.py     ✅ done
│   │   │   ├── mlb_stats.py ✅ done
│   │   │   └── pybaseball.py ✅ done
│   │   ├── ingest.py       ✅ done
│   │   ├── features.py     ✅ done
│   │   ├── matchups.py     ✅ done
│   │   ├── lineups.py      ✅ done
│   │   ├── daily_refresh.py ✅ done
│   │   └── retrain_scheduler.py ✅ done
│   ├── db/
│   │   ├── schema.py       ✅ done
│   │   ├── connection.py   ✅ done
│   │   └── queries.py      ✅ done
│   ├── models/
│   │   ├── base.py         ✅ done
│   │   ├── run_expectancy.py ✅ done
│   │   ├── win_probability.py ✅ done
│   │   ├── pitcher_model.py ✅ done
│   │   ├── batter_matchup.py ✅ done
│   │   └── simulator.py    ✅ done
│   ├── training/
│   │   ├── train_all.py    ✅ done
│   │   └── evaluate.py     ✅ done
│   └── api/
│       ├── main.py         ✅ done
│       ├── routers/
│       │   ├── predictions.py ✅ done
│       │   ├── stats.py    ✅ done
│       │   └── system.py   ✅ done
│       └── schemas.py      ✅ done
├── frontend/               ✅ done
│   ├── Dockerfile          ✅ done
│   ├── next.config.js      ✅ done (standalone output for Docker)
│   ├── src/
│   │   ├── lib/api.ts      ✅ done
│   │   ├── components/
│   │   │   ├── RefreshStatusBar.tsx ✅ done
│   │   │   ├── ErrorBanner.tsx ✅ done
│   │   │   ├── WinProbBar.tsx ✅ done
│   │   │   ├── MatchupCard.tsx ✅ done
│   │   │   ├── BatterMatchupTable.tsx ✅ done
│   │   │   └── RunDistributionChart.tsx ✅ done
│   │   ├── pages/
│   │   │   ├── _app.tsx    ✅ done
│   │   │   ├── index.tsx   ✅ done (today's games dashboard)
│   │   │   ├── game/[id].tsx ✅ done (game detail page)
│   │   │   └── player/[id].tsx ✅ done (player spotlight)
│   │   └── styles/globals.css ✅ done
├── notebooks/              ⬜ todo (EDA stubs — low priority)
├── alembic/                ✅ done (migrations configured)
├── docker-compose.yml      ✅ done
├── requirements.txt        ✅ done
├── .env.example            ✅ done
└── CLAUDE.md               ✅ this file
```

---

## Implementation Status

### ✅ All 17 steps complete

1. Scaffold: CLAUDE.md, config/, docker-compose, requirements.txt
2. DB layer: schema.py, connection.py, queries.py, alembic setup
3. Pipeline adapters: base.py, mlb_stats.py, pybaseball.py
4. Ingest pipeline: ingest.py with resumable historical fetch
5. Lineup logic: lineups.py (official + projected)
6. Feature engineering: features.py driven by config/features.yaml
7. Matchup aggregation: matchups.py with Bayesian blending
8. ML models: BaseMLBModel, XGBoostRunModel, Skellam win prob, XGBoostPitcherModel, LGBMBatterMatchup, MonteCarloSimulator
9. Training: train_all.py + evaluate.py (walk-forward CV, Brier score)
10. Daily refresh: daily_refresh.py (9 AM + 3 PM + 5 PM ET) + retrain_scheduler.py
11. FastAPI: main.py + all routers (predictions, stats, system) + schemas.py
12. Frontend: all components + all pages + Dockerfile

### ⬜ Remaining (optional)
- `notebooks/` — EDA stub notebooks (low priority, purely documentation)
- Railway deployment — push to GitHub, connect Railway, add PostgreSQL plugin

---

## Known Limitations (from design review)

1. **xFIP per game has high variance** — outputs framed as distributions, not point estimates; confidence intervals are intentionally wide
2. **Official lineup timing** — available 3–20 minutes before game; 3 PM and 5 PM refreshes cover most games; predictions always show whether lineup is confirmed or projected
3. **Monte Carlo simulator built incrementally** — validate batter matchup model first, then plug into simulator; if score distributions are unrealistic, falls back to Poisson model and logs a warning
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

Dashboard is at http://localhost:3000 once running.

---

## Branch
`claude/mlb-prediction-model-RsCvq`
