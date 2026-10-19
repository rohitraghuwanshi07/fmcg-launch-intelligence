# FMCG Launch Intelligence

A full-stack decision platform for FMCG product launches: **React + TypeScript** frontend, **FastAPI** backend, **SQL database** (SQLite locally, PostgreSQL in Docker), JWT auth, a persisted background-job queue, and a tested analytics engine.

> FMCG launches are one-shot, expensive decisions with no sales history. This turns a plan into a **probability of success**, a **risk-adjusted strategy**, and a **pre-mortem of exactly what would have to go wrong** - every number labelled *observed, modelled or assumed*.

## Run it (pick one)

**A. Python only (the built UI is included)**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app.main:app --port 8000        # open http://localhost:8000  (API docs: /docs)
```
Create an account on the sign-in screen, make a plan, press **Run analysis**.

**B. Development (hot reload)**
```bash
make install            # python + node deps
make api                # terminal 1: backend on :8000
make web                # terminal 2: frontend on :5173 (proxies /api)
```

**C. Docker (PostgreSQL + app)**
```bash
cp .env.example .env    # set POSTGRES_PASSWORD and SECRET_KEY
docker compose up --build        # http://localhost:8000
```

Models and data ship in the repo. To rebuild: `python scripts/validate_data.py && python scripts/train_models.py` (demo models) and `python scripts/train_real.py` (real-data estimates, ~3 min).

## What is inside

| Layer | Tech | Where |
|---|---|---|
| Frontend SPA | React 18, TypeScript, Vite, Tailwind, TanStack Query, Recharts | `frontend/` |
| API | FastAPI, Pydantic v2, JWT (PyJWT) + bcrypt, request IDs, `/api/metrics`, auth rate limit | `backend/app/` |
| Database | SQLAlchemy 2, **Alembic** migrations, SQLite or PostgreSQL | `backend/app/db`, `backend/migrations` |
| Jobs | Persisted thread-pool queue (DB rows, polled by the UI, crash-safe) | `backend/app/services/jobs.py` |
| Engine | Cold-start GBM, Monte Carlo, conformal, CVaR, DE optimiser, Red Team, k-NN analogs, Thompson sampling, exact TreeSHAP | `src/` |
| Real data | 7 real datasets (scanner panels, ABS retail turnover) with SHA-256 provenance | `data/real/`, `src/real_data.py` |

**Data model:** `users` -> `projects` (a launch plan, JSON inputs) -> `assessments` (immutable snapshots: plan history & audit trail) ; `jobs` (optimizer / Red Team / Sobol / pilot runs with status, result, error).

**App pages:** Dashboard (plans + market backdrop) - per plan: Plan, Assessment (P(hurdle) ring, readiness radar, fan charts, VaR/CVaR, SHAP, recommendations, history), What-if, Optimizer, **Red Team**, Risk Lab (Sobol), Analogs, Pilot Planner, AI Analyst, PDF report - global: Real Data Lab, Model Performance, Interview Mode.

## API at a glance
`POST /api/auth/register|login` - `GET /api/auth/me` - `CRUD /api/projects` - `POST /api/projects/{id}/assess` - `GET .../latest|assessments|sensitivity|analogs|report.pdf` - `POST .../whatif|ask` - `POST .../jobs {kind: optimize|red_team|sobol|pilot}` -> `GET /api/jobs/{id}` - `GET /api/real/*` - `GET /api/analytics/overview` - `GET /api/health`, `/api/metrics`. Interactive docs at `/docs`.

## Quality gates
```bash
python -m pytest -q                      # engine + 13 end-to-end API tests (SQLite)
TEST_DATABASE_URL=postgresql+psycopg2://user:pw@host/db python -m pytest -q tests/test_backend.py   # same API tests on PostgreSQL
cd frontend && npm run typecheck && npm test && npm run build
python scripts/smoke_e2e.py             # real server + SQLite + built SPA, full user journey
```
CI (`.github/workflows/ci.yml`) runs all of the above, including PostgreSQL.

## Honest limitations
* The **launch-curve (cold-start) model is validated on simulated data**; no public real new-product launch panel was available. On held-out simulated products it does not clearly beat a category-average baseline - the platform carries that error into every probability instead of hiding it.
* **Real data** covers price/promotion response (brand-choice scanner panels from historical US supermarkets, observational) and retail seasonality/forecasting (Australian macro turnover, not SKU-level). Home-care and personal-care price response still use the simulated estimate.
* Marketing response, channel efficacy, regional surprise and Red Team plausibility scales are **assumptions** and labelled so.
* The statistical benchmark (SARIMA/Holt-Winters vs GBM) uses a small sample; differences are indicative.
* The frontend is verified by type-checking, unit tests and a production build, and the full journey by `smoke_e2e.py`; it has not been reviewed visually in a browser by the author of this README. Docker/Compose files are untested in this environment (PostgreSQL itself was tested natively).
* Auth is single-tenant email/password with JWT in `localStorage` (simple, XSS-sensitive). For production put it behind HTTPS, consider httpOnly cookies, and replace the in-memory rate limiter / thread-pool jobs with Redis + a worker when scaling beyond one node.

See `docs/ARCHITECTURE.md` and `docs/CASE_STUDY.md`. Data provenance: `data_sources.md`, `data/real/PROVENANCE.json`. Decision support only, not financial advice.
