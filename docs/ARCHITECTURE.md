# foreGASt — Architecture

This document describes the data, modelling, and serving stack as the code actually is today. The recent change is a data-layer migration: raw sources now land in Postgres, dbt owns the merge/precedence and the monthly panel, and `engineer.py` reads that panel by default. The model layer and UI are unchanged.

## 1. End-to-end data flow (current state)

Legend: `[NEW]` = added by the migration. `[PRE]` = pre-existing, unchanged.

```
                              ┌───────────────────────────────┐
                              │  External / on-disk sources   │
                              │                               │
                              │  • EIA API v2                 │
                              │      RNGWHHD (NG spot, daily) │
                              │      R48/SWO   (storage, wk)  │
                              │  • data/natural-gas-prices/   │
                              │      monthly.csv              │
                              │  • data/series_data/          │
                              │      NG_SUM_LSUM_DCU_NUS_M.xls│
                              │  • data/raw/                  │
                              │      CMO-Historical-Data-     │
                              │      Monthly[-2026].xlsx      │
                              │  • data/RNGWHHDd_henry_hub_   │
                              │      ...30_day_moving.xls     │
                              └───────────────┬───────────────┘
                                              │
                                              │ raw_sources.py fetchers
                                              │ (native grain, +source / loaded_at)
                                              ▼
                       ┌──────────────────────────────────────────┐
                       │  Postgres  —  schema `raw`        [NEW]  │
                       │                                          │
                       │  eia_ng_spot_daily                       │
                       │  eia_ng_storage_weekly                   │
                       │  henry_hub_spot_monthly_csv              │
                       │  eia_ng_storage_monthly_xls              │
                       │  worldbank_pinksheet_monthly             │
                       │  henry_hub_spot_daily_xls                │
                       │                                          │
                       │  postgres_loader.py:  if_exists=replace  │
                       └──────────────────────┬───────────────────┘
                                              │
                                              │ dbt run
                                              ▼
                       ┌──────────────────────────────────────────┐
                       │  Postgres  —  `dev_staging` (views) [NEW]│
                       │                                          │
                       │  stg_henry_hub   ← eia daily-mean ∪ CSV  │
                       │                    (EIA wins on overlap) │
                       │  stg_ng_storage  ← eia weekly-last ∪ XLS │
                       │                    (EIA wins on overlap) │
                       │  stg_urea        ← pinksheet (urea col)  │
                       │  stg_dap         ← pinksheet (dap col)   │
                       └──────────────────────┬───────────────────┘
                                              │
                                              ▼
                       ┌──────────────────────────────────────────┐
                       │  Postgres  —  `dev_marts` (table)  [NEW] │
                       │                                          │
                       │  mart_commodity_panel_monthly            │
                       │   month, ng_spot, urea, dap, storage_mmcf│
                       │   (full-outer-join via months CTE)       │
                       └──────────────────────┬───────────────────┘
                                              │
                                              │  build_features(source='mart')
                                              │  reads via DATABASE_URL
                                              ▼
   data/RNGWHHDd_..._30_day_moving.xls ─►┌──────────────────────────────────┐
       (daily MA features path,  [PRE])  │  backend/src/features/engineer.py│
                                         │  → wide feature frame            │
                                         │     (FEATURE_COLS + targets)     │
                                         └──────────────────┬───────────────┘
                                                            │
                              save_feature_store()          │
                                                            ▼
                              data/processed/feature_store.parquet  [PRE]
                                                            │
                                                            ▼
                       ┌──────────────────────────────────────────┐
                       │  train_models.py  [PRE]                  │
                       │   → src.models.forecaster.train          │
                       │   → XGBoost models t1 / t2 / t3 + meta   │
                       └──────────────────────┬───────────────────┘
                                              ▼
                       ┌──────────────────────────────────────────┐
                       │  run_pipeline.py  [PRE]                  │
                       │   load_models → predict                  │
                       │   → run_monte_carlo (10k paths)          │
                       │   → generate_signal                      │
                       │   history payload via                    │
                       │       pipeline.load_full_history()       │
                       └──────────────────────┬───────────────────┘
                                              ▼
                              data/processed/cache.json   [PRE]
                                              │
                                              ▼
                       app.py + pages/* (Streamlit UI)    [PRE]
```

`refresh.py` chains `train_models.py` → `run_pipeline.py`; `scheduler.py` (APScheduler) fires `refresh.run_refresh()` once on start and then daily at 06:00 UTC.

## 2. What changed vs before

**Before.** `backend/src/ingestion/pipeline.run_ingestion()` did everything in one pandas function: try the EIA API for spot + storage, fall back to the local CSV/XLS, merge live-over-backbone in-process, load Pink Sheet, reindex to a monthly grid, return a dict of Series. `engineer.py` consumed that dict directly. There was no Postgres and no dbt — the only persisted artifacts were `feature_store.parquet` and `cache.json`.

**Now.**

- **Raw is preserved at native grain.** `raw_sources.py` fetchers return each source in its original shape (daily, weekly, or monthly) with `source` / `source_file` / `loaded_at` columns. No merging or resampling happens in Python.
- **Postgres is the landing zone.** `postgres_loader.py` rewrites all six raw tables under schema `raw` on every run (`if_exists="replace"`). Each fetcher already returns full history, so a full replace is the simplest idempotent choice and refuses to run if any fetcher returns zero rows.
- **dbt owns the merge / precedence.** `stg_henry_hub` aggregates the EIA daily series to a monthly mean and unions it with the local monthly CSV, keeping the EIA value where both exist (`source_priority`). `stg_ng_storage` does the same with EIA weekly-last-in-month (×1000 BCF → MMcf) vs the local XLS. `stg_urea` and `stg_dap` are simple dedupes of the Pink Sheet table by month. The mart full-outer-joins the four staging models via a `months` CTE so a missing month in any one series still yields a row with NULL in that column.
- **The mart is the queryable monthly panel.** `dev_marts.mart_commodity_panel_monthly` (table) is intentionally the exact shape `run_ingestion()` returned: `month, ng_spot, urea, dap, storage_mmcf`. `engineer.py` reads it directly via SQLAlchemy and reindexes to a contiguous month-start range.

The reason for the split: lineage. The merge logic and the source-precedence rule are now declarative SQL with tests, and any cell in the mart can be traced back to the raw table that produced it. Python-side feature engineering and modelling did not have to change shape.

## 3. File-by-file

### New

- **`backend/src/ingestion/raw_sources.py`** — Six fetchers, one per native-grain source. Each returns a DataFrame with the source's natural columns plus `source` / `source_file` / `loaded_at`. The EIA helpers paginate the v2 endpoint with `length=5000` ascending by period to pull the full series.
- **`backend/src/ingestion/postgres_loader.py`** — Reads `DATABASE_URL`, creates schema `raw` if needed, calls each fetcher, and writes the resulting frame with `to_sql(..., if_exists="replace", method="multi")`. Probes the connection up front and refuses to write an empty table. `--only <names>` runs a subset.
- **`foregast_dbt/`** — The dbt project. `dbt_project.yml` materialises `staging` as views and `marts` as tables, with `+schema` suffixes that combine with the profile's `schema=dev` to produce `dev_staging` and `dev_marts`. `_sources.yml` declares the six raw tables; `stg_*.sql` are the staging views; `mart_commodity_panel_monthly.sql` is the panel; `_staging.yml` and `_marts.yml` carry the tests.
- **`scripts/validate_mart_parity.py`** — Parity gate between the mart and `run_ingestion()`. See guardrails below.
- **`scripts/validate_engineer_mart_swap.py`** — Companion validator used while flipping engineer's default to the mart (untracked; present alongside the parity script).

### Modified

- **`backend/src/features/engineer.py`** — `build_features` now accepts `data=<dict>` (legacy passthrough), `source='mart'` (default, reads `dev_marts.mart_commodity_panel_monthly` via `DATABASE_URL`), or `source='pipeline'` (lazy-imports `run_ingestion` as an explicit fallback). `_load_panel_from_mart` reindexes to a contiguous MS range whose end is the latest month with any non-null value — slightly looser than `run_ingestion`'s end-of-shortest-series rule, which matters because NG can extend past the Pink Sheet's last month. The daily-derived MA block (`_load_daily_ng_features`) is untouched.
- **`backend/train_models.py`** — Added `_resolve_source()`: reads `FEATURE_SOURCE` (default `mart`), validates it, and falls back to `pipeline` with a logged warning if `DATABASE_URL` is unset. `forecaster.train` is imported lazily so `--dry-run` works without xgboost.
- **`requirements.txt`** — Added `sqlalchemy>=2.0.0`, `psycopg2-binary>=2.9.9`, `wandb>=0.15.0`.

### Key unchanged (and why it matters)

- **`backend/src/ingestion/pipeline.py`** — Still the source of truth for `source='pipeline'` and for `load_full_history()`, which `run_pipeline.py` uses to build the chart payload. The legacy path is intentionally live.
- **`backend/run_pipeline.py`** — Reads the parquet feature store, runs prediction / Monte Carlo / signal, and writes `cache.json`. None of this had to change because the feature store shape didn't change.
- **`backend/src/features/engineer.py` daily-MA path** — `_load_daily_ng_features()` still reads `data/RNGWHHDd_..._30_day_moving.xls` directly. That XLS is also landed as `raw.henry_hub_spot_daily_xls` for future use, but the feature builder hasn't been switched over.
- **`backend/src/models/forecaster.py`, `simulation/monte_carlo.py`, `signals/engine.py`** — Untouched. The model artifacts in `backend/src/models/` and the metadata they produce are byte-identical when mart parity holds.
- **`app.py`, `pages/login.py`, `pages/main_app.py`** — Streamlit UI reads `data/processed/cache.json` and blocks on startup until that file exists. No code change needed; the cache contract is unchanged.

## 4. Configuration & switches

- **`DATABASE_URL`** — SQLAlchemy URL used by both the loader and the mart-reading branch of engineer. Current dev value in `.env`: `postgresql+psycopg2://postgres:devpass@localhost:5432/foregast`. The `.env.example` checked into the repo does **not** include this key yet — it predates the migration.
- **`FEATURE_SOURCE`** — Read by `train_models._resolve_source()`. `mart` (default) → engineer reads the mart. `pipeline` → engineer calls `run_ingestion()`. Anything else raises. If `mart` is requested but `DATABASE_URL` is empty, the resolver logs a warning and silently downgrades to `pipeline` so a Postgres-less deploy still trains.
- **`data=<dict>` legacy branch in `build_features`** — When a caller passes `data` explicitly, neither the mart nor the pipeline path runs; the dict is used as-is. This is how anything still constructing the panel by hand (notebooks, validation scripts) plugs in.
- **dbt profile location** — `~/.dbt/profiles.yml`, target `dev`, schema `dev`. Combined with the model-level `+schema: staging|marts` in `dbt_project.yml`, the live schemas are `dev_staging` and `dev_marts`.
- **Working dbt binary** — `.venv/Scripts/dbt.exe` (the user-site install is an unsupported Fusion alpha and is not on PATH).
- **Flipping between mart and pipeline** — Set `FEATURE_SOURCE=pipeline` (or just clear `DATABASE_URL`) before invoking `train_models.py` or `refresh.py`. No code change needed.

## 5. Guardrails

**`scripts/validate_mart_parity.py`** — Reads the mart and calls `run_ingestion()` over the mart's full window, then compares row-for-row across the four panel columns. NaN/NaN is treated as matching; otherwise the absolute diff must be ≤ `1e-6`. Index alignment is reported separately (months only in mart vs only in pipeline). Exit 0 means parity.

The script has a narrow, opt-in whitelist behind `--accept-eia-precision-gap` that only relaxes `ng_spot`:

1. **Sub-cent rounding**, any month: `|diff| < 0.01`. The legacy `_fetch_eia_ng_spot` fetches `length=5000` rows without pagination, so for earlier months it falls back to the 2-decimal local CSV while the mart aggregates the full-precision EIA daily mean.
2. **2006-08 truncation boundary**: the 5000-row cap lands at 2006-08-07, so the legacy monthly mean for August 2006 averages only days 7–31 (the dropped first six days were ~$8.09, well above the rest of the month). Mart uses all 23 trading days and is authoritative; criterion is `month = 2006-08-01 AND |diff| < 0.25`.

Anything outside those two classes is reported as a real diff and fails the check. The whitelist is not a silencer — diffs are still printed.

**dbt tests** (run via `dbt test`):

- **Sources.** `not_null` on `period` for every raw table; `not_null` on the value column for the five non-Pink-Sheet sources. The Pink Sheet's `urea_usd_per_mt` / `dap_usd_per_mt` are left nullable because the spreadsheet has legitimate NaNs in older history.
- **Staging.** `not_null + unique` on `month` and `not_null` on the value column for all four `stg_*` models.
- **Mart.** `not_null + unique` on `month`. The four value columns are intentionally nullable (NG predates urea, storage predates NG, Pink Sheet ends before NG's latest month).

## 6. Deferred / open items

**Production networking + source decision.** `DATABASE_URL` currently points at `localhost:5432`. That works from the host machine but does not reach Postgres from inside a container — `docker-compose.yml` doesn't define a Postgres service, only the `agrisignal` Streamlit container, so a deployed `refresh.py` running inside that container has no resolvable `localhost` database. Two paths, to be picked at deploy time, not patched in code:

1. **Keep the mart path.** Run Postgres as a sibling service (compose service name, e.g. `db:5432`) or a managed instance (RDS, Cloud SQL), and set `DATABASE_URL` to that address in the container's env. The loader + dbt + mart-reading engineer all work unchanged.
2. **Skip the mart in prod.** Set `FEATURE_SOURCE=pipeline` (or leave `DATABASE_URL` empty) in the deployed environment. `train_models` will log the fallback and use the legacy `run_ingestion()` directly. The mart still exists for dev / analysis, but production never reads it.

Either is fine; the choice is purely operational. The fall-back-on-missing-`DATABASE_URL` behaviour in `_resolve_source` is the safety net that makes option 2 work without code changes.
