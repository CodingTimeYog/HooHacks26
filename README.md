# foreGASt — Fertilizer Price Forecasting

Forecasting urea (and DAP) fertilizer prices 1–3 months ahead from natural-gas market signals, served as a probabilistic buy/wait recommendation for farmers. Built end-to-end: a versioned data pipeline, a dbt-modeled analytics mart, an XGBoost + Monte Carlo forecasting layer, and a FastAPI inference service behind a Streamlit app.

> Thesis: natural gas is the dominant feedstock for nitrogen fertilizer, and Henry Hub price movements lead urea by roughly 4–6 months. foreGASt turns that lead-lag relationship into a calibrated forecast with explicit uncertainty.

*(Originally prototyped at HooHacks 2026, where it won the Finance track; substantially re-engineered since.)*

---

## What it does

- Pulls natural-gas spot and storage data (EIA API, 1997–present) and monthly fertilizer prices (World Bank Pink Sheet).
- Models them into a clean monthly commodity panel with **dbt** on **Postgres**.
- Trains horizon-specific **XGBoost** regressors and direction classifiers (t+1 / t+2 / t+3 months).
- Runs a **10,000-path Monte Carlo** simulation to produce P10/P50/P90 price bands and a probability of rising.
- Emits a rule-based **BUY / WAIT** signal with a confidence score.
- Serves the result through a typed **FastAPI** endpoint and a **Streamlit** dashboard.

## Architecture

```
EIA API ─┐
         ├─► raw (Postgres) ─► dbt staging (views) ─► mart_commodity_panel_monthly ─┐
World Bank ┘                                                                         │
                                                                                     ▼
                                                          feature engineering (engineer.py)
                                                                                     │
                                                                                     ▼
                                              XGBoost regressors + classifiers (t1/t2/t3)
                                                                                     │
                                                                                     ▼
                                           Monte Carlo (P10/P50/P90) + BUY/WAIT signal
                                                                                     │
                                                          run_pipeline.py writes cache.json
                                                                                     │
                                                          ┌──────────────────────────┴───────────┐
                                                          ▼                                       ▼
                                              FastAPI service (/forecast/urea)          Streamlit app
```

Data lineage is enforced by a **parity gate** (`scripts/validate_mart_parity.py`) that checks the dbt mart against the legacy pipeline cell-for-cell before any model trains. Every data feed — staging, daily moving-average features, fertilizer prices, and the history payload — is sourced from the same dbt mart, with local-file fallbacks when the database is unavailable.

## Methodology notes

- **Validation:** walk-forward cross-validation (`TimeSeriesSplit`, 5 folds, 1-month gap) for model selection, with recency-weighted training.
- **Uncertainty:** forecasts are distributions, not point estimates — the Monte Carlo band and probability of rising are first-class outputs, and the buy/wait signal is derived from them.
- **Honesty about regime breaks:** in early 2026 a real fertilizer supply shock (Strait of Hormuz disruption) pushed urea to multi-year highs. The model's error widens through that window because it's a genuine out-of-distribution event, not a model defect — the forecast confidence should be discounted during such breaks. This is surfaced rather than hidden.

## Results

> Reported on a held-out test set in the pre-shock (stationary) price regime:
> - 90-day direction accuracy: **~XX%** *(pin exact figure before publishing)*
> - 90-day price RMSE: **~$XX/mt** *(pre-shock holdout; error is higher across the 2026 supply-shock window by design)*

*(Replace the above with your final temporal-holdout numbers once locked.)*

## Tech stack

Python · Postgres · dbt · XGBoost · pandas/NumPy · FastAPI · Pydantic v2 · Streamlit · Docker · Weights & Biases (experiment tracking)

## Running it

### Prerequisites
- Docker Desktop, or Python 3.12 with a virtual environment
- An [EIA API key](https://www.eia.gov/opendata/) (free)

### Quick start (Docker)
```bash
cp .env.example .env          # then set EIA_API_KEY=your_key_here
docker-compose up --build
```
Open the app at http://localhost:8501.

### Local (Python)
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python backend/train_models.py     # train the XGBoost models (run once)
python backend/run_pipeline.py     # ingest data + generate forecasts → cache.json
streamlit run app.py               # launch the dashboard
```

### Inference API
```bash
docker build -f Dockerfile.api -t foregast-api .
docker run -p 8000:8000 -v "$(pwd)/data/processed:/data/processed:ro" foregast-api
curl http://localhost:8000/forecast/urea
curl http://localhost:8000/health
```
The API is a read-only consumer of the forecast cache, designed for a serverless swap (local file → S3) behind API Gateway + Lambda.

## Repository layout

```
backend/
  api/            FastAPI service (routes, Pydantic schemas, cache store)
  src/            ingestion, feature engineering, models, signal logic
  train_models.py, run_pipeline.py
foregast_dbt/     dbt project (staging views + commodity-panel mart)
scripts/          mart-parity validation gate
pages/, app.py    Streamlit UI
docs/             architecture
```

## Roadmap

- Cloud deployment: containerized API on AWS Lambda + API Gateway, scheduled refresh via EventBridge → Fargate, forecast cache on S3.
- Temporal-holdout reporting and a soft data-quality alert on anomalous month-over-month moves.

## License & data

Data from the U.S. EIA (public domain) and the World Bank Commodity Markets "Pink Sheet" (publicly available). This project is a personal portfolio piece and is not investment advice.
