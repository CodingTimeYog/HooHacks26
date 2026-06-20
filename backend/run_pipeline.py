"""
run_pipeline.py — Loads trained models, generates forecast + Monte Carlo + signal,
and writes data/processed/cache.json for the Express server to serve.

Run this after train_models.py. Re-run any time you want fresh forecasts.

Usage (from project root):
    python backend/run_pipeline.py
"""

import sys
import os
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

_THIS = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.normpath(os.path.join(_THIS, ".."))

from src.ingestion.pipeline import run_ingestion, load_full_history
from src.features.engineer import build_features, load_feature_store, FEATURE_COLS
from src.models.forecaster import load_models, predict
from src.simulation.monte_carlo import run_monte_carlo
from src.signals.engine import generate_signal

print("=== Gas Forecast Pipeline ===\n")

# ── 1. Load trained models
print("1/5 — Loading models...")
models, metadata = load_models()

# ── 2. Load feature store (built during training)
print("2/5 — Loading feature store...")
fs = load_feature_store()

# Use the last row that has complete features (may not have targets — that's fine)
feature_rows = fs.dropna(subset=FEATURE_COLS)
latest_row   = feature_rows.iloc[[-1]]   # single-row DataFrame
as_of_date   = latest_row.index[0]

print(f"     As-of date: {as_of_date.strftime('%Y-%m')}")

current_urea = float(fs.loc[as_of_date, "urea"])
current_ng   = float(fs.loc[as_of_date, "ng_spot"])

# 30-day nat gas change (1-month momentum)
ng_change_30d = float(fs.loc[as_of_date, "ng_mom_1m"])

# ── 3. Point forecast
print("3/5 — Running XGBoost forecast...")
point = predict(models, latest_row)
pct_change_t2 = (point["t2"] - current_urea) / current_urea

forecast = {
    "current":       current_urea,
    "t1":            point["t1"],
    "t2":            point["t2"],
    "t3":            point["t3"],
    "pct_change_t2": pct_change_t2,
    "ng_current":    current_ng,
}
print(f"     Current urea: ${current_urea:.0f}/mt")
print(f"     Forecast t1:  ${point['t1']:.0f}  t2: ${point['t2']:.0f}  t3: ${point['t3']:.0f}")

# ── 4. Monte Carlo
print("4/5 — Running Monte Carlo (10,000 paths)...")
mc = run_monte_carlo(forecast, metadata)
print(f"     t2 80% band: ${mc['p10_t2']:.0f} – ${mc['p90_t2']:.0f}/mt")
print(f"     Prob rising: {mc['prob_rising_t2']*100:.0f}%")

# ── 5. Signal
print("5/5 — Generating signal...")
signal = generate_signal(forecast, mc, ng_change_30d)
print(f"     Signal: {signal['signal']}  ({signal['urgency']})")

# ── Build forecast months labels
import pandas as pd
t1_date = as_of_date + pd.DateOffset(months=1)
t2_date = as_of_date + pd.DateOffset(months=2)
t3_date = as_of_date + pd.DateOffset(months=3)
fmt = "%b %Y"

# ── Historical chart payload (2018 onwards, aligned where both series exist)
print("\nBuilding history payload...")
history = load_full_history()
ng_hist   = history["ng_spot"]
urea_hist = history["urea"]

# Align to common index where both have data
common_idx = ng_hist.dropna().index.intersection(urea_hist.dropna().index)
common_idx = common_idx[common_idx >= "2018-01-01"]

# ── Assemble cache
cache = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "as_of_date":   as_of_date.strftime("%Y-%m"),

    # GET /api/prices/history?commodity=urea
    "urea_history": {
        "labels": [d.strftime(fmt) for d in common_idx],
        "values": [round(float(urea_hist[d]), 2) for d in common_idx],
    },

    # GET /api/prices/natgas/history
    "natgas_history": {
        "labels": [d.strftime(fmt) for d in common_idx],
        "values": [round(float(ng_hist[d]), 3) for d in common_idx],
    },

    # GET /api/prices/forecast
    "forecast": {
        "labels": [t1_date.strftime(fmt), t2_date.strftime(fmt), t3_date.strftime(fmt)],
        "mean":   [round(mc["p50_t1"], 1), round(mc["p50_t2"], 1), round(mc["p50_t3"], 1)],
        "low":    [round(mc["p10_t1"], 1), round(mc["p10_t2"], 1), round(mc["p10_t3"], 1)],
        "high":   [round(mc["p90_t1"], 1), round(mc["p90_t2"], 1), round(mc["p90_t3"], 1)],
    },

    # GET /api/signal/buy-timing
    "signal": signal,

    # Full MC result for /api/monte-carlo if needed later
    "monte_carlo": {k: v for k, v in mc.items() if k != "sim_t2_distribution"},
    "sim_t2_distribution": mc["sim_t2_distribution"],

    # Model quality info
    "model_metadata": {
        k: v for k, v in metadata.items()
        if k not in ("feature_cols",)
    },
}

out_path = os.path.join(ROOT, "data", "processed", "cache.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(cache, f, indent=2, ensure_ascii=False)

print(f"\nCache written -> {out_path}")
print("=== Done. Fresh data is live — no server restart needed. ===")
