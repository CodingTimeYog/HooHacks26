"""
analyze_lag.py — Empirically verify the NG → urea price lag.

Runs:
  1. Cross-Correlation Function (CCF): correlation between urea[t] and ng[t-k]
     at lags k = 0..12 months. Peak lag = how far back NG best predicts urea.
  2. Granger Causality: formal test of whether past NG values contain
     predictive information about urea beyond urea's own history.

Usage (from project root):
    python scripts/analyze_lag.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend'))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from src.ingestion.pipeline import run_ingestion

# ── Load data (full 1997–2024 history) ────────────────────────────────────────
print("Loading data (1997–2024)...")
data = run_ingestion(start="1997-01", end="2024-12")

ng   = data["ng_spot"].dropna()
urea = data["urea"].dropna()

# Align to common index
idx  = ng.index.intersection(urea.index)
ng   = ng[idx]
urea = urea[idx]

print(f"  {len(idx)} monthly observations aligned\n")

# ── 1. CROSS-CORRELATION FUNCTION ─────────────────────────────────────────────
# corr(urea[t], ng[t-k]) for k = 0..12
# A high correlation at lag k means NG prices k months ago are a good
# predictor of urea prices today.

print("=" * 60)
print("CROSS-CORRELATION: urea[t] vs ng[t - lag]")
print("=" * 60)
print(f"  {'Lag (months)':<16} {'Correlation':>12}  {'Signal'}")
print(f"  {'-'*16} {'-'*12}  {'-'*20}")

correlations = {}
urea_arr = urea.values
ng_arr   = ng.values

for lag in range(0, 13):
    if lag == 0:
        corr = float(np.corrcoef(urea_arr, ng_arr)[0, 1])
    else:
        # urea[t] vs ng[t - lag]: drop first `lag` urea values, last `lag` ng values
        corr = float(np.corrcoef(urea_arr[lag:], ng_arr[:-lag])[0, 1])
    correlations[lag] = corr
    bar = "█" * int(abs(corr) * 20)
    print(f"  Lag {lag:<12} {corr:>+.4f}      {bar}")

peak_lag = max(correlations, key=lambda k: correlations[k])
print(f"\n  >>> Peak correlation at lag {peak_lag} months  (r = {correlations[peak_lag]:+.4f})")
print("      Current features cover lags 1–4.")

if peak_lag <= 4:
    print("      RESULT: Feature lags 1–4 adequately capture the peak.")
else:
    print(f"      RESULT: Peak lag {peak_lag} is outside current feature window.")
    print(f"              Recommend adding ng_lag5 through ng_lag{peak_lag}.")

# ── 2. GRANGER CAUSALITY TEST ─────────────────────────────────────────────────
# H0: NG prices do NOT Granger-cause urea prices (i.e. NG adds no predictive value)
# If p < 0.05 at lag k, we reject H0 — NG at lag k genuinely helps predict urea.

print(f"\n{'=' * 60}")
print("GRANGER CAUSALITY: does ng Granger-cause urea?")
print("H0: NG adds no predictive value for urea beyond urea's own history")
print(f"{'=' * 60}")
print(f"  {'Lag':<6} {'F-stat':>10} {'p-value':>10}  {'Reject H0?'}")
print(f"  {'-'*6} {'-'*10} {'-'*10}  {'-'*10}")

df_gc = pd.DataFrame({"urea": urea.values, "ng": ng.values})

try:
    gc_results = grangercausalitytests(df_gc[["urea", "ng"]], maxlag=12, verbose=False)
    significant_lags = []
    for lag in range(1, 13):
        # ssr_ftest is the standard F-test result
        f_stat = gc_results[lag][0]["ssr_ftest"][0]
        p_val  = gc_results[lag][0]["ssr_ftest"][1]
        reject = "YES ***" if p_val < 0.05 else ("marginal" if p_val < 0.10 else "no")
        if p_val < 0.05:
            significant_lags.append(lag)
        print(f"  Lag {lag:<3}  {f_stat:>10.3f} {p_val:>10.4f}  {reject}")

    print(f"\n  >>> Significant at p<0.05: lags {significant_lags}")
    if significant_lags:
        max_sig = max(significant_lags)
        print(f"      NG Granger-causes urea up to lag {max_sig} months.")
        if max_sig > 4:
            print(f"      Recommend extending features to ng_lag{max_sig}.")
        else:
            print("      Current feature lags (1–4) cover the significant range.")
    else:
        print("      No significant Granger causality found (unusual — check data).")
except Exception as e:
    print(f"  Granger test error: {e}")

# ── 3. ROLLING CORRELATION (stability check) ──────────────────────────────────
# Check whether the lag relationship is stable over time, or only holds in
# certain periods. Use a 36-month rolling window at the peak lag.

print(f"\n{'=' * 60}")
print(f"ROLLING CORRELATION at lag {peak_lag}m (36-month window)")
print("Checks whether the NG→urea relationship is stable over time.")
print(f"{'=' * 60}")

urea_vals = urea.reset_index(drop=True)
ng_lagged = ng.shift(peak_lag).reset_index(drop=True)
roll_corr = pd.Series([
    urea_vals.iloc[i:i+36].corr(ng_lagged.iloc[i:i+36])
    for i in range(0, len(urea_vals) - 36, 12)
])

years = [urea.index[i].year for i in range(0, len(urea_vals) - 36, 12)]

print(f"  {'Period':<12} {'Rolling r':>10}  {'Strength'}")
print(f"  {'-'*12} {'-'*10}  {'-'*20}")
for yr, r in zip(years, roll_corr):
    if pd.isna(r):
        continue
    bar = "█" * int(abs(r) * 20)
    strength = "Strong" if abs(r) > 0.7 else ("Moderate" if abs(r) > 0.4 else "Weak")
    print(f"  {yr}–{yr+2:<8}  {r:>+.4f}      {bar}  {strength}")

print(f"\n  Mean rolling r: {roll_corr.mean():.4f}")
print(f"  Min  rolling r: {roll_corr.min():.4f}  (weakest period)")
print(f"  Max  rolling r: {roll_corr.max():.4f}  (strongest period)")

print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"  Peak CCF lag         : {peak_lag} months")
print("  Current feature lags : 1–4 months")
if peak_lag > 4:
    print(f"  ACTION NEEDED        : Add ng_lag5 through ng_lag{peak_lag} to engineer.py")
else:
    print("  ACTION NEEDED        : None — current lags cover peak")
