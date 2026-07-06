"""Shared fixtures + sys.path setup for the foregast test suite.

The backend runs with `backend/` on the import path (routes do
`from api.cache_store import ...`, engineer does `from src.ingestion...`),
so we mirror that here rather than restructuring backend into a package.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

_TESTS = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.normpath(os.path.join(_TESTS, "..", "backend"))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


@pytest.fixture
def mc_metadata() -> dict:
    """Minimal metadata dict — just the residual_{mean,std}_t{1,2,3} keys MC reads."""
    return {
        "residual_mean_t1": 0.0, "residual_std_t1": 25.0,
        "residual_mean_t2": 0.0, "residual_std_t2": 40.0,
        "residual_mean_t3": 0.0, "residual_std_t3": 55.0,
    }


@pytest.fixture
def mc_forecast() -> dict:
    return {
        "current": 500.0,
        "t1": 505.0, "t2": 510.0, "t3": 515.0,
        "ng_current": 3.20,
    }


@pytest.fixture
def synthetic_panel() -> dict:
    """36 months of nudged data. Long enough for rolling(24) and shift(12) to
    yield some non-NaN rows, short enough to keep tests cheap. Passed as
    build_features(data=<dict>) so no DB / XLS is touched."""
    idx = pd.date_range("2020-01-01", periods=36, freq="MS")
    rng = np.random.default_rng(0)
    return {
        "ng_spot":      pd.Series(3.0   + rng.normal(0, 0.4, len(idx)).cumsum() * 0.05, index=idx),
        "urea":         pd.Series(400.0 + rng.normal(0,  20, len(idx)).cumsum() * 0.10, index=idx),
        "dap":          pd.Series(600.0 + rng.normal(0,  15, len(idx)).cumsum() * 0.10, index=idx),
        "storage_mmcf": pd.Series(2500.0+ rng.normal(0, 100, len(idx)).cumsum() * 0.20, index=idx),
    }


@pytest.fixture
def cache_fixture() -> dict:
    """Cache.json-shaped payload that satisfies every response model. Uses
    camelCase for the signal fields the schema aliases (currentPrice etc.),
    matching what run_pipeline.py actually writes on disk."""
    return {
        "generated_at": datetime(2026, 7, 1, tzinfo=timezone.utc).isoformat(),
        "as_of_date":   "2026-06",
        "urea_history": {
            "labels": ["May 2026", "Jun 2026"],
            "values": [640.0, 655.0],
        },
        "natgas_history": {
            "labels": ["May 2026", "Jun 2026"],
            "values": [3.10, 3.25],
        },
        "forecast": {
            "labels": ["Jul 2026", "Aug 2026", "Sep 2026"],
            "mean":   [660.0, 670.0, 675.0],
            "low":    [620.0, 615.0, 610.0],
            "high":   [700.0, 725.0, 740.0],
        },
        "signal": {
            "signal": "NEUTRAL", "urgency": "LOW",
            "recommendation": "Neutral", "rationale": "test",
            "key_driver": "test", "forecast_summary": "test",
            "confidence": 0.55,
            "currentPrice": 655.0, "bestMonth": "In 2 months", "bestPrice": 670.0,
            "forecast_t1": 660.0, "forecast_t2": 670.0, "forecast_t3": 675.0,
            "p10_t2": 615.0, "p90_t2": 725.0,
            "prob_rising": 0.55,
            "ng_current": 3.25, "ng_change_30d": 0.05,
        },
        "monte_carlo": {
            "n_simulations": 10000,
            "p10_t1": 630.0, "p25_t1": 645.0, "p50_t1": 660.0, "p75_t1": 675.0, "p90_t1": 690.0,
            "p10_t2": 615.0, "p25_t2": 640.0, "p50_t2": 670.0, "p75_t2": 700.0, "p90_t2": 725.0,
            "p10_t3": 610.0, "p50_t3": 675.0, "p90_t3": 740.0,
            "prob_rising_t2":         0.55,
            "prob_10pct_increase_t2": 0.10,
            "prob_20pct_increase_t2": 0.02,
        },
        "sim_t2_distribution": [650.0, 660.0, 670.0, 680.0, 690.0],
        "model_metadata": {
            "training_window_start": "1997-01",
            "training_window_end":   "2021-12",
            "test_rmse_t1": 25.0, "test_mae_t1": 20.0, "test_dir_acc_t1": 0.55, "test_clf_acc_t1": 0.55,
            "residual_mean_t1": 0.0, "residual_std_t1": 25.0,
            "test_rmse_t2": 40.0, "test_mae_t2": 30.0, "test_dir_acc_t2": 0.55, "test_clf_acc_t2": 0.55,
            "residual_mean_t2": 0.0, "residual_std_t2": 40.0,
            "test_rmse_t3": 55.0, "test_mae_t3": 45.0, "test_dir_acc_t3": 0.55, "test_clf_acc_t3": 0.55,
            "residual_mean_t3": 0.0, "residual_std_t3": 55.0,
        },
    }


@pytest.fixture
def fake_cache_store(cache_fixture):
    """In-memory CacheStore. Substituted via app.dependency_overrides."""
    class _FakeStore:
        def __init__(self, payload: dict) -> None:
            self._payload = payload
        def read(self) -> dict:
            return self._payload
        def mtime(self) -> datetime:
            return datetime(2026, 7, 1, tzinfo=timezone.utc)
    return _FakeStore(cache_fixture)
