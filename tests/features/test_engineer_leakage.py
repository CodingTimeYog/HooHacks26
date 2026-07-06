"""Feature-engineering leakage tests — the highest-value tests in the suite.

The mechanism: build the feature store from a synthetic panel, then perturb a
FUTURE row of one of the input series. Any feature at an EARLIER row whose
value changes under that perturbation is by definition using future data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.engineer import build_features, FEATURE_COLS


def _force_empty_daily_features(monkeypatch) -> None:
    """The daily-NG loader reads from Postgres or the local XLS — both external.
    Force the empty-frame fallback so build_features is a pure function of
    the synthetic panel dict we pass in."""
    monkeypatch.setattr(
        "src.features.engineer._load_daily_ng_features",
        lambda daily_source="auto": pd.DataFrame(),
    )


# Features intentionally excluded from the no-leak sweep. Empty today; if a
# known leak is ever accepted (unlikely — the point of this suite is to catch
# them), add its column name here and file a dedicated test for it.
LEAKY_FEATURES: set[str] = set()


def _perturb_last_row(panel: dict, series: str, delta: float) -> dict:
    """Return a deep-ish copy of the panel with the last row of one series shifted."""
    out = {k: v.copy() for k, v in panel.items()}
    out[series].iloc[-1] = out[series].iloc[-1] + delta
    return out


def _series_equal_nan_safe(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.isna() & b.isna()) | (a == b)


@pytest.mark.parametrize("series", ["ng_spot", "urea", "dap", "storage_mmcf"])
def test_no_feature_uses_future_data(monkeypatch, synthetic_panel, series):
    """For each input series: perturb the last row. Assert that every feature
    column's value at every earlier row is unchanged. This catches forward-
    looking bugs generically — no need to enumerate per-column shift signs."""
    _force_empty_daily_features(monkeypatch)

    baseline  = build_features(synthetic_panel).loc[:, FEATURE_COLS]
    perturbed = build_features(_perturb_last_row(synthetic_panel, series, 100.0)).loc[:, FEATURE_COLS]

    # Compare all rows before the perturbed last row.
    for col in FEATURE_COLS:
        if col in LEAKY_FEATURES:
            continue
        base_before = baseline[col].iloc[:-1]
        pert_before = perturbed[col].iloc[:-1]
        same = _series_equal_nan_safe(base_before, pert_before)
        assert same.all(), (
            f"leak: feature '{col}' at rows before the perturbed one changed "
            f"when the last-row '{series}' value was perturbed"
        )


def test_storage_zscore_uses_only_past_and_present(monkeypatch, synthetic_panel):
    """Regression guard for a fix landed alongside this test: storage_zscore
    used to normalise by df['storage_mmcf'].mean()/.std() over the full panel,
    leaking future storage values into every past row. Now uses expanding()
    stats up to and including row t."""
    _force_empty_daily_features(monkeypatch)
    baseline  = build_features(synthetic_panel)["storage_zscore"].iloc[:-1]
    perturbed = build_features(
        _perturb_last_row(synthetic_panel, "storage_mmcf", 1000.0)
    )["storage_zscore"].iloc[:-1]
    assert _series_equal_nan_safe(baseline, perturbed).all(), (
        "storage_zscore at earlier rows changed under a last-row perturbation "
        "of storage_mmcf — leakage has regressed"
    )


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_target_columns_are_forward_shifts(monkeypatch, synthetic_panel, horizon):
    """target_urea_tN at row t must equal urea at row t+N (shift(-N))."""
    _force_empty_daily_features(monkeypatch)
    df = build_features(synthetic_panel)
    urea = synthetic_panel["urea"]
    expected = urea.shift(-horizon)
    got = df[f"target_urea_t{horizon}"]
    assert _series_equal_nan_safe(got, expected).all(), (
        f"target_urea_t{horizon} does not equal urea.shift(-{horizon})"
    )


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_target_columns_have_exactly_h_trailing_nans(monkeypatch, synthetic_panel, horizon):
    _force_empty_daily_features(monkeypatch)
    df = build_features(synthetic_panel)
    col = df[f"target_urea_t{horizon}"]
    assert col.iloc[-horizon:].isna().all(),   f"{col.name} should be NaN in last {horizon} rows"
    assert col.iloc[:-horizon].notna().all(),  f"{col.name} should be non-NaN before last {horizon} rows"
