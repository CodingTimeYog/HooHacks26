"""Regression guard for the chronological-holdout leakage bug fixed this
session. The split logic inside forecaster.train() is duplicated here so we
can assert its invariants without needing to run the full XGBoost training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.forecaster import HOLDOUT_START, SHOCK_TARGET_START


@pytest.fixture
def feature_index() -> pd.DatetimeIndex:
    """Long enough to span well before HOLDOUT_START (2022-01) and past
    SHOCK_TARGET_START (2026-01) so both windows are non-empty."""
    return pd.date_range("2000-01-01", "2026-06-01", freq="MS")


def _train_test_masks(index: pd.DatetimeIndex, holdout_start: pd.Timestamp, horizon: int):
    """Byte-for-byte replica of the split in forecaster.train():
        target_month = X.index + DateOffset(months=horizon)
        train_mask   = target_month < holdout_start
        test_mask    = X.index >= holdout_start
    """
    target_month = index + pd.DateOffset(months=horizon)
    return np.asarray(target_month < holdout_start), np.asarray(index >= holdout_start)


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_train_end_strictly_precedes_holdout_start(feature_index, horizon):
    """No training feature-row date may sit at or beyond the earliest holdout
    date — otherwise the two windows overlap and evaluation leaks."""
    holdout_start = pd.Timestamp(HOLDOUT_START)
    train_mask, test_mask = _train_test_masks(feature_index, holdout_start, horizon)

    train_dates = feature_index[train_mask]
    test_dates  = feature_index[test_mask]

    assert len(train_dates) > 0, "sanity: some training rows must remain after purge"
    assert len(test_dates)  > 0, "sanity: some holdout rows must exist"
    assert train_dates.max() < test_dates.min(), (
        f"train/test overlap: latest train {train_dates.max()}, "
        f"earliest test {test_dates.min()} at horizon={horizon}"
    )


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_no_training_targets_land_in_holdout(feature_index, horizon):
    """The per-horizon purge exists so no training row's forward target month
    (index + horizon) crosses into the holdout — otherwise the model trains
    on holdout-period urea prices."""
    holdout_start = pd.Timestamp(HOLDOUT_START)
    train_mask, _ = _train_test_masks(feature_index, holdout_start, horizon)

    train_dates = feature_index[train_mask]
    max_train_target = (train_dates + pd.DateOffset(months=horizon)).max()

    assert max_train_target < holdout_start, (
        f"training target month {max_train_target} lands in holdout window "
        f"(HOLDOUT_START={holdout_start.date()}) at horizon={horizon}"
    )


def test_shock_window_lies_within_or_after_holdout():
    """The shock window is a subset of / after the holdout by design; a
    regression that reordered these dates would silently misclassify shock
    residuals as pre-shock (or vice versa)."""
    assert pd.Timestamp(HOLDOUT_START) <= pd.Timestamp(SHOCK_TARGET_START)
