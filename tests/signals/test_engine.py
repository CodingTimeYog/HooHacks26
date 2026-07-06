"""Signal-engine tests — pin down every branch and every threshold.

Thresholds in engine.py (all STRICT `>` / `<`):
  BUY_NOW         : pct > 0.08 AND prob > 0.65
  CONSIDER_BUYING : pct > 0.04 OR  prob > 0.55
  WAIT            : pct < -0.04 AND prob < 0.40
  NEUTRAL         : else
"""

from __future__ import annotations

import pytest

from src.signals.engine import generate_signal


def _mc(prob_rising_t2: float, p50_t2: float = 650.0) -> dict:
    return {
        "p50_t1": 640.0, "p50_t2": p50_t2, "p50_t3": 660.0,
        "p10_t2": 600.0, "p90_t2": 700.0,
        "prob_rising_t2": prob_rising_t2,
    }


def _forecast(pct_change_t2: float, current: float = 600.0) -> dict:
    return {
        "current": current,
        "t1": current, "t2": current * (1 + pct_change_t2), "t3": current,
        "pct_change_t2": pct_change_t2,
        "ng_current": 3.20,
    }


BRANCH_CASES = [
    # id,                     pct,    prob,  signal,             urgency
    ("buy_now_clear",         0.15,   0.80,  "BUY_NOW",          "HIGH"),
    ("consider_pct_arm",      0.05,   0.30,  "CONSIDER_BUYING",  "MODERATE"),
    ("consider_prob_arm",     0.02,   0.60,  "CONSIDER_BUYING",  "MODERATE"),
    ("wait_clear",           -0.10,   0.20,  "WAIT",             "LOW"),
    ("neutral_middle",        0.02,   0.50,  "NEUTRAL",          "LOW"),
]


@pytest.mark.parametrize(
    "pct,prob,expected_signal,expected_urgency",
    [(c[1], c[2], c[3], c[4]) for c in BRANCH_CASES],
    ids=[c[0] for c in BRANCH_CASES],
)
def test_signal_branches(pct, prob, expected_signal, expected_urgency):
    r = generate_signal(_forecast(pct), _mc(prob), ng_change_30d=0.05)
    assert r["signal"]  == expected_signal
    assert r["urgency"] == expected_urgency


BOUNDARY_CASES = [
    # Every threshold is STRICT — a value exactly on the boundary must NOT
    # trigger the stricter branch. Documented in engine.py.
    #  id,                     pct,     prob,  signal-that-should-NOT-fire → falls to
    ("pct_exactly_0.08",       0.08,    0.80,  "CONSIDER_BUYING"),   # BUY_NOW pct arm
    ("prob_exactly_0.65",      0.12,    0.65,  "CONSIDER_BUYING"),   # BUY_NOW prob arm
    ("pct_exactly_0.04",       0.04,    0.30,  "NEUTRAL"),           # CONSIDER pct arm
    ("prob_exactly_0.55",      0.03,    0.55,  "NEUTRAL"),           # CONSIDER prob arm
    ("pct_exactly_neg_0.04",  -0.04,    0.20,  "NEUTRAL"),           # WAIT pct arm
    ("prob_exactly_0.40",     -0.10,    0.40,  "NEUTRAL"),           # WAIT prob arm
]


@pytest.mark.parametrize(
    "pct,prob,expected_signal",
    [(c[1], c[2], c[3]) for c in BOUNDARY_CASES],
    ids=[c[0] for c in BOUNDARY_CASES],
)
def test_signal_thresholds_are_strict(pct, prob, expected_signal):
    r = generate_signal(_forecast(pct), _mc(prob), ng_change_30d=0.05)
    assert r["signal"] == expected_signal


def test_wait_confidence_is_inverted():
    """confidence = 1 - prob_rising for WAIT (spec: probability the call is right)."""
    r = generate_signal(_forecast(-0.10), _mc(0.20), ng_change_30d=0.05)
    assert r["signal"] == "WAIT"
    assert r["confidence"] == pytest.approx(0.80, abs=0.005)


def test_non_wait_confidence_equals_prob_rising():
    r = generate_signal(_forecast(0.05), _mc(0.72), ng_change_30d=0.05)
    assert r["signal"] != "WAIT"
    assert r["confidence"] == pytest.approx(0.72, abs=0.005)
