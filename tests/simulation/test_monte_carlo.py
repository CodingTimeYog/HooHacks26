"""Monte Carlo simulator tests — determinism, echoed sim count, and monotone
percentiles across a few input distributions (including zero volatility)."""

from __future__ import annotations

import pytest

from src.simulation.monte_carlo import run_monte_carlo


def test_determinism_same_seed_bit_identical_output(mc_forecast, mc_metadata):
    a = run_monte_carlo(mc_forecast, mc_metadata, n_simulations=5000, random_seed=42)
    b = run_monte_carlo(mc_forecast, mc_metadata, n_simulations=5000, random_seed=42)
    assert a.keys() == b.keys()
    for key in a:
        assert a[key] == b[key], f"determinism broken on key: {key}"


def test_n_simulations_field_matches_request(mc_forecast, mc_metadata):
    for n in (1000, 5000, 12345):
        r = run_monte_carlo(mc_forecast, mc_metadata, n_simulations=n, random_seed=42)
        assert r["n_simulations"] == n


@pytest.mark.parametrize(
    "std_scale", [1.0, 0.0, 3.0], ids=["typical", "zero_vol", "high_vol"],
)
def test_percentile_ordering_holds_for_all_horizons(mc_forecast, mc_metadata, std_scale):
    """p10 <= p25 <= p50 <= p75 <= p90 for t1/t2 (t3 only exposes p10/p50/p90).
    Zero-vol edge case pins that a degenerate distribution doesn't accidentally
    invert order via floating-point noise."""
    scaled = {
        k: (v * std_scale if k.startswith("residual_std") else v)
        for k, v in mc_metadata.items()
    }
    r = run_monte_carlo(mc_forecast, scaled, n_simulations=5000, random_seed=42)

    assert r["p10_t1"] <= r["p25_t1"] <= r["p50_t1"] <= r["p75_t1"] <= r["p90_t1"], \
        f"t1 percentile order broken at std_scale={std_scale}"
    assert r["p10_t2"] <= r["p25_t2"] <= r["p50_t2"] <= r["p75_t2"] <= r["p90_t2"], \
        f"t2 percentile order broken at std_scale={std_scale}"
    assert r["p10_t3"] <= r["p50_t3"] <= r["p90_t3"], \
        f"t3 percentile order broken at std_scale={std_scale}"
