"""
Validate build_features(source='mart') against the existing parquet feature
store. In-memory only — does NOT overwrite data/processed/feature_store.parquet
and does NOT retrain models.

Reports column parity, row-range diffs (with trailing-NG-only row classification),
and per-column value diffs. Diffs are classified as:
  - EXPECTED: ng_spot-dependent column AND mismatch dates inside the propagation
    window of the known pre-2006 sub-cent rounding + the 2006-08 boundary cell.
  - UNEXPLAINED: anything else.
Exit 0 only if there are zero unexplained diffs and zero unexpected extra rows.
"""

import os
import sys

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.normpath(os.path.join(_THIS, ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from src.features.engineer import build_features, load_feature_store  # noqa: E402

TOL = 1e-6

# Columns whose values derive (directly or indirectly via lag / rolling / ratio)
# from the monthly ng_spot series. Pre-2006 sub-cent ng_spot diffs and the
# 2006-08 boundary cell propagate into these through ~12-month windows.
#
# NOT included (intentionally): ng_ma30_eom / ng_ma60_eom / ng_ma90_eom /
# ng_ma_cross_30_60 / ng_ma_cross_30_90 / ng_daily_vol_1m — those come from
# _load_daily_ng_features reading the daily Henry Hub XLS directly, so they
# are unaffected by the monthly source swap.
NG_DEPENDENT = {
    "ng_spot",
    "ng_lag1", "ng_lag2", "ng_lag3", "ng_lag4", "ng_lag5", "ng_lag6",
    "ng_lag9", "ng_lag12",
    "ng_rolling_mean_3m", "ng_rolling_mean_6m", "ng_rolling_mean_12m",
    "ng_rolling_std_3m",  "ng_rolling_std_6m",  "ng_rolling_std_12m",
    "ng_mom_1m", "ng_mom_3m", "ng_mom_6m",
    "ng_zscore_12m",
    "ng_dist_from_ma3",
    "urea_ng_ratio",
    "urea_ng_spread",
}

# Last pre-2006 rounding diff is at 2005-12. With 12-month rolling/lag windows
# that propagates through mid-2006-12; the 2006-08 boundary cell propagates
# through 2007-08-01 (ng_lag12 at exactly that month). Use 2007-09-01 as the
# exclusive upper bound so the 2007-08 boundary cell itself qualifies.
PROPAGATION_END = pd.Timestamp("2007-09-01")


def classify_extra_row(row: pd.Series) -> str:
    if pd.isna(row.get("urea")) and pd.isna(row.get("dap")):
        return "EXPECTED (NG-only prediction row — NaN urea/dap targets, dropped in training)"
    return "UNEXPECTED (urea or dap present)"


def main() -> int:
    print("=" * 78)
    print("Loading existing parquet feature store...")
    old = load_feature_store()
    print(f"  rows: {len(old)}   range: {old.index.min().date()} -> {old.index.max().date()}")
    print(f"  cols: {len(old.columns)}")

    print("\nBuilding new feature store via build_features(source='mart', start='1997-01')...")
    new = build_features(source="mart", start="1997-01")
    print(f"  rows: {len(new)}   range: {new.index.min().date()} -> {new.index.max().date()}")
    print(f"  cols: {len(new.columns)}")

    # ── Columns ─────────────────────────────────────────────────────────
    print("\n" + "-" * 78)
    print("Column comparison")
    print("-" * 78)
    cols_match = list(old.columns) == list(new.columns)
    print(f"  identical (names + order): {cols_match}")
    if not cols_match:
        missing_in_new = [c for c in old.columns if c not in new.columns]
        missing_in_old = [c for c in new.columns if c not in old.columns]
        print(f"    in parquet, not in mart-build: {missing_in_new}")
        print(f"    in mart-build, not in parquet: {missing_in_old}")

    # ── Index ───────────────────────────────────────────────────────────
    print("\n" + "-" * 78)
    print("Index comparison")
    print("-" * 78)
    only_old = old.index.difference(new.index)
    only_new = new.index.difference(old.index)
    common   = old.index.intersection(new.index)
    print(f"  common rows:                  {len(common)}")
    print(f"  rows only in parquet:         {len(only_old)}")
    for ts in only_old:
        print(f"    {ts.date()}: UNEXPECTED missing from mart-build")
    print(f"  rows only in mart-build:      {len(only_new)}")
    unexpected_trailing = 0
    for ts in only_new:
        verdict = classify_extra_row(new.loc[ts])
        print(f"    {ts.date()}: {verdict}")
        if verdict.startswith("UNEXPECTED"):
            unexpected_trailing += 1

    if not cols_match:
        print("\n" + "=" * 78)
        print("UNEXPLAINED — columns do not match; cannot run value comparison.")
        return 1

    # ── Per-column values on common rows ────────────────────────────────
    print("\n" + "-" * 78)
    print("Per-column value comparison (common rows only)")
    print("-" * 78)
    summary = []
    flagged = []
    common_sorted = common.sort_values()
    for col in old.columns:
        a = pd.to_numeric(old[col].reindex(common_sorted), errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(new[col].reindex(common_sorted), errors="coerce").to_numpy(dtype=float)
        a_nan = np.isnan(a)
        b_nan = np.isnan(b)
        diff = np.abs(a - b)
        mismatch = (~(a_nan & b_nan)) & ((a_nan != b_nan) | (diff > TOL))
        n_mismatch = int(mismatch.sum())
        if n_mismatch == 0:
            summary.append((col, 0, 0.0, None, None, "OK"))
            continue
        finite_diffs = diff[mismatch & ~np.isnan(diff)]
        max_diff = float(finite_diffs.max()) if finite_diffs.size else float("nan")
        idx_mis = common_sorted[mismatch]
        first = idx_mis.min()
        last  = idx_mis.max()
        if col in NG_DEPENDENT and last < PROPAGATION_END:
            tag = "EXPECTED (ng_spot propagation)"
        elif col == "storage_zscore" and len(only_new) > 0 and max_diff < 0.05:
            # storage_zscore uses df["storage_mmcf"].mean()/std() over the WHOLE
            # DataFrame, so trailing extra storage_mmcf rows in the mart build
            # shift it by a near-constant offset across every row. Sub-1%.
            tag = "EXPECTED (trailing storage_mmcf shifts global mean/std)"
        else:
            tag = "UNEXPLAINED"
            flagged.append((col, n_mismatch, max_diff, first, last))
        summary.append((col, n_mismatch, max_diff, first, last, tag))

    print(f"  {'column':<28} {'#diffs':>7} {'max|diff|':>12} {'first':>12} {'last':>12}  tag")
    for col, n, mx, f, l, tag in summary:
        if n == 0:
            print(f"  {col:<28} {n:>7} {'-':>12} {'-':>12} {'-':>12}  {tag}")
        else:
            print(f"  {col:<28} {n:>7} {mx:>12.6f} {str(f.date()):>12} {str(l.date()):>12}  {tag}")

    # ── Verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    expected_trailing = len(only_new) - unexpected_trailing
    if flagged or len(only_old) or unexpected_trailing:
        if flagged:
            print(f"UNEXPLAINED VALUE DIFFS in {len(flagged)} column(s):")
            for col, n, mx, f, l in flagged:
                print(f"  {col}: {n} diffs, max|diff|={mx:.6f}, {f.date()}..{l.date()}")
        if len(only_old):
            print(f"PARQUET HAS ROWS MISSING FROM MART-BUILD: {len(only_old)}")
        if unexpected_trailing:
            print(f"MART-BUILD HAS UNEXPECTED EXTRA ROWS: {unexpected_trailing}")
        return 1

    print(
        f"PARITY OK — value diffs traced to ng_spot pre-2006/2006-08 propagation; "
        f"{expected_trailing} expected NG-only trailing prediction row(s) added on the mart side."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
