"""
Validate that dev_marts.mart_commodity_panel_monthly matches run_ingestion()
row-for-row.

Comparison rules:
  - month index: report any months present in only one side.
  - per-cell values: NaN/NULL == NaN/NULL is treated as matching; otherwise
    absolute diff must be <= TOL.

Exit code: 0 on parity, 1 on any unexpected difference.

Provenance: the original version of this script carried an --accept-eia-precision-gap
flag whitelisting pre-2006 ng_spot diffs that came from pipeline._fetch_eia_ng_spot's
unpaginated length=5000 fetch. That truncation was fixed on 2026-06-07 by porting
raw_sources' _eia_paginated helper into pipeline.py, so the whitelist was removed.

Usage (from project root):
    python scripts/validate_mart_parity.py
"""

import os
import sys
from datetime import date

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

_THIS = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.normpath(os.path.join(_THIS, ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))
load_dotenv(os.path.join(ROOT, ".env"))

from src.ingestion.pipeline import run_ingestion  # noqa: E402

# Absolute tolerance. Series live in ~[0, 10_000] (storage MMcf) and ~[0, 1_000]
# (urea/dap). 1e-6 catches any real difference, ignores float round-trip noise.
TOL = 1e-6
COLS = ["ng_spot", "urea", "dap", "storage_mmcf"]

MART_TABLE = "dev_marts.mart_commodity_panel_monthly"


def read_mart() -> pd.DataFrame:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL not set in environment")
    engine = create_engine(url, future=True)
    df = pd.read_sql(
        f"SELECT month, ng_spot, urea, dap, storage_mmcf "
        f"FROM {MART_TABLE} ORDER BY month",
        engine,
    )
    df["month"] = pd.to_datetime(df["month"])
    df = df.set_index("month").sort_index()
    for c in COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[COLS]


def read_pipeline(start: str, end: str) -> pd.DataFrame:
    """run_ingestion() returns a dict of monthly Series; collapse to a DataFrame."""
    data = run_ingestion(start=start, end=end)
    df = pd.DataFrame({k: data[k] for k in COLS})
    for c in COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _fmt(v) -> str:
    return "NaN" if pd.isna(v) else f"{float(v):.6f}"


def main() -> int:
    print("=" * 70)
    print(f"Reading mart: {MART_TABLE}")
    mart = read_mart()
    print(f"  rows: {len(mart):>4}   range: {mart.index.min().date()} -> {mart.index.max().date()}")

    # Compare over the mart's full window so we surface any expected NaN tails
    # (e.g. Pink Sheet ends 2025-12 while NG reaches 2026-06).
    start = mart.index.min().strftime("%Y-%m-%d")
    end   = mart.index.max().strftime("%Y-%m-%d")

    print(f"\nCalling run_ingestion(start={start}, end={end})")
    pipe = read_pipeline(start, end)
    print(f"  rows: {len(pipe):>4}   range: {pipe.index.min().date()} -> {pipe.index.max().date()}")

    # ── Index alignment ────────────────────────────────────────────────────
    only_in_mart = mart.index.difference(pipe.index)
    only_in_pipe = pipe.index.difference(mart.index)
    common       = mart.index.intersection(pipe.index)

    print("\n" + "-" * 70)
    print("Index alignment")
    print("-" * 70)
    print(f"  common months:        {len(common)}")
    print(f"  only in mart:         {len(only_in_mart)}")
    if len(only_in_mart):
        for m in only_in_mart[:10]:
            print(f"    {m.date()}  {mart.loc[m].to_dict()}")
        if len(only_in_mart) > 10:
            print(f"    ... ({len(only_in_mart) - 10} more)")
    print(f"  only in pipeline:     {len(only_in_pipe)}")
    if len(only_in_pipe):
        for m in only_in_pipe[:10]:
            print(f"    {m.date()}  {pipe.loc[m].to_dict()}")
        if len(only_in_pipe) > 10:
            print(f"    ... ({len(only_in_pipe) - 10} more)")

    # ── Per-column cell comparison ────────────────────────────────────────
    print("\n" + "-" * 70)
    print("Value comparison (common months only)")
    print("-" * 70)
    real_diffs: dict[str, list[tuple[date, float, float]]] = {c: [] for c in COLS}
    for m in common:
        for c in COLS:
            mv, pv = mart.at[m, c], pipe.at[m, c]
            mn, pn = pd.isna(mv), pd.isna(pv)
            if mn and pn:
                continue
            if mn != pn or abs(float(mv) - float(pv)) > TOL:
                real_diffs[c].append((m, mv, pv))

    total_cells     = len(common) * len(COLS)
    real_diff_cells = sum(len(v) for v in real_diffs.values())
    print(f"  matching cells:       {total_cells - real_diff_cells} / {total_cells}")
    print(f"  real differing cells: {real_diff_cells}")

    for c in COLS:
        rows = real_diffs[c]
        if not rows:
            print(f"    {c:<14} OK")
            continue
        print(f"    {c:<14} {len(rows)} diff(s):")
        for m, mv, pv in rows[:10]:
            d = "NaN mismatch" if (pd.isna(mv) or pd.isna(pv)) else f"{float(mv) - float(pv):+.6f}"
            print(f"      {m.date()}  mart={_fmt(mv)}  pipe={_fmt(pv)}  diff={d}")
        if len(rows) > 10:
            print(f"      ... ({len(rows) - 10} more)")

    # ── Verdict ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if real_diff_cells == 0 and len(only_in_mart) == 0 and len(only_in_pipe) == 0:
        print("PARITY OK — mart matches run_ingestion() across overlapping months.")
        return 0
    print("MISMATCH PRESENT — see report above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
