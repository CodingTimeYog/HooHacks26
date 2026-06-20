"""
Data ingestion layer — loads all data sources, normalizes to a monthly
DatetimeIndex, and returns a clean dict of pd.Series ready for feature engineering.

For natural gas spot price and storage, the EIA API v2 is tried first so the
pipeline picks up the latest available month automatically.  If the API call
fails (no key, network error, rate-limit) the code falls back to the local
static files that ship with the repo.

Urea and DAP still come from the World Bank Pink Sheet Excel file — no free
live source exists for those commodities.
"""

import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Resolve paths relative to this file regardless of cwd
_THIS  = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.normpath(os.path.join(_THIS, "..", "..", ".."))
DATA   = os.path.join(ROOT, "data")

# Load .env from project root (picks up EIA_API_KEY)
load_dotenv(os.path.join(ROOT, ".env"))

_EIA_KEY = os.getenv("EIA_API_KEY", "")
_EIA_PAGE_SIZE = 5000


# ── EIA API helpers ────────────────────────────────────────────────────────────

def _eia_paginated(url: str, params: dict) -> list[dict]:
    """Pull all pages from an EIA v2 endpoint, ascending by period.

    Mirrors the helper in raw_sources.py — without pagination we cap at one page
    (~5000 daily rows = ~2006→today for NG spot) and silently drop earlier
    history, which is what caused the pre-2006 parity diffs against the mart.
    """
    rows: list[dict] = []
    offset = 0
    while True:
        page_params = {
            **params,
            "api_key":            _EIA_KEY,
            "length":             _EIA_PAGE_SIZE,
            "offset":             offset,
            "sort[0][column]":    "period",
            "sort[0][direction]": "asc",
        }
        r = requests.get(url, params=page_params, timeout=60)
        r.raise_for_status()
        page = r.json()["response"]["data"]
        if not page:
            break
        rows.extend(page)
        if len(page) < _EIA_PAGE_SIZE:
            break
        offset += _EIA_PAGE_SIZE
    return rows


def _fetch_eia_ng_spot() -> pd.Series | None:
    """
    Fetch Henry Hub spot price ($/MMBtu) from EIA API v2, then merge with
    local CSV to get full history back to 1997. EIA data takes precedence
    for recent months where both overlap.
    Returns None on any failure.
    """
    if not _EIA_KEY:
        return None
    try:
        rows = _eia_paginated(
            "https://api.eia.gov/v2/natural-gas/pri/fut/data/",
            {"frequency": "daily", "data[0]": "value", "facets[series][]": "RNGWHHD"},
        )
        valid_rows = [row for row in rows if row.get("value") is not None and row.get("period")]
        if not valid_rows:
            print("     [EIA] NG spot — empty response, using local file.")
            return None

        dates  = pd.to_datetime([row["period"] for row in valid_rows], errors="coerce")
        values = [float(row["value"]) for row in valid_rows]

        # Daily → monthly average
        live = pd.Series(values, index=dates, name="ng_spot").dropna().sort_index()
        live = live.resample("MS").mean()

        # Load local CSV for full history back to 1997
        local = load_ng_monthly()

        # Merge: local provides backbone, live EIA overwrites recent months
        combined = local.copy()
        combined.update(live)
        # Extend beyond local file's end date with live data
        new_months = live[live.index > local.index.max()]
        combined = pd.concat([combined, new_months]).sort_index()
        combined.name = "ng_spot"

        print(f"     [EIA] NG spot merged — {combined.index[0].strftime('%Y-%m')} to {combined.index[-1].strftime('%Y-%m')} ({len(combined)} months)")
        return combined

    except Exception as exc:
        print(f"     [EIA] NG spot fetch failed ({exc}) — using local file.")
        return None
    
def _fetch_eia_ng_storage() -> pd.Series | None:
    """
    Fetch US Lower 48 underground working gas storage (BCF) from EIA API v2,
    then merge with local XLS to get full history back to 1997.
    Uses duoarea=R48 + process=SWO (total working gas).
    EIA returns values in BCF; we convert to MMcf (* 1000) to match local file.
    Returns None on any failure.
    """
    if not _EIA_KEY:
        return None
    try:
        rows = _eia_paginated(
            "https://api.eia.gov/v2/natural-gas/stor/wkly/data/",
            {
                "frequency":         "weekly",
                "data[0]":           "value",
                "facets[duoarea][]": "R48",
                "facets[process][]": "SWO",
            },
        )
        valid_rows = [row for row in rows if row.get("value") is not None and row.get("period")]
        if not valid_rows:
            print("     [EIA] Storage — empty response, using local file.")
            return None

        dates  = pd.to_datetime([row["period"] for row in valid_rows], errors="coerce")
        values = [float(row["value"]) * 1000 for row in valid_rows]

        live = pd.Series(values, index=dates, name="storage_mmcf").dropna().sort_index()
        # Deduplicate then resample weekly → monthly
        live = live.groupby(live.index).mean()
        live = live.resample("MS").last()

        # Load local XLS for full history back to 1997
        local = load_ng_storage()
        local = local.resample("MS").last()

        # Merge: local provides backbone, live EIA overwrites recent months
        combined = local.copy()
        combined.update(live)
        # Extend beyond local file's end date with live data
        new_months = live[live.index > local.index.max()]
        combined = pd.concat([combined, new_months]).sort_index()
        combined.name = "storage_mmcf"

        print(f"     [EIA] Storage merged — {combined.index[0].strftime('%Y-%m')} to {combined.index[-1].strftime('%Y-%m')} ({len(combined)} months)")
        return combined

    except Exception as exc:
        print(f"     [EIA] Storage fetch failed ({exc}) — using local file.")
        return None    

# ── Local file loaders (fallback) ─────────────────────────────────────────────

def load_ng_monthly() -> pd.Series:
    """Henry Hub monthly spot price ($/MMBtu) — local CSV fallback."""
    path = os.path.join(DATA, "natural-gas-prices", "monthly.csv")
    df = pd.read_csv(path)
    df["Month"] = pd.to_datetime(df["Month"])
    df = df.set_index("Month").sort_index()
    series = df["Price"].ffill()
    series.index = series.index.to_period("M").to_timestamp()
    return series.rename("ng_spot")


def load_fertilizer_prices() -> pd.DataFrame:
    """
    World Bank Pink Sheet — Urea and DAP monthly prices ($/mt).
    Returns DataFrame with columns ['urea', 'dap'], DatetimeIndex (month-start).
    """
    path_2026 = os.path.join(DATA, "raw", "CMO-Historical-Data-Monthly-2026.xlsx")
    path      = path_2026 if os.path.exists(path_2026) else os.path.join(DATA, "raw", "CMO-Historical-Data-Monthly.xlsx")
    raw = pd.read_excel(path, sheet_name="Monthly Prices", header=4, index_col=0)
    # Row 0 after the header is the units row — drop it
    raw = raw.iloc[1:]

    fert = raw[["Urea ", "DAP"]].copy()
    fert.columns = ["urea", "dap"]
    fert = fert.apply(pd.to_numeric, errors="coerce")

    # Index format: '1960M01' → parse to Timestamp
    def _parse(s):
        try:
            year, month = str(s).strip().split("M")
            return pd.Timestamp(f"{year}-{month}-01")
        except Exception:
            return pd.NaT

    fert.index = fert.index.map(_parse)
    fert = fert[fert.index.notna()].sort_index()
    # Forward-fill up to 2 consecutive missing months (World Bank has occasional gaps)
    fert = fert.ffill(limit=2)
    return fert


def load_ng_storage() -> pd.Series:
    """EIA underground storage working gas (MMcf), monthly — local XLS fallback."""
    path = os.path.join(DATA, "series_data", "NG_SUM_LSUM_DCU_NUS_M.xls")
    df = pd.read_excel(path, sheet_name="Data 4", header=2)
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    df.index = df.index.to_period("M").to_timestamp()

    col = next(c for c in df.columns if "Working Gas" in str(c))
    series = pd.to_numeric(df[col], errors="coerce").rename("storage_mmcf")
    return series


def run_ingestion(start: str = "2018-01", end: str | None = None) -> dict:
    """
    Returns dict with keys:
      'ng_spot'      — monthly Henry Hub spot price ($/MMBtu)
      'urea'         — monthly urea price ($/mt)
      'dap'          — monthly DAP price ($/mt)
      'storage_mmcf' — monthly underground storage working gas (MMcf)

    Tries EIA API v2 first for nat gas spot and storage. Falls back to local
    files if the API fails. Urea/DAP come from World Bank Pink Sheet (no live
    free source exists for fertilizer prices).
    """
    # Try EIA API first; fall back to local files if it fails
    ng = _fetch_eia_ng_spot()
    if ng is None:
        print("     [EIA] Using local NG spot file.")
        ng = load_ng_monthly()

    stor = _fetch_eia_ng_storage()
    if stor is None:
        print("     [EIA] Using local storage file.")
        stor = load_ng_storage()

    # Urea/DAP: World Bank only — no live API
    fert = load_fertilizer_prices()

    # Determine end date: use latest month present across all series
    if end is None:
        end = min(
            ng.dropna().index.max(),
            fert["urea"].dropna().index.max(),
            fert["dap"].dropna().index.max(),
            stor.dropna().index.max(),
        ).strftime("%Y-%m")

    idx = pd.date_range(start=start, end=end, freq="MS")

    return {
        "ng_spot":      ng.reindex(idx).rename("ng_spot"),
        "urea":         fert["urea"].reindex(idx).rename("urea"),
        "dap":          fert["dap"].reindex(idx).rename("dap"),
        "storage_mmcf": stor.reindex(idx).rename("storage_mmcf"),
    }

_MART_TABLE = "dev_marts.mart_commodity_panel_monthly"


def _load_full_history_from_mart() -> dict:
    """All-history NG/urea/DAP read from the mart, mirroring
    engineer.build_features(source='mart'). Requires DATABASE_URL."""
    from sqlalchemy import create_engine  # local import keeps pipeline.py import-time light

    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL not set — required for mart-backed history.")
    engine = create_engine(url, future=True)
    df = pd.read_sql(
        f"SELECT month, ng_spot, urea, dap FROM {_MART_TABLE} ORDER BY month",
        engine,
    )
    df["month"] = pd.to_datetime(df["month"])
    df = df.set_index("month").sort_index()
    for c in ("ng_spot", "urea", "dap"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return {
        "ng_spot": df["ng_spot"].rename("ng_spot"),
        "urea":    df["urea"].rename("urea"),
        "dap":     df["dap"].rename("dap"),
    }


def _load_full_history_from_local() -> dict:
    """Legacy local-file path — kept as the fallback when DATABASE_URL is unset."""
    ng   = load_ng_monthly()
    fert = load_fertilizer_prices()
    return {"ng_spot": ng, "urea": fert["urea"], "dap": fert["dap"]}


def load_full_history(source: str = "auto") -> dict:
    """
    All-history NG/urea/DAP for the chart payload.

    Mirrors the source='mart'/'pipeline' pattern in engineer.build_features:
      - 'auto' (default) : mart when DATABASE_URL is set, else local files.
      - 'mart'           : dev_marts.mart_commodity_panel_monthly; errors if env unset.
      - 'local'          : legacy CSV/XLSX path.
    """
    if source == "mart":
        return _load_full_history_from_mart()
    if source == "local":
        return _load_full_history_from_local()
    if source != "auto":
        raise ValueError(f"Unknown source={source!r}. Use 'auto', 'mart', or 'local'.")

    if os.getenv("DATABASE_URL", "").strip():
        try:
            return _load_full_history_from_mart()
        except Exception as exc:
            print(f"     [history] Mart load failed ({exc}) — falling back to local files.")
    return _load_full_history_from_local()