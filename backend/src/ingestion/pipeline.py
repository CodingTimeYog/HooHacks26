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


# ── EIA API helpers ────────────────────────────────────────────────────────────

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
        r = requests.get(
            "https://api.eia.gov/v2/natural-gas/pri/fut/data/",
            params={
                "api_key":              _EIA_KEY,
                "frequency":            "daily",
                "data[0]":              "value",
                "facets[series][]":     "RNGWHHD",
                "sort[0][column]":      "period",
                "sort[0][direction]":   "desc",
                "length":               5000,
            },
            timeout=30,
        )
        r.raise_for_status()
        rows = r.json()["response"]["data"]
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
        r = requests.get(
            "https://api.eia.gov/v2/natural-gas/stor/wkly/data/",
            params={
                "api_key":              _EIA_KEY,
                "frequency":            "weekly",
                "data[0]":              "value",
                "facets[duoarea][]":    "R48",
                "facets[process][]":    "SWO",
                "sort[0][column]":      "period",
                "sort[0][direction]":   "desc",
                "length":               1000,
            },
            timeout=30,
        )
        r.raise_for_status()
        rows = r.json()["response"]["data"]
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

def load_full_history() -> dict:
    """
    Same as run_ingestion but returns all available history (no date clipping).
    Used to build the historical chart payload.
    """
    ng   = load_ng_monthly()
    fert = load_fertilizer_prices()
    return {"ng_spot": ng, "urea": fert["urea"], "dap": fert["dap"]}