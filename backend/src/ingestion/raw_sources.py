"""
Raw-source fetchers for the Postgres landing layer.

Each function returns a DataFrame at the source's NATIVE grain with minimal
metadata columns (source / source_file / loaded_at). No merging, resampling,
or precedence logic happens here — that work lives in dbt staging models so
the lineage stays explicit.
"""

import os
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

_THIS = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.normpath(os.path.join(_THIS, "..", "..", ".."))
DATA  = os.path.join(ROOT, "data")

load_dotenv(os.path.join(ROOT, ".env"))

_EIA_KEY       = os.getenv("EIA_API_KEY", "").strip()
_EIA_PAGE_SIZE = 5000


def _eia_paginated(url: str, params: dict) -> list[dict]:
    """Pull all pages from an EIA v2 endpoint, sorted ascending by period."""
    if not _EIA_KEY:
        raise RuntimeError("EIA_API_KEY not set in environment")
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


def fetch_eia_ng_spot_daily() -> pd.DataFrame:
    """
    Henry Hub spot ($/MMBtu) at daily grain from EIA API v2 series RNGWHHD.
    Columns: period, value_usd_per_mmbtu, series_id, source, loaded_at.
    """
    rows = _eia_paginated(
        "https://api.eia.gov/v2/natural-gas/pri/fut/data/",
        {"frequency": "daily", "data[0]": "value", "facets[series][]": "RNGWHHD"},
    )
    df = pd.DataFrame(
        [{"period": r["period"], "value_usd_per_mmbtu": r.get("value")}
         for r in rows if r.get("value") is not None]
    )
    df["period"] = pd.to_datetime(df["period"]).dt.date
    df["value_usd_per_mmbtu"] = pd.to_numeric(df["value_usd_per_mmbtu"], errors="coerce")
    df = df.dropna(subset=["value_usd_per_mmbtu"]).sort_values("period").reset_index(drop=True)
    df["series_id"] = "RNGWHHD"
    df["source"]    = "eia_api_v2"
    df["loaded_at"] = datetime.now(timezone.utc)
    return df


def fetch_eia_ng_storage_weekly() -> pd.DataFrame:
    """
    US Lower-48 underground working gas (BCF) at weekly grain from EIA API v2.
    duoarea=R48, process=SWO.
    Columns: period, value_bcf, duoarea, process, source, loaded_at.
    """
    rows = _eia_paginated(
        "https://api.eia.gov/v2/natural-gas/stor/wkly/data/",
        {
            "frequency":         "weekly",
            "data[0]":           "value",
            "facets[duoarea][]": "R48",
            "facets[process][]": "SWO",
        },
    )
    df = pd.DataFrame(
        [{"period": r["period"], "value_bcf": r.get("value")}
         for r in rows if r.get("value") is not None]
    )
    df["period"] = pd.to_datetime(df["period"]).dt.date
    df["value_bcf"] = pd.to_numeric(df["value_bcf"], errors="coerce")
    df = df.dropna(subset=["value_bcf"]).sort_values("period").reset_index(drop=True)
    df["duoarea"]   = "R48"
    df["process"]   = "SWO"
    df["source"]    = "eia_api_v2"
    df["loaded_at"] = datetime.now(timezone.utc)
    return df


def load_henry_hub_spot_monthly_csv() -> pd.DataFrame:
    """
    Henry Hub monthly spot ($/MMBtu) from the local CSV backbone.
    Columns: period (month-start), price_usd_per_mmbtu, source_file, loaded_at.
    """
    rel  = os.path.join("natural-gas-prices", "monthly.csv")
    path = os.path.join(DATA, rel)
    raw  = pd.read_csv(path)
    df = pd.DataFrame({
        "period": pd.to_datetime(raw["Month"]).dt.to_period("M").dt.to_timestamp().dt.date,
        "price_usd_per_mmbtu": pd.to_numeric(raw["Price"], errors="coerce"),
    })
    df = df.dropna(subset=["price_usd_per_mmbtu"]).sort_values("period").reset_index(drop=True)
    df["source_file"] = rel.replace("\\", "/")
    df["loaded_at"]   = datetime.now(timezone.utc)
    return df


def load_eia_ng_storage_monthly_xls() -> pd.DataFrame:
    """
    EIA underground working gas storage (MMcf) at monthly grain, from local XLS.
    Columns: period (month-start), working_gas_mmcf, source_file, loaded_at.
    """
    rel  = os.path.join("series_data", "NG_SUM_LSUM_DCU_NUS_M.xls")
    path = os.path.join(DATA, rel)
    raw  = pd.read_excel(path, sheet_name="Data 4", header=2)
    raw  = raw.rename(columns={raw.columns[0]: "Date"})
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"])
    col  = next(c for c in raw.columns if "Working Gas" in str(c))
    df = pd.DataFrame({
        "period": raw["Date"].dt.to_period("M").dt.to_timestamp().dt.date,
        "working_gas_mmcf": pd.to_numeric(raw[col], errors="coerce"),
    })
    df = df.dropna(subset=["working_gas_mmcf"]).sort_values("period").reset_index(drop=True)
    df["source_file"] = rel.replace("\\", "/")
    df["loaded_at"]   = datetime.now(timezone.utc)
    return df


def load_worldbank_pinksheet_monthly() -> pd.DataFrame:
    """
    World Bank Pink Sheet monthly urea & DAP prices ($/mt).
    Columns: period (month-start), urea_usd_per_mt, dap_usd_per_mt,
             source_file, loaded_at.
    """
    path_2026 = os.path.join(DATA, "raw", "CMO-Historical-Data-Monthly-2026.xlsx")
    path      = path_2026 if os.path.exists(path_2026) else os.path.join(DATA, "raw", "CMO-Historical-Data-Monthly.xlsx")
    raw = pd.read_excel(path, sheet_name="Monthly Prices", header=4, index_col=0)
    raw = raw.iloc[1:]
    fert = raw[["Urea ", "DAP"]].copy()
    fert.columns = ["urea_usd_per_mt", "dap_usd_per_mt"]
    fert = fert.apply(pd.to_numeric, errors="coerce")

    def _parse(s):
        try:
            year, month = str(s).strip().split("M")
            return pd.Timestamp(f"{year}-{month}-01")
        except Exception:
            return pd.NaT

    fert.index = fert.index.map(_parse)
    fert = fert[fert.index.notna()].sort_index()
    df = fert.reset_index().rename(columns={"index": "period"})
    df["period"] = df["period"].dt.date
    df["source_file"] = f"raw/{os.path.basename(path)}"
    df["loaded_at"]   = datetime.now(timezone.utc)
    return df[["period", "urea_usd_per_mt", "dap_usd_per_mt", "source_file", "loaded_at"]]


def load_henry_hub_spot_daily_xls() -> pd.DataFrame:
    """
    Henry Hub daily spot ($/MMBtu) from the local XLS — landed for future use by
    daily-derived features. engineer.py still reads the XLS directly today.
    Columns: period, price_usd_per_mmbtu, source_file, loaded_at.
    """
    rel  = "RNGWHHDd_henry_hub_nat_gas_spot_price_30_day_moving.xls"
    path = os.path.join(DATA, rel)
    raw  = pd.read_excel(path, sheet_name="Data 1", header=2)
    raw.columns = ["date", "price"]
    df = pd.DataFrame({
        "period":              pd.to_datetime(raw["date"], errors="coerce").dt.date,
        "price_usd_per_mmbtu": pd.to_numeric(raw["price"], errors="coerce"),
    })
    df = df.dropna(subset=["period", "price_usd_per_mmbtu"]).sort_values("period").reset_index(drop=True)
    df["source_file"] = rel
    df["loaded_at"]   = datetime.now(timezone.utc)
    return df
