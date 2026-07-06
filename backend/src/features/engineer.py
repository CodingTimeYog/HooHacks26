"""
Feature engineering — takes the ingestion dict and produces the feature store.
All features are derived exclusively from ng_spot, urea, dap, and storage.
Targets use negative shifts (look-forward) and are never used as input features.
"""

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

_THIS = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.normpath(os.path.join(_THIS, "..", "..", ".."))
DATA  = os.path.join(ROOT, "data")

load_dotenv(os.path.join(ROOT, ".env"))

MART_TABLE  = os.getenv("MART_TABLE", "dev_marts.mart_commodity_panel_monthly")
_PANEL_COLS = ["ng_spot", "urea", "dap", "storage_mmcf"]

FEATURE_COLS = [
    # Raw + lags
    "urea",
    "dap",
    "ng_spot",
    "ng_lag1", "ng_lag2", "ng_lag3", "ng_lag4", "ng_lag5", "ng_lag6",
    "ng_lag9", "ng_lag12",
    # Monthly rolling stats
    "ng_rolling_mean_3m", "ng_rolling_mean_6m", "ng_rolling_mean_12m",
    "ng_rolling_std_3m",  "ng_rolling_std_6m",  "ng_rolling_std_12m",
    # Monthly momentum
    "ng_mom_1m", "ng_mom_3m", "ng_mom_6m",
    # Daily-derived MA features (computed from Henry Hub daily data)
    "ng_ma30_eom", "ng_ma60_eom", "ng_ma90_eom",
    "ng_ma_cross_30_60", "ng_ma_cross_30_90",
    "ng_daily_vol_1m",
    # Urea features
    "urea_lag1", "urea_rolling_mean_3m", "urea_ng_ratio",
    "urea_mom_1m", "urea_mom_3m",
    # Mean-reversion features (key directional signals for commodities)
    "urea_zscore_12m", "urea_zscore_24m",
    "urea_vs_12m_high", "urea_vs_12m_low",
    "ng_zscore_12m",
    # DAP features
    "dap_lag1", "dap_urea_ratio",
    # Storage
    "storage_zscore",
    # Seasonality
    "season_q1", "season_q2", "season_q3", "season_q4",
]

TARGET_COLS = ["target_urea_t1", "target_urea_t2", "target_urea_t3"]


def _load_panel_from_mart(start: str = "1997-01", end: str | None = None) -> dict:
    """
    Read the four monthly series from dev_marts.mart_commodity_panel_monthly
    and return the same dict-of-Series shape that run_ingestion() returns.

    Reindexes to a contiguous MS range so callers see no gaps. When end is None,
    extends to the latest month that has ANY non-null value in the mart (which
    today is 2026-06 via NG — urea/DAP are NaN past 2025-12). This differs from
    run_ingestion's end=None semantics (min of latest-non-null across ALL four
    series, which clips to the shortest = 2025-12). Trailing NG-only rows are
    prediction-time rows; training drops them via NaN-target filtering.
    """
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL not set — required for source='mart'. "
            "Set it in .env or call build_features(source='pipeline')."
        )
    engine = create_engine(url, future=True)
    df = pd.read_sql(
        f"SELECT month, ng_spot, urea, dap, storage_mmcf "
        f"FROM {MART_TABLE} ORDER BY month",
        engine,
    )
    df["month"] = pd.to_datetime(df["month"])
    df = df.set_index("month").sort_index()
    for c in _PANEL_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    end_ts = pd.Timestamp(end) if end is not None else df.dropna(how="all").index.max()
    idx = pd.date_range(start=pd.Timestamp(start), end=end_ts, freq="MS")
    df = df.reindex(idx)
    return {c: df[c].rename(c) for c in _PANEL_COLS}


def _load_panel_from_pipeline(start: str = "1997-01", end: str | None = None) -> dict:
    """Selectable fallback — kept so we can flip back to the original pandas
    pipeline by passing source='pipeline'. Imported lazily so the default mart
    path doesn't pay pipeline.py's import-time costs."""
    from src.ingestion.pipeline import run_ingestion
    return run_ingestion(start=start, end=end)


_DAILY_NG_TABLE = "raw.eia_ng_spot_daily"


def _aggregate_daily_ng(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shared monthly-aggregation kernel for daily Henry Hub prices. Input must have
    a 'date' column and a 'price' column; both XLS and DB paths normalise to this
    shape before calling here so the rolling/EOM semantics are bit-identical.
    """
    df = df.dropna().set_index("date").sort_index()

    # Daily rolling MAs (positional row windows over business-day-only series,
    # min_periods = half window).
    df["ma30"] = df["price"].rolling(30, min_periods=15).mean()
    df["ma60"] = df["price"].rolling(60, min_periods=30).mean()
    df["ma90"] = df["price"].rolling(90, min_periods=45).mean()

    # Aggregate to month-start: last available daily value within the month for
    # MAs, std of daily prices for volatility.
    monthly = pd.DataFrame({
        "ng_ma30_eom":    df["ma30"].resample("MS").last(),
        "ng_ma60_eom":    df["ma60"].resample("MS").last(),
        "ng_ma90_eom":    df["ma90"].resample("MS").last(),
        "ng_daily_vol_1m": df["price"].resample("MS").std(),
    })

    # MA crossover signals: >0 = short-term trend above long-term (bullish)
    monthly["ng_ma_cross_30_60"] = monthly["ng_ma30_eom"] / monthly["ng_ma60_eom"] - 1
    monthly["ng_ma_cross_30_90"] = monthly["ng_ma30_eom"] / monthly["ng_ma90_eom"] - 1

    return monthly


def _load_daily_ng_features_from_xls() -> pd.DataFrame:
    """Daily Henry Hub MA features from the local EIA XLS. Returns empty DataFrame
    if the file is missing or unreadable."""
    path = os.path.join(DATA, "RNGWHHDd_henry_hub_nat_gas_spot_price_30_day_moving.xls")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_excel(path, sheet_name="Data 1", header=2)
        df.columns = ["date", "price"]
        df["date"]  = pd.to_datetime(df["date"], errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        return _aggregate_daily_ng(df)
    except Exception as exc:
        print(f"     [features] Daily NG XLS load failed ({exc}) — skipping daily features.")
        return pd.DataFrame()


def _load_daily_ng_features_from_db() -> pd.DataFrame:
    """Daily Henry Hub MA features from raw.eia_ng_spot_daily via DATABASE_URL.
    Parity-validated against the XLS path: daily prices match exactly on the
    7327-row overlap; the only divergence is upstream EIA holiday-fill revisions
    on 9 dates, shifting 5 historical monthly MAs by ≤$0.04."""
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL not set — required for daily_source='db'.")
    engine = create_engine(url, future=True)
    df = pd.read_sql(
        f"SELECT period AS date, value_usd_per_mmbtu AS price "
        f"FROM {_DAILY_NG_TABLE} ORDER BY period",
        engine,
    )
    df["date"]  = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return _aggregate_daily_ng(df)


def _load_daily_ng_features(daily_source: str = "auto") -> pd.DataFrame:
    """
    Dispatcher for daily NG MA features. Mirrors the source='mart'/'pipeline'
    pattern on the monthly panel:
      - 'auto' (default) : DB when DATABASE_URL is set, else XLS fallback.
      - 'db'             : raw.eia_ng_spot_daily; errors if DATABASE_URL unset.
      - 'xls'            : local Henry Hub XLS.
    """
    if daily_source == "db":
        return _load_daily_ng_features_from_db()
    if daily_source == "xls":
        return _load_daily_ng_features_from_xls()
    if daily_source != "auto":
        raise ValueError(f"Unknown daily_source={daily_source!r}. Use 'auto', 'db', or 'xls'.")

    if os.getenv("DATABASE_URL", "").strip():
        try:
            return _load_daily_ng_features_from_db()
        except Exception as exc:
            print(f"     [features] Daily NG DB load failed ({exc}) — falling back to XLS.")
    return _load_daily_ng_features_from_xls()


def build_features(
    data: dict | None = None,
    *,
    source: str = "mart",
    daily_source: str = "auto",
    start: str = "1997-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Build the wide feature-store DataFrame.

    Sources for the four monthly series (ng_spot, urea, dap, storage_mmcf):
      - data=<dict>                    : use the supplied dict as-is (legacy
                                         path — train_models.py still passes
                                         run_ingestion's output this way).
      - data=None, source='mart'       : read dev_marts.mart_commodity_panel_monthly
                                         via DATABASE_URL (default).
      - data=None, source='pipeline'   : fall back to run_ingestion() (kept
                                         selectable so we can flip back).

    Daily-derived MA features (ng_ma30/60/90_eom, ng_daily_vol_1m, crossovers):
      - daily_source='auto' (default)  : raw.eia_ng_spot_daily via DATABASE_URL
                                         if set, else local Henry Hub XLS.
      - daily_source='db' / 'xls'      : force either source.

    Rows with all-NaN targets (last 3 rows) are kept — they are used for
    out-of-sample prediction; drop them only during model training.
    """
    if data is None:
        if source == "mart":
            data = _load_panel_from_mart(start=start, end=end)
        elif source == "pipeline":
            data = _load_panel_from_pipeline(start=start, end=end)
        else:
            raise ValueError(f"Unknown source={source!r}. Use 'mart' or 'pipeline'.")

    df = pd.DataFrame(data)

    # ── Nat gas lags (1–6 monthly + 9 and 12 for seasonality)
    for lag in [1, 2, 3, 4, 5, 6, 9, 12]:
        df[f"ng_lag{lag}"] = df["ng_spot"].shift(lag)

    # ── Rolling stats (3m, 6m, 12m)
    df["ng_rolling_mean_3m"]  = df["ng_spot"].rolling(3).mean()
    df["ng_rolling_mean_6m"]  = df["ng_spot"].rolling(6).mean()
    df["ng_rolling_mean_12m"] = df["ng_spot"].rolling(12).mean()
    df["ng_rolling_std_3m"]   = df["ng_spot"].rolling(3).std()
    df["ng_rolling_std_6m"]   = df["ng_spot"].rolling(6).std()
    df["ng_rolling_std_12m"]  = df["ng_spot"].rolling(12).std()

    # ── Momentum (% change)
    df["ng_mom_1m"] = df["ng_spot"].pct_change(1)
    df["ng_mom_3m"] = df["ng_spot"].pct_change(3)
    df["ng_mom_6m"] = df["ng_spot"].pct_change(6)

    # ── Daily-derived MA features — join by month-start index
    daily_feats = _load_daily_ng_features(daily_source=daily_source)
    if not daily_feats.empty:
        df = df.join(daily_feats, how="left")
    else:
        # Fall back to monthly proxies so training rows aren't dropped
        df["ng_ma30_eom"]      = df["ng_rolling_mean_3m"]
        df["ng_ma60_eom"]      = df["ng_rolling_mean_6m"]
        df["ng_ma90_eom"]      = df["ng_rolling_mean_6m"]
        df["ng_ma_cross_30_60"] = df["ng_rolling_mean_3m"] / df["ng_rolling_mean_6m"] - 1
        df["ng_ma_cross_30_90"] = df["ng_rolling_mean_3m"] / df["ng_rolling_mean_6m"] - 1
        df["ng_daily_vol_1m"]  = df["ng_rolling_std_3m"]

    # ── Urea features
    df["urea_lag1"]            = df["urea"].shift(1)
    df["urea_rolling_mean_3m"] = df["urea"].rolling(3).mean()
    df["urea_ng_ratio"]        = df["urea"] / df["ng_spot"]
    df["urea_mom_1m"]          = df["urea"].pct_change(1)
    df["urea_mom_3m"]          = df["urea"].pct_change(3)

    # ── Mean-reversion features — where is urea/NG relative to recent history?
    # High z-score = overextended high → more likely to fall (and vice versa)
    urea_mean_12m = df["urea"].rolling(12).mean()
    urea_std_12m  = df["urea"].rolling(12).std()
    urea_mean_24m = df["urea"].rolling(24).mean()
    urea_std_24m  = df["urea"].rolling(24).std()
    df["urea_zscore_12m"]  = (df["urea"] - urea_mean_12m) / urea_std_12m
    df["urea_zscore_24m"]  = (df["urea"] - urea_mean_24m) / urea_std_24m
    df["urea_vs_12m_high"] = df["urea"] / df["urea"].rolling(12).max()   # 1.0 = at 12m high
    df["urea_vs_12m_low"]  = df["urea"] / df["urea"].rolling(12).min()   # 1.0 = at 12m low
    ng_mean_12m = df["ng_spot"].rolling(12).mean()
    ng_std_12m  = df["ng_spot"].rolling(12).std()
    df["ng_zscore_12m"] = (df["ng_spot"] - ng_mean_12m) / ng_std_12m

    # ── DAP features
    df["dap_lag1"]      = df["dap"].shift(1)
    df["dap_urea_ratio"] = df["dap"] / df["urea"]

    # ── Storage z-score (low = tight supply = upward price pressure)
    # Expanding stats up to and including row t — matches the rolling-window
    # convention used elsewhere in this file (rolling(N) includes t), and is
    # leakage-safe because row t never sees rows > t. std() at row 0 is NaN,
    # so storage_zscore is NaN only at the very first row; that row is already
    # dropped by the 24-month urea z-score gating in the training filter.
    storage_mean = df["storage_mmcf"].expanding().mean()
    storage_std  = df["storage_mmcf"].expanding().std()
    df["storage_zscore"] = (df["storage_mmcf"] - storage_mean) / storage_std

    # ── Season dummies (quarter indicators)
    df["season_q1"] = df.index.month.isin([1, 2, 3]).astype(int)
    df["season_q2"] = df.index.month.isin([4, 5, 6]).astype(int)
    df["season_q3"] = df.index.month.isin([7, 8, 9]).astype(int)
    df["season_q4"] = df.index.month.isin([10, 11, 12]).astype(int)

    # ── Targets — negative shifts look FORWARD in time
    # These are NaN for the last 1/2/3 rows respectively
    df["target_urea_t1"] = df["urea"].shift(-1)
    df["target_urea_t2"] = df["urea"].shift(-2)
    df["target_urea_t3"] = df["urea"].shift(-3)

    return df


def save_feature_store(df: pd.DataFrame) -> str:
    out_dir = os.path.join(ROOT, "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "feature_store.parquet")
    df.to_parquet(path)
    return path


def load_feature_store() -> pd.DataFrame:
    path = os.path.join(ROOT, "data", "processed", "feature_store.parquet")
    return pd.read_parquet(path)
