"""
Land each raw source DataFrame into Postgres under the `raw` schema.

Each table is rewritten in full on every run (if_exists="replace") — each
source fetcher already returns the complete available history, so per-run
truncation keeps things simple and idempotent. dbt staging models handle
merge/precedence downstream.

Usage (from project root):
    python backend/src/ingestion/postgres_loader.py
    python backend/src/ingestion/postgres_loader.py --only eia_ng_spot_daily,worldbank_pinksheet_monthly
"""

import argparse
import os
import sys
from typing import Callable

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

# Make `src.*` importable when run as a script from the project root,
# matching the convention in backend/run_pipeline.py.
_THIS = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.normpath(os.path.join(_THIS, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from src.ingestion.raw_sources import (  # noqa: E402
    fetch_eia_ng_spot_daily,
    fetch_eia_ng_storage_weekly,
    load_eia_ng_storage_monthly_xls,
    load_henry_hub_spot_daily_xls,
    load_henry_hub_spot_monthly_csv,
    load_worldbank_pinksheet_monthly,
)

load_dotenv(os.path.join(ROOT, ".env"))

RAW_SCHEMA = "raw"

# (table_name, fetcher) pairs — order is the load order
RAW_TABLES: list[tuple[str, Callable[[], pd.DataFrame]]] = [
    ("eia_ng_spot_daily",           fetch_eia_ng_spot_daily),
    ("eia_ng_storage_weekly",       fetch_eia_ng_storage_weekly),
    ("henry_hub_spot_monthly_csv",  load_henry_hub_spot_monthly_csv),
    ("eia_ng_storage_monthly_xls",  load_eia_ng_storage_monthly_xls),
    ("worldbank_pinksheet_monthly", load_worldbank_pinksheet_monthly),
    ("henry_hub_spot_daily_xls",    load_henry_hub_spot_daily_xls),
]


def get_engine() -> Engine:
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL not set. Expected e.g. "
            "postgresql+psycopg2://user:pass@host:5432/foregast"
        )
    return create_engine(url, future=True)


def ensure_schema(engine: Engine, schema: str = RAW_SCHEMA) -> None:
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))


def write_raw_table(engine: Engine, df: pd.DataFrame, table: str, schema: str = RAW_SCHEMA) -> None:
    # Preserve the table object on re-runs so dependent dbt views (dev_staging.stg_*)
    # don't block the load with DependentObjectsStillExist. TRUNCATE + append keeps
    # the same relation; we only fall back to replace on the very first run when the
    # table doesn't exist yet.
    exists = inspect(engine).has_table(table, schema=schema)
    if exists:
        with engine.begin() as conn:
            conn.execute(text(f'TRUNCATE TABLE "{schema}"."{table}"'))
        if_exists = "append"
    else:
        if_exists = "replace"
    df.to_sql(
        name=table,
        schema=schema,
        con=engine,
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=1000,
    )


def _report(table: str, df: pd.DataFrame) -> None:
    if df.empty or "period" not in df.columns:
        print(f"  [{table:<32}] {len(df):>6} rows")
        return
    print(
        f"  [{table:<32}] {len(df):>6} rows "
        f"({df['period'].min()} -> {df['period'].max()})"
    )


def load_all_raw(only: set[str] | None = None) -> dict[str, int]:
    """
    Fetch every (or a subset of) raw source and write it to Postgres.

    Any fetch or write failure propagates — we do NOT swallow exceptions or
    continue past a failure. The caller (or Python's default handler) gets the
    full traceback and a non-zero exit code.
    """
    engine = get_engine()
    ensure_schema(engine)
    # Probe the connection up front so a bad DATABASE_URL fails before any fetch work.
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

    counts: dict[str, int] = {}
    print(f"Writing raw tables to schema `{RAW_SCHEMA}`")
    for table, fetcher in RAW_TABLES:
        if only is not None and table not in only:
            continue
        df = fetcher()
        if df.empty:
            raise RuntimeError(f"{table}: fetcher returned 0 rows — refusing to write an empty table")
        write_raw_table(engine, df, table)
        _report(table, df)
        counts[table] = len(df)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Land raw sources into Postgres `raw` schema.")
    parser.add_argument(
        "--only",
        help="Comma-separated subset of table names to load (default: all).",
    )
    args = parser.parse_args()
    only = {t.strip() for t in args.only.split(",")} if args.only else None
    # No try/except: let any failure print a full traceback and exit non-zero.
    load_all_raw(only=only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
