"""Forecast and projection routes — all read from the same cache bundle."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Response

from api.cache_store import CacheStore, get_cache_store
from api.schemas import (
    ForecastBand,
    ForecastBundle,
    ForecastSummary,
    LabeledSeries,
    MonteCarloBundle,
)

router         = APIRouter(prefix="/forecast",    tags=["forecast"])
history_router = APIRouter(prefix="/history",     tags=["history"])
mc_router      = APIRouter(prefix="/monte-carlo", tags=["monte-carlo"])


def _months_from_anchor(anchor: str, n: int) -> list[str]:
    """anchor='2026-05', n=3 → ['2026-06','2026-07','2026-08']."""
    base = datetime.strptime(anchor, "%Y-%m")
    out: list[str] = []
    for i in range(1, n + 1):
        y = base.year + (base.month - 1 + i) // 12
        m = (base.month - 1 + i) % 12 + 1
        out.append(f"{y:04d}-{m:02d}")
    return out


def _build_forecast(payload: dict) -> ForecastBand:
    fc = payload["forecast"]
    return ForecastBand(
        labels=fc["labels"],
        months=_months_from_anchor(payload["as_of_date"], len(fc["labels"])),
        mean=fc["mean"],
        low=fc["low"],
        high=fc["high"],
    )


def _set_cache_headers(response: Response, store: CacheStore) -> None:
    mtime = store.mtime()
    response.headers["ETag"]          = f'W/"{int(mtime.timestamp())}"'
    response.headers["Last-Modified"] = mtime.strftime("%a, %d %b %Y %H:%M:%S GMT")
    response.headers["Cache-Control"] = "public, max-age=300"


@router.get("/urea", response_model=ForecastBundle)
def get_urea_forecast(
    response: Response,
    store: CacheStore = Depends(get_cache_store),
) -> ForecastBundle:
    payload = store.read()
    _set_cache_headers(response, store)
    return ForecastBundle(
        generated_at=payload["generated_at"],
        as_of_date=payload["as_of_date"],
        urea_history=payload["urea_history"],
        natgas_history=payload["natgas_history"],
        forecast=_build_forecast(payload),
        signal=payload["signal"],
        monte_carlo=payload["monte_carlo"],
        sim_t2_distribution=payload["sim_t2_distribution"],
        model_metadata=payload["model_metadata"],
    )


@router.get("/urea/summary", response_model=ForecastSummary)
def get_urea_summary(
    response: Response,
    store: CacheStore = Depends(get_cache_store),
) -> ForecastSummary:
    payload = store.read()
    _set_cache_headers(response, store)
    return ForecastSummary(
        generated_at=payload["generated_at"],
        as_of_date=payload["as_of_date"],
        forecast=_build_forecast(payload),
        signal=payload["signal"],
    )


@history_router.get("/urea", response_model=LabeledSeries)
def get_urea_history(
    response: Response,
    store: CacheStore = Depends(get_cache_store),
) -> LabeledSeries:
    payload = store.read()
    _set_cache_headers(response, store)
    return LabeledSeries(**payload["urea_history"])


@history_router.get("/natgas", response_model=LabeledSeries)
def get_natgas_history(
    response: Response,
    store: CacheStore = Depends(get_cache_store),
) -> LabeledSeries:
    payload = store.read()
    _set_cache_headers(response, store)
    return LabeledSeries(**payload["natgas_history"])


@mc_router.get("/urea", response_model=MonteCarloBundle)
def get_urea_monte_carlo(
    response: Response,
    store: CacheStore = Depends(get_cache_store),
) -> MonteCarloBundle:
    payload = store.read()
    _set_cache_headers(response, store)
    return MonteCarloBundle(
        monte_carlo=payload["monte_carlo"],
        sim_t2_distribution=payload["sim_t2_distribution"],
    )
