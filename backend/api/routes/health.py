"""GET /health — cache freshness + data-extent reporting."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from api.cache_store import CacheStore, get_cache_store
from api.schemas import HealthResponse

router = APIRouter()


def _data_through(payload: dict, anchor: str) -> str:
    """Derive YYYY-MM of the last month present in either history series. With
    today's producer this equals the anchor (run_pipeline intersects NG and urea),
    but the field is reported separately so the contract survives a producer
    change that exposes trailing NG-only months."""
    candidates: list[str] = []
    for key in ("natgas_history", "urea_history"):
        labels = payload.get(key, {}).get("labels") or []
        if labels:
            try:
                candidates.append(datetime.strptime(labels[-1], "%b %Y").strftime("%Y-%m"))
            except ValueError:
                pass
    return max(candidates) if candidates else anchor


@router.get("/health", response_model=HealthResponse)
def health(store: CacheStore = Depends(get_cache_store)) -> HealthResponse:
    payload = store.read()
    age = int((datetime.now(timezone.utc) - store.mtime()).total_seconds())
    anchor = payload["as_of_date"]
    return HealthResponse(
        status="ok",
        cache_age_seconds=age,
        as_of_date=anchor,
        data_through=_data_through(payload, anchor),
    )
