"""FastAPI application for the Foregast forecast service.

Read-only consumer of data/processed/cache.json — never recomputes per request.
The pipeline (run_pipeline.py) is the producer; this module never imports it.

Run (from backend/):
    uvicorn api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.cache_store import CacheUnavailable
from api.routes import forecast as forecast_routes
from api.routes import health as health_routes

app = FastAPI(title="Foregast Forecast API", version="1.0.0")

# CORS — permissive for local development only. MUST be locked down before any
# public deploy: replace allow_origins=["*"] with the explicit frontend origin(s),
# and keep allow_credentials=False unless cookie/auth flows require otherwise.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.exception_handler(CacheUnavailable)
async def cache_unavailable_handler(_: Request, exc: CacheUnavailable) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc), "code": "cache_unavailable"},
        headers={"Retry-After": "5"},
    )


app.include_router(health_routes.router)
app.include_router(forecast_routes.router)
app.include_router(forecast_routes.history_router)
app.include_router(forecast_routes.mc_router)
