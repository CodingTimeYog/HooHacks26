"""API contract tests — swap a fake CacheStore via app.dependency_overrides
(the hook cache_store.get_cache_store already exposes) and assert the wire
shape of /forecast/urea, /health, and the 503 fallback."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.cache_store import CacheUnavailable, get_cache_store


@pytest.fixture
def client(fake_cache_store):
    """Client with the fake store wired in. Cleans overrides on teardown so
    the next test starts fresh."""
    app.dependency_overrides[get_cache_store] = lambda: fake_cache_store
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_forecast_urea_returns_200_and_typed_shape(client):
    r = client.get("/forecast/urea")
    assert r.status_code == 200, r.text
    body = r.json()

    # Envelope
    assert body["schema_version"] == 1
    assert body["as_of_date"] == "2026-06"

    # Anchor + N derives ISO months alongside display labels
    assert body["forecast"]["months"] == ["2026-07", "2026-08", "2026-09"]
    assert body["forecast"]["labels"] == ["Jul 2026", "Aug 2026", "Sep 2026"]

    # Signal fields are normalised to snake_case on the wire even though the
    # on-disk cache uses camelCase (currentPrice / bestMonth / bestPrice).
    sig = body["signal"]
    assert sig["current_price"] == 655.0
    assert sig["best_month"]    == "In 2 months"
    assert sig["best_price"]    == 670.0
    assert "currentPrice" not in sig, "wire format must be snake_case, not aliased"
    assert "bestMonth"    not in sig
    assert "bestPrice"    not in sig


def test_health_returns_expected_fields(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()

    assert body["status"] == "ok"
    assert body["schema_version"] == 1
    assert body["as_of_date"]   == "2026-06"
    assert body["data_through"] == "2026-06"
    assert isinstance(body["cache_age_seconds"], int)


def test_cache_unavailable_returns_503_with_retry_after():
    """A CacheStore that raises CacheUnavailable must surface as 503 with a
    Retry-After header — the exception handler in main.py owns this contract,
    routes shouldn't crash the process."""
    class BrokenStore:
        def read(self):
            raise CacheUnavailable("test: cache missing")
        def mtime(self):
            raise CacheUnavailable("test: cache missing")

    app.dependency_overrides[get_cache_store] = lambda: BrokenStore()
    try:
        client = TestClient(app)
        r = client.get("/forecast/urea")
        assert r.status_code == 503, r.text
        assert r.headers.get("Retry-After") == "5"
        assert r.json()["code"] == "cache_unavailable"
    finally:
        app.dependency_overrides.clear()
