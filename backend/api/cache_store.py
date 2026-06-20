"""Read-only cache abstraction for the forecast API.

Keeps the route layer agnostic of where the cache lives: today it's a local
file at data/processed/cache.json; tomorrow it can be S3/GCS by swapping the
implementation. Parse is memoized on mtime so re-reads only happen when the
producer (run_pipeline.py) rewrites the file.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


_THIS = Path(__file__).resolve().parent
ROOT  = _THIS.parent.parent
DEFAULT_CACHE_PATH = ROOT / "data" / "processed" / "cache.json"


def _resolve_default_path() -> Path:
    """Honour the CACHE_PATH env var (set in the container), else fall back to
    the in-repo cache. Resolved at module import time."""
    env = os.environ.get("CACHE_PATH", "").strip()
    return Path(env) if env else DEFAULT_CACHE_PATH


class CacheUnavailable(RuntimeError):
    """Raised when the cache cannot be read or parsed (missing file, partial
    write caught mid-flight, corrupt JSON). Routes map this to HTTP 503."""


class CacheStore(Protocol):
    def read(self) -> dict: ...
    def mtime(self) -> datetime: ...


class LocalFileCacheStore:
    """Reads cache.json from disk; memoizes the parsed dict keyed on mtime."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else _resolve_default_path()
        self._cached_mtime: float | None = None
        self._cached_payload: dict | None = None

    def _stat_mtime(self) -> float:
        try:
            return self.path.stat().st_mtime
        except FileNotFoundError as exc:
            raise CacheUnavailable(f"Cache file not found: {self.path}") from exc

    def mtime(self) -> datetime:
        return datetime.fromtimestamp(self._stat_mtime(), tz=timezone.utc)

    def read(self) -> dict:
        ts = self._stat_mtime()
        if self._cached_payload is not None and self._cached_mtime == ts:
            return self._cached_payload
        try:
            with self.path.open(encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as exc:
            # run_pipeline.py writes non-atomically; a mid-write read can land here.
            raise CacheUnavailable(f"Cache file is not valid JSON: {exc}") from exc
        self._cached_mtime = ts
        self._cached_payload = payload
        return payload


_default_store = LocalFileCacheStore()


def get_cache_store() -> CacheStore:
    """FastAPI dependency. Tests override via app.dependency_overrides."""
    return _default_store
