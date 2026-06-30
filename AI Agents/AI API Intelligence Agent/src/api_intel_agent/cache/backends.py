"""Cache backends for API payloads, embeddings, and reports."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import redis
except Exception:  # pragma: no cover - optional dependency at runtime
    redis = None

from api_intel_agent.config import load_settings


@dataclass(slots=True)
class CacheValue:
    value: Any
    expires_at: float


class SQLiteCache:
    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_items (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
                """
            )

    def get(self, namespace: str, key: str) -> Any | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT value, expires_at FROM cache_items WHERE namespace=? AND key=?",
                (namespace, key),
            ).fetchone()
            if not row:
                return None
            value, expires_at = row
            if expires_at < time.time():
                conn.execute(
                    "DELETE FROM cache_items WHERE namespace=? AND key=?",
                    (namespace, key),
                )
                return None
            return json.loads(value)

    def set(self, namespace: str, key: str, value: Any, ttl_seconds: int) -> None:
        expires_at = time.time() + ttl_seconds
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO cache_items(namespace, key, value, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, key)
                DO UPDATE SET value=excluded.value, expires_at=excluded.expires_at
                """,
                (namespace, key, json.dumps(value), expires_at),
            )


class RedisCache:
    def __init__(self, url: str) -> None:
        if redis is None:
            raise RuntimeError("redis package not installed")
        self.client = redis.Redis.from_url(url, decode_responses=True)

    def get(self, namespace: str, key: str) -> Any | None:
        payload = self.client.get(f"{namespace}:{key}")
        if payload is None:
            return None
        return json.loads(payload)

    def set(self, namespace: str, key: str, value: Any, ttl_seconds: int) -> None:
        self.client.setex(f"{namespace}:{key}", ttl_seconds, json.dumps(value))


class CacheManager:
    def __init__(self) -> None:
        self.settings = load_settings()
        if self.settings.cache.backend == "redis" and self.settings.cache.redis_url:
            self.backend = RedisCache(self.settings.cache.redis_url)
        else:
            self.backend = SQLiteCache(self.settings.cache.sqlite_path)

    def get(self, namespace: str, key: str) -> Any | None:
        return self.backend.get(namespace, key)

    def set(self, namespace: str, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        ttl = ttl_seconds or self.settings.cache.default_ttl_seconds
        self.backend.set(namespace, key, value, ttl)
