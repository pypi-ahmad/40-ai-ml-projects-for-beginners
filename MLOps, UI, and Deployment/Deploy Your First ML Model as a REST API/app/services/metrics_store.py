"""SQLite-backed API metrics storage."""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.config import settings


@dataclass(slots=True)
class RequestMetricEvent:
    """Single request event persisted by middleware."""

    timestamp_utc: str
    endpoint: str
    method: str
    status_code: int
    latency_ms: float
    request_id: str


class MetricsStore:
    """Thread-safe SQLite writer/reader for API monitoring."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._ready = False
        self._started_at_utc: datetime | None = None

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def started_at_utc(self) -> datetime | None:
        return self._started_at_utc

    def init(self) -> None:
        """Create schema if missing."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS request_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp_utc TEXT NOT NULL,
                        endpoint TEXT NOT NULL,
                        method TEXT NOT NULL,
                        status_code INTEGER NOT NULL,
                        latency_ms REAL NOT NULL,
                        request_id TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_request_metrics_timestamp
                    ON request_metrics(timestamp_utc)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_request_metrics_endpoint
                    ON request_metrics(endpoint)
                    """
                )
                conn.commit()

        self._ready = True
        if self._started_at_utc is None:
            self._started_at_utc = datetime.now(UTC)

    def record(self, event: RequestMetricEvent) -> None:
        """Persist one request metric event."""
        if not self._ready:
            self.init()
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO request_metrics (
                        timestamp_utc,
                        endpoint,
                        method,
                        status_code,
                        latency_ms,
                        request_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.timestamp_utc,
                        event.endpoint,
                        event.method,
                        event.status_code,
                        event.latency_ms,
                        event.request_id,
                    ),
                )
                conn.commit()

    def summary(self) -> dict:
        """Return aggregate API metrics used by `/metrics` endpoint."""
        if not self._ready:
            self.init()

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                totals = conn.execute(
                    """
                    SELECT
                        COUNT(*) AS total_requests,
                        SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) AS total_errors,
                        AVG(latency_ms) AS avg_latency_ms
                    FROM request_metrics
                    """
                ).fetchone()

                percentile_rows = conn.execute(
                    "SELECT latency_ms FROM request_metrics ORDER BY latency_ms"
                ).fetchall()
                latencies = [float(row["latency_ms"]) for row in percentile_rows]

                endpoint_rows = conn.execute(
                    """
                    SELECT
                        endpoint,
                        COUNT(*) AS requests,
                        SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) AS errors,
                        AVG(latency_ms) AS avg_latency_ms
                    FROM request_metrics
                    GROUP BY endpoint
                    ORDER BY requests DESC
                    """
                ).fetchall()

                minute_ago = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
                last_minute = conn.execute(
                    """
                    SELECT COUNT(*) AS requests_last_minute
                    FROM request_metrics
                    WHERE timestamp_utc >= ?
                    """,
                    (minute_ago,),
                ).fetchone()

        total_requests = int(totals["total_requests"] or 0)
        total_errors = int(totals["total_errors"] or 0)

        p50 = 0.0
        p95 = 0.0
        if latencies:
            p50 = latencies[int(0.50 * (len(latencies) - 1))]
            p95 = latencies[int(0.95 * (len(latencies) - 1))]

        endpoint_metrics = [
            {
                "endpoint": str(row["endpoint"]),
                "requests": int(row["requests"] or 0),
                "errors": int(row["errors"] or 0),
                "avg_latency_ms": float(row["avg_latency_ms"] or 0.0),
            }
            for row in endpoint_rows
        ]

        started_at = self._started_at_utc or datetime.now(UTC)
        uptime_seconds = (datetime.now(UTC) - started_at).total_seconds()

        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": float(total_errors / total_requests) if total_requests else 0.0,
            "avg_latency_ms": float(totals["avg_latency_ms"] or 0.0),
            "p50_latency_ms": float(p50),
            "p95_latency_ms": float(p95),
            "throughput_rps_last_minute": float((last_minute["requests_last_minute"] or 0) / 60.0),
            "uptime_seconds": float(max(uptime_seconds, 0.0)),
            "by_endpoint": endpoint_metrics,
        }


metrics_store = MetricsStore(settings.metrics_db_path)
