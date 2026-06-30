"""Unit tests for SQLite metrics store behavior."""
from __future__ import annotations

from datetime import UTC, datetime

from app.services.metrics_store import MetricsStore, RequestMetricEvent


def test_metrics_store_summary_contains_uptime(tmp_path) -> None:
    store = MetricsStore(tmp_path / "metrics.db")
    store.init()

    store.record(
        RequestMetricEvent(
            timestamp_utc=datetime.now(UTC).isoformat(),
            endpoint="/predict",
            method="POST",
            status_code=200,
            latency_ms=12.4,
            request_id="req-1",
        )
    )
    store.record(
        RequestMetricEvent(
            timestamp_utc=datetime.now(UTC).isoformat(),
            endpoint="/predict",
            method="POST",
            status_code=500,
            latency_ms=30.1,
            request_id="req-2",
        )
    )

    summary = store.summary()
    assert summary["total_requests"] == 2
    assert summary["total_errors"] == 1
    assert summary["uptime_seconds"] >= 0.0
    assert summary["p95_latency_ms"] >= summary["p50_latency_ms"]
