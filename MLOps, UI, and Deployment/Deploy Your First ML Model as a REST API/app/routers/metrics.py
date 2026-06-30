"""API + model metrics endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.models.predictor import predictor
from app.models.schemas import EndpointMetric, MetricsResponse
from app.services.metrics_store import metrics_store

router = APIRouter(tags=["metrics"])


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="API Metrics",
    description="Return operational API telemetry merged with model evaluation metrics.",
)
async def metrics() -> MetricsResponse:
    """Return operational API metrics merged with model evaluation metrics."""
    summary = metrics_store.summary()

    model_metrics = {}
    if predictor.is_loaded:
        model_metrics = predictor.metadata
    else:
        try:
            predictor.load()
            model_metrics = predictor.metadata
        except FileNotFoundError:
            model_metrics = {}

    endpoint_rows = [EndpointMetric(**row) for row in summary.get("by_endpoint", [])]

    return MetricsResponse(
        total_requests=int(summary.get("total_requests", 0)),
        total_errors=int(summary.get("total_errors", 0)),
        error_rate=float(summary.get("error_rate", 0.0)),
        avg_latency_ms=float(summary.get("avg_latency_ms", 0.0)),
        p50_latency_ms=float(summary.get("p50_latency_ms", 0.0)),
        p95_latency_ms=float(summary.get("p95_latency_ms", 0.0)),
        throughput_rps_last_minute=float(summary.get("throughput_rps_last_minute", 0.0)),
        uptime_seconds=float(summary.get("uptime_seconds", 0.0)),
        by_endpoint=endpoint_rows,
        model_name=model_metrics.get("model_name"),
        model_version=model_metrics.get("model_version"),
        feature_schema_version=model_metrics.get("feature_schema_version", settings.feature_schema_version),
        mae=model_metrics.get("mae"),
        mse=model_metrics.get("mse"),
        rmse=model_metrics.get("rmse"),
        r2=model_metrics.get("r2"),
        mape=model_metrics.get("mape"),
    )
