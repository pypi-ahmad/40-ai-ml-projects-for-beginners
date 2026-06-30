"""Prometheus metrics for API traffic and model-inference observability."""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import time

PREDICT_COUNTER = Counter(
    "ml_predictions_total", "Total prediction requests", ["endpoint", "status"]
)
PREDICT_LATENCY = Histogram(
    "ml_prediction_latency_seconds", "Prediction latency", ["endpoint"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0),
)
ERROR_COUNTER = Counter(
    "ml_errors_total", "API errors", ["endpoint", "status_code"]
)
INFLIGHT_REQUESTS = Gauge("ml_inflight_requests", "In-flight API requests")


class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect request latency and status metrics for all API routes."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
        start = time.monotonic()
        INFLIGHT_REQUESTS.inc()
        response = await call_next(request)
        try:
            elapsed = time.monotonic() - start
            endpoint = request.url.path
            PREDICT_LATENCY.labels(endpoint=endpoint).observe(elapsed)
            if response.status_code >= 400:
                ERROR_COUNTER.labels(endpoint=endpoint, status_code=response.status_code).inc()
            status = "ok" if response.status_code < 400 else "error"
            PREDICT_COUNTER.labels(endpoint=endpoint, status=status).inc()
            return response
        finally:
            INFLIGHT_REQUESTS.dec()


def metrics_endpoint(request: Request) -> Response:
    """Return Prometheus exposition format output."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
