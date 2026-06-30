"""HTTP endpoints for model serving and observability."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request

from ml_api.core.config import Settings
from ml_api.core.errors import BatchLimitError
from ml_api.core.metrics import MetricsStore
from ml_api.schemas.prediction import BatchPredictionRequest, ExplainRequest, PredictionRequest
from ml_api.schemas.responses import (
    BatchPredictionItem,
    BatchPredictionResponse,
    ExplainContribution,
    ExplainResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from ml_api.services.dependencies import get_metrics_store, get_model_service, get_settings
from ml_api.services.model_service import ModelService

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(
    request: Request,
    settings: Settings = Depends(get_settings),
    model_service: ModelService = Depends(get_model_service),
) -> HealthResponse:
    return HealthResponse(
        status="ok" if model_service.is_ready else "degraded",
        app_name=settings.app_name,
        app_version=settings.app_version,
        uptime_seconds=round(time.time() - request.app.state.started_at, 3),
        model_loaded=model_service.is_ready,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info(model_service: ModelService = Depends(get_model_service)) -> ModelInfoResponse:
    return ModelInfoResponse(**model_service.model_info())


@router.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(
    payload: PredictionRequest,
    request: Request,
    model_service: ModelService = Depends(get_model_service),
) -> PredictionResponse:
    started = time.perf_counter()
    prediction = model_service.predict_one(payload.record)
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    info = model_service.model_info()

    return PredictionResponse(
        prediction=prediction,
        model_name=info["model_name"],
        model_version=info["model_version"],
        latency_ms=latency_ms,
        request_id=request.state.request_id,
    )


@router.post("/predict-batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    model_service: ModelService = Depends(get_model_service),
) -> BatchPredictionResponse:
    if len(payload.records) > settings.max_batch_size:
        raise BatchLimitError(max_batch_size=settings.max_batch_size, requested=len(payload.records))

    started = time.perf_counter()
    predictions = model_service.predict_batch(payload.records)
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    throughput = round((len(predictions) / max(latency_ms, 1e-6)) * 1000, 3)

    return BatchPredictionResponse(
        count=len(predictions),
        predictions=[BatchPredictionItem(index=i, prediction=value) for i, value in enumerate(predictions)],
        latency_ms=latency_ms,
        throughput_rows_per_sec=throughput,
        request_id=request.state.request_id,
    )


@router.get("/metrics", response_model=MetricsResponse, tags=["observability"])
def metrics(
    metrics_store: MetricsStore = Depends(get_metrics_store),
    model_service: ModelService = Depends(get_model_service),
) -> MetricsResponse:
    metrics_payload = metrics_store.snapshot()
    info = model_service.model_info()
    return MetricsResponse(
        **metrics_payload,
        model_name=info["model_name"],
        model_version=info["model_version"],
    )


@router.post("/explain", response_model=ExplainResponse, tags=["explainability"])
def explain(
    payload: ExplainRequest,
    request: Request,
    model_service: ModelService = Depends(get_model_service),
) -> ExplainResponse:
    explanation = model_service.explain(payload.record, top_k=payload.top_k)
    return ExplainResponse(
        prediction=explanation["prediction"],
        base_value=explanation["base_value"],
        method=explanation["method"],
        top_k=payload.top_k,
        contributions=[ExplainContribution(**item) for item in explanation["contributions"]],
        request_id=request.state.request_id,
    )
