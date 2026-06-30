"""Prediction endpoints for single and batch inference."""
from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request, status

from app.config import settings
from app.models.predictor import predictor
from app.models.schemas import (
    ApiError,
    BatchPredictRequest,
    BatchPredictResponse,
    HousingFeatures,
    PredictResponse,
)
from app.services.validation import validate_batch_size, validate_record

router = APIRouter(tags=["predict"])


def _ensure_model_loaded() -> None:
    if predictor.is_loaded:
        return
    try:
        predictor.load()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model not loaded.") from exc


COMMON_ERROR_RESPONSES = {
    status.HTTP_401_UNAUTHORIZED: {"model": ApiError, "description": "Missing/invalid API key."},
    status.HTTP_413_CONTENT_TOO_LARGE: {"model": ApiError, "description": "Payload too large."},
    status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ApiError, "description": "Validation error."},
    status.HTTP_429_TOO_MANY_REQUESTS: {"model": ApiError, "description": "Rate limit exceeded."},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ApiError, "description": "Model not available."},
}


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict Single",
    description="Return one regression prediction for a California Housing feature record.",
    responses=COMMON_ERROR_RESPONSES,
)
async def predict_single(body: HousingFeatures, request: Request) -> PredictResponse:
    """Run single-record inference with strict schema validation."""
    _ensure_model_loaded()
    payload = validate_record(body)

    start = time.perf_counter()
    prediction = predictor.predict_one(payload)
    latency_ms = (time.perf_counter() - start) * 1000.0

    return PredictResponse(
        prediction=prediction,
        model_name=str(predictor.metadata.get("model_name", "unknown")),
        model_version=str(predictor.metadata.get("model_version", "unknown")),
        feature_schema_version=settings.feature_schema_version,
        request_id=getattr(request.state, "request_id", None),
        latency_ms=latency_ms,
    )


@router.post(
    "/predict-batch",
    response_model=BatchPredictResponse,
    summary="Predict Batch",
    description="Return predictions for a batch of California Housing records in one request.",
    responses=COMMON_ERROR_RESPONSES,
)
async def predict_batch(body: BatchPredictRequest, request: Request) -> BatchPredictResponse:
    """Run batch inference for multiple records in one request."""
    _ensure_model_loaded()
    validate_batch_size(len(body.records))

    records = [validate_record(record) for record in body.records]

    start = time.perf_counter()
    predictions = predictor.predict_batch(records)
    latency_ms = (time.perf_counter() - start) * 1000.0
    throughput = len(records) / (latency_ms / 1000.0) if latency_ms > 0 else 0.0

    return BatchPredictResponse(
        predictions=predictions,
        n_records=len(records),
        model_name=str(predictor.metadata.get("model_name", "unknown")),
        model_version=str(predictor.metadata.get("model_version", "unknown")),
        feature_schema_version=settings.feature_schema_version,
        request_id=getattr(request.state, "request_id", None),
        latency_ms=latency_ms,
        throughput_records_per_second=throughput,
    )
