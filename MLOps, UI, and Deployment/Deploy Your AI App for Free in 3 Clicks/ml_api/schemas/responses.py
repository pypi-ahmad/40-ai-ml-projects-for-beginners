"""Pydantic response schemas for API contract consistency."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app_name: str
    app_version: str
    uptime_seconds: float
    model_loaded: bool
    timestamp_utc: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    model_version: str
    training_rows: int
    feature_count: int
    target_column: str
    dataset_hash: str
    validation_rmse: float
    test_rmse: float
    serialization_formats: list[str]


class PredictionResponse(BaseModel):
    prediction: float
    model_name: str
    model_version: str
    latency_ms: float
    request_id: str


class BatchPredictionItem(BaseModel):
    index: int
    prediction: float


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[BatchPredictionItem]
    latency_ms: float
    throughput_rows_per_sec: float
    request_id: str


class ExplainContribution(BaseModel):
    feature: str
    value: float | str
    contribution: float


class ExplainResponse(BaseModel):
    prediction: float
    base_value: float
    method: str
    top_k: int
    contributions: list[ExplainContribution]
    request_id: str


class MetricsResponse(BaseModel):
    uptime_seconds: float
    request_counts: dict[str, int]
    status_counts: dict[int, int]
    latency_ms: dict[str, float]
    route_latency_ms: dict[str, dict[str, float]]
    model_name: str
    model_version: str
