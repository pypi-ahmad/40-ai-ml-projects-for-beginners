"""Pydantic schemas for request/response contracts."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

CALIFORNIA_EXAMPLE = {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.9841,
    "AveBedrms": 1.0238,
    "Population": 322.0,
    "AveOccup": 2.5556,
    "Latitude": 37.88,
    "Longitude": -122.23,
}


class FieldError(BaseModel):
    """Field-level validation error payload."""

    field: str
    message: str


class ApiError(BaseModel):
    """Standard error envelope for all API failures."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": "VALIDATION_ERROR",
                "detail": "Request validation failed.",
                "request_id": "3c3decd3-5b4e-4ec1-b9f7-0fc5004e2df1",
                "field_errors": [{"field": "Latitude", "message": "Field required"}],
            }
        }
    )

    code: str
    detail: str
    request_id: str | None = None
    field_errors: list[FieldError] | None = None


class HousingFeatures(BaseModel):
    """Named California Housing input features with range validation."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": CALIFORNIA_EXAMPLE})

    MedInc: float = Field(..., ge=0.0, le=30.0, description="Median income in block group.")
    HouseAge: float = Field(..., ge=0.0, le=100.0, description="Median house age.")
    AveRooms: float = Field(..., gt=0.0, le=100.0, description="Average rooms per household.")
    AveBedrms: float = Field(..., gt=0.0, le=20.0, description="Average bedrooms per household.")
    Population: float = Field(..., gt=0.0, le=100000.0, description="Population in block group.")
    AveOccup: float = Field(..., gt=0.0, le=100.0, description="Average household occupancy.")
    Latitude: float = Field(..., ge=32.0, le=43.0, description="Latitude coordinate.")
    Longitude: float = Field(..., ge=-125.0, le=-113.0, description="Longitude coordinate.")


class BatchPredictRequest(BaseModel):
    """Batch prediction payload."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"records": [CALIFORNIA_EXAMPLE, CALIFORNIA_EXAMPLE]}},
    )
    records: list[HousingFeatures] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    """Response schema for single prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 4.3477,
                "model_name": "LightGBM",
                "model_version": "1.0.0",
                "feature_schema_version": "california-housing-v1",
                "request_id": "3c3decd3-5b4e-4ec1-b9f7-0fc5004e2df1",
                "latency_ms": 12.41,
            }
        }
    )

    prediction: float = Field(..., description="Predicted median house value.")
    model_name: str = Field(..., description="Model identifier used for inference.")
    model_version: str = Field(..., description="Model version string.")
    feature_schema_version: str = Field(..., description="Feature schema version used by model.")
    request_id: str | None = Field(None, description="Correlation request ID.")
    latency_ms: float = Field(..., ge=0.0, description="Request latency in milliseconds.")


class BatchPredictResponse(BaseModel):
    """Response schema for batch prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": [4.31, 2.18],
                "n_records": 2,
                "model_name": "LightGBM",
                "model_version": "1.0.0",
                "feature_schema_version": "california-housing-v1",
                "request_id": "3c3decd3-5b4e-4ec1-b9f7-0fc5004e2df1",
                "latency_ms": 9.88,
                "throughput_records_per_second": 202.5,
            }
        }
    )

    predictions: list[float] = Field(..., description="Predictions aligned with input record order.")
    n_records: int = Field(..., ge=1, description="Number of inferred records.")
    model_name: str = Field(...)
    model_version: str = Field(...)
    feature_schema_version: str = Field(...)
    request_id: str | None = Field(None)
    latency_ms: float = Field(..., ge=0.0)
    throughput_records_per_second: float = Field(..., ge=0.0)


class ExplainResponse(BaseModel):
    """Response schema for local SHAP explanation."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 4.3477,
                "base_value": 2.0690,
                "shap_values": [1.82, 0.07, 0.22, 0.01, -0.05, 0.04, -0.30, 0.46],
                "feature_contributions": {
                    "MedInc": 1.82,
                    "HouseAge": 0.07,
                    "AveRooms": 0.22,
                    "AveBedrms": 0.01,
                    "Population": -0.05,
                    "AveOccup": 0.04,
                    "Latitude": -0.30,
                    "Longitude": 0.46,
                },
                "explainer_type": "TreeExplainer",
                "explanation_note": None,
                "model_name": "LightGBM",
                "model_version": "1.0.0",
                "feature_schema_version": "california-housing-v1",
                "request_id": "3c3decd3-5b4e-4ec1-b9f7-0fc5004e2df1",
            }
        }
    )

    prediction: float = Field(...)
    base_value: float = Field(...)
    shap_values: list[float] = Field(...)
    feature_contributions: dict[str, float] = Field(...)
    explainer_type: str = Field(..., description="SHAP explainer backend selected at runtime.")
    explanation_note: str | None = Field(None, description="Optional note about explainer fallback path.")
    model_name: str = Field(...)
    model_version: str = Field(...)
    feature_schema_version: str = Field(...)
    request_id: str | None = Field(None)


class ReadinessChecks(BaseModel):
    """Health readiness checks."""

    metrics_db_ready: bool
    model_metadata_ready: bool


class HealthResponse(BaseModel):
    """Health check payload."""

    status: str
    app_name: str
    app_version: str
    model_loaded: bool
    uptime_seconds: float
    checks: ReadinessChecks
    timestamp_utc: datetime


class EndpointMetric(BaseModel):
    """Aggregated endpoint metrics row."""

    endpoint: str
    requests: int
    errors: int
    avg_latency_ms: float


class MetricsResponse(BaseModel):
    """Response schema for platform + model metrics."""

    total_requests: int
    total_errors: int
    error_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    throughput_rps_last_minute: float
    uptime_seconds: float
    by_endpoint: list[EndpointMetric]
    model_name: str | None = None
    model_version: str | None = None
    feature_schema_version: str | None = None
    mae: float | None = None
    mse: float | None = None
    rmse: float | None = None
    r2: float | None = None
    mape: float | None = None


class ModelInfoResponse(BaseModel):
    """Response schema for model metadata endpoint."""

    app_name: str
    app_version: str
    model_loaded: bool
    feature_schema_version: str
    metadata: dict


class ReloadResponse(BaseModel):
    """Admin reload payload."""

    status: str
    model_loaded: bool
    model_name: str | None = None
