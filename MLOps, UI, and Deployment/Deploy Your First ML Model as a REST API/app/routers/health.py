"""Health-check and optional admin model reload endpoints."""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.models.predictor import predictor
from app.models.schemas import ApiError, HealthResponse, ReadinessChecks, ReloadResponse
from app.services.metrics_store import metrics_store

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Return liveness and readiness checks for API runtime and model artifacts.",
)
async def health_check() -> HealthResponse:
    """Return API liveness/readiness with model and metrics DB status."""
    if not predictor.is_loaded:
        try:
            predictor.load()
        except FileNotFoundError:
            pass

    now = datetime.now(UTC)
    started_at = metrics_store.started_at_utc or now
    uptime_seconds = max((now - started_at).total_seconds(), 0.0)

    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        app_version=settings.app_version,
        model_loaded=predictor.is_loaded,
        uptime_seconds=uptime_seconds,
        checks=ReadinessChecks(
            metrics_db_ready=metrics_store.is_ready,
            model_metadata_ready=settings.metadata_path.is_file(),
        ),
        timestamp_utc=now,
    )


@router.post(
    "/admin/reload",
    response_model=ReloadResponse,
    summary="Reload Model",
    description="Reload model and metadata from disk. Protected by API key when API_KEY is set.",
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ApiError, "description": "Missing/invalid API key."},
        status.HTTP_403_FORBIDDEN: {"model": ApiError, "description": "Reload endpoint disabled."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ApiError, "description": "Model files unavailable."},
    },
)
async def reload_model() -> ReloadResponse:
    """Reload model artifact from disk for local operational workflows."""
    if not settings.enable_reload_endpoint:
        raise HTTPException(status_code=403, detail="Model reload endpoint is disabled.")

    try:
        predictor.reload()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model files not found during reload.") from exc

    return ReloadResponse(
        status="ok",
        model_loaded=predictor.is_loaded,
        model_name=str(predictor.metadata.get("model_name")) if predictor.is_loaded else None,
    )
