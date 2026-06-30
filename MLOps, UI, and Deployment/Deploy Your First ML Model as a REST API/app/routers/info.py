"""Model metadata endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.models.predictor import predictor
from app.models.schemas import ModelInfoResponse

router = APIRouter(tags=["model-info"])


def _load_if_possible() -> None:
    if predictor.is_loaded:
        return
    try:
        predictor.load()
    except FileNotFoundError:
        return


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model Info",
    description="Return model card metadata and serving feature schema version.",
)
async def model_info() -> ModelInfoResponse:
    """Return model metadata, feature schema version, and load status."""
    _load_if_possible()
    meta = predictor.metadata if predictor.is_loaded else {}

    if predictor.is_loaded and "feature_schema_version" not in meta:
        meta = dict(meta)
        meta["feature_schema_version"] = settings.feature_schema_version

    return ModelInfoResponse(
        app_name=settings.app_name,
        app_version=settings.app_version,
        model_loaded=predictor.is_loaded,
        feature_schema_version=settings.feature_schema_version,
        metadata=meta,
    )


@router.get("/info", response_model=ModelInfoResponse, deprecated=True)
async def model_info_alias() -> ModelInfoResponse:
    """Backward-compatible alias retained for existing integrations."""
    return await model_info()
