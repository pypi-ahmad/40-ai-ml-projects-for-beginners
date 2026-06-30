"""SHAP-based local explanation endpoint."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from loguru import logger

from app.config import settings
from app.models.predictor import predictor
from app.models.schemas import ApiError, ExplainResponse, HousingFeatures
from app.services.validation import validate_record

router = APIRouter(tags=["explain"])


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Explain Prediction",
    description="Return local SHAP contributions for one California Housing record.",
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ApiError, "description": "Missing/invalid API key."},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ApiError, "description": "Validation error."},
        status.HTTP_429_TOO_MANY_REQUESTS: {"model": ApiError, "description": "Rate limit exceeded."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ApiError, "description": "Explainer failure."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ApiError, "description": "Model not available."},
    },
)
async def explain(body: HousingFeatures, request: Request) -> ExplainResponse:
    """Generate SHAP contributions for a single inference record."""
    if not predictor.is_loaded:
        try:
            predictor.load()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail="Model not loaded.") from exc

    try:
        payload = validate_record(body)
        (
            prediction,
            base_value,
            shap_values,
            contributions,
            explainer_type,
            explanation_note,
        ) = predictor.explain_one(payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("SHAP explain failed")
        raise HTTPException(status_code=500, detail=f"Explainability failure: {exc}") from exc

    return ExplainResponse(
        prediction=prediction,
        base_value=base_value,
        shap_values=shap_values,
        feature_contributions=contributions,
        explainer_type=explainer_type,
        explanation_note=explanation_note,
        model_name=str(predictor.metadata.get("model_name", "unknown")),
        model_version=str(predictor.metadata.get("model_version", "unknown")),
        feature_schema_version=settings.feature_schema_version,
        request_id=getattr(request.state, "request_id", None),
    )
