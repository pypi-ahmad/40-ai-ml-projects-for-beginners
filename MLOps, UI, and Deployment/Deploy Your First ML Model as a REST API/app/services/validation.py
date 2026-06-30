"""Validation helpers layered on top of Pydantic models."""
from __future__ import annotations

import math

from fastapi import HTTPException

from app.config import settings
from app.models.schemas import HousingFeatures


def _assert_finite(value: float, field_name: str) -> None:
    """Reject NaN and infinite values with actionable messages."""
    if math.isnan(value) or math.isinf(value):
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be a finite numeric value.",
        )


def validate_record(record: HousingFeatures) -> dict[str, float]:
    """Return model-ready dict after finite-number guardrails."""
    payload = record.model_dump()
    for field_name, value in payload.items():
        _assert_finite(float(value), field_name)
    return {k: float(v) for k, v in payload.items()}


def validate_batch_size(batch_size: int) -> None:
    """Check batch length against configured safety limit."""
    if batch_size < 1:
        raise HTTPException(status_code=422, detail="Batch request must include at least one record.")
    if batch_size > settings.max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Batch size {batch_size} exceeds configured max_batch_size "
                f"({settings.max_batch_size})."
            ),
        )
