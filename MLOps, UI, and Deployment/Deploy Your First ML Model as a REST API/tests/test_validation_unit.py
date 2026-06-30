"""Unit tests for validation service helpers."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.models.schemas import HousingFeatures
from app.services.validation import validate_batch_size, validate_record


def test_validate_record_accepts_valid_payload() -> None:
    record = HousingFeatures(
        MedInc=8.3252,
        HouseAge=41.0,
        AveRooms=6.9841,
        AveBedrms=1.0238,
        Population=322.0,
        AveOccup=2.5556,
        Latitude=37.88,
        Longitude=-122.23,
    )
    payload = validate_record(record)
    assert payload["MedInc"] == 8.3252


def test_validate_record_rejects_nan() -> None:
    # model_construct bypasses schema validators so we can explicitly test finite guards.
    record = HousingFeatures.model_construct(
        MedInc=float("nan"),
        HouseAge=41.0,
        AveRooms=6.9841,
        AveBedrms=1.0238,
        Population=322.0,
        AveOccup=2.5556,
        Latitude=37.88,
        Longitude=-122.23,
    )
    with pytest.raises(HTTPException) as exc:
        validate_record(record)
    assert exc.value.status_code == 422


def test_validate_batch_size_rejects_zero() -> None:
    with pytest.raises(HTTPException):
        validate_batch_size(0)
