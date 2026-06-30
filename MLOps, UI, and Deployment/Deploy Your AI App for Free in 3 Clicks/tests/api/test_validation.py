"""Validation and error-contract tests for schema robustness."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ml_api.schemas.prediction import HouseFeatures
from tests.api.payloads import sample_record


def test_house_features_validates_expected_payload():
    model = HouseFeatures(**sample_record())
    assert model.overall_qual == 7


def test_house_features_rejects_out_of_range():
    payload = sample_record()
    payload["overall_qual"] = 11
    with pytest.raises(ValidationError):
        HouseFeatures(**payload)


def test_house_features_rejects_remod_before_build():
    payload = sample_record()
    payload["year_built"] = 2005
    payload["year_remod_add"] = 2001
    with pytest.raises(ValidationError):
        HouseFeatures(**payload)


def test_house_features_forbids_unknown_fields():
    payload = sample_record()
    payload["unknown_feature"] = "bad"
    with pytest.raises(ValidationError):
        HouseFeatures(**payload)
