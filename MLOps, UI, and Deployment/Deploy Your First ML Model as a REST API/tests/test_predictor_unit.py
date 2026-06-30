"""Unit tests for predictor load/reload compatibility behavior."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from app.config import settings
from app.models.predictor import predictor
from src.constants import FEATURE_NAMES


@pytest.fixture(autouse=True)
def _reset_predictor_state():
    predictor.unload()
    yield
    predictor.unload()


def _write_artifacts(tmp_path, *, schema_version: str) -> tuple:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 8))
    y = X[:, 0] - X[:, 1]
    X_df = pd.DataFrame(X, columns=FEATURE_NAMES)

    model = RandomForestRegressor(n_estimators=20, random_state=0)
    model.fit(X_df, y)

    model_path = tmp_path / "model.joblib"
    meta_path = tmp_path / "metadata.json"
    joblib.dump(model, model_path)

    metadata = {
        "model_name": "unit-rf",
        "model_version": "0.1.0",
        "feature_names": FEATURE_NAMES,
        "feature_schema_version": schema_version,
    }
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    return model_path, meta_path


def test_predictor_reload_cycle(tmp_path):
    model_path, meta_path = _write_artifacts(tmp_path, schema_version="california-housing-v1")

    settings.model_path = model_path
    settings.metadata_path = meta_path
    settings.feature_schema_version = "california-housing-v1"

    predictor.load()
    assert predictor.is_loaded

    sample = dict(zip(FEATURE_NAMES, [1.0] * len(FEATURE_NAMES), strict=False))
    first = predictor.predict_one(sample)
    assert isinstance(first, float)

    for _ in range(3):
        predictor.reload()
        assert predictor.is_loaded
        pred = predictor.predict_one(sample)
        assert isinstance(pred, float)

    predictor.unload()
    assert not predictor.is_loaded


def test_predictor_schema_mismatch_rejected(tmp_path):
    model_path, meta_path = _write_artifacts(tmp_path, schema_version="other-schema")

    settings.model_path = model_path
    settings.metadata_path = meta_path
    settings.feature_schema_version = "california-housing-v1"

    with pytest.raises(RuntimeError):
        predictor.load()
