"""Shared fixtures for API integration tests."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from sklearn.ensemble import RandomForestRegressor

from app.config import settings
from app.main import app
from app.models.predictor import predictor
from app.services.metrics_store import metrics_store
from src.constants import FEATURE_NAMES


@pytest.fixture(autouse=True)
def _clean_state(tmp_path: Path):
    """Reset singleton predictor/settings and metrics paths between tests."""
    predictor.unload()

    original_model_path = settings.model_path
    original_metadata_path = settings.metadata_path
    original_metrics_path = settings.metrics_db_path
    original_api_key = settings.api_key
    original_max_batch_size = settings.max_batch_size
    original_max_body_bytes = settings.max_request_body_bytes
    original_schema_version = settings.feature_schema_version

    test_metrics_path = tmp_path / "api_metrics.db"
    settings.metrics_db_path = test_metrics_path
    metrics_store.db_path = test_metrics_path
    metrics_store._ready = False
    metrics_store._started_at_utc = None
    metrics_store.init()

    yield

    settings.model_path = original_model_path
    settings.metadata_path = original_metadata_path
    settings.metrics_db_path = original_metrics_path
    settings.api_key = original_api_key
    settings.max_batch_size = original_max_batch_size
    settings.max_request_body_bytes = original_max_body_bytes
    settings.feature_schema_version = original_schema_version

    metrics_store.db_path = original_metrics_path
    metrics_store._ready = False
    metrics_store._started_at_utc = None


@pytest.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def sample_payload() -> dict[str, float]:
    return {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.9841,
        "AveBedrms": 1.0238,
        "Population": 322.0,
        "AveOccup": 2.5556,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }


@pytest.fixture
def dummy_model_dir(tmp_path: Path) -> Path:
    """Train quick regression model and write matching metadata artifacts."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 8))
    y = X[:, 0] * 1.5 - X[:, 1] * 0.5 + rng.normal(scale=0.2, size=120)
    X_df = pd.DataFrame(X, columns=FEATURE_NAMES)

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_df, y)

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    meta = {
        "model_name": "test-random-forest",
        "model_version": "0.0.0-test",
        "problem_type": "regression",
        "feature_names": FEATURE_NAMES,
        "feature_schema_version": "california-housing-v1",
        "mae": 0.12,
        "mse": 0.03,
        "rmse": 0.17,
        "r2": 0.91,
        "mape": 0.14,
        "n_train": 80,
        "n_val": 20,
        "n_test": 20,
    }
    meta_path = tmp_path / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f)

    settings.model_path = model_path
    settings.metadata_path = meta_path

    return tmp_path
