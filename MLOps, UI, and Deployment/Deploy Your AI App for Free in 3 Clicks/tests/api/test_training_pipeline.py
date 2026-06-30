"""Training pipeline and serialization integrity tests."""

from __future__ import annotations

from pathlib import Path

from ml_api.core.config import get_settings
from ml_api.services.model_service import ModelService
from ml_api.training.models import build_model_catalog
from ml_api.training.pipeline import train_and_serialize


def test_model_catalog_includes_required_core_models():
    catalog = build_model_catalog(random_seed=42)
    required = {
        "linear_regression",
        "ridge",
        "lasso",
        "elasticnet",
        "random_forest",
        "extra_trees",
    }
    assert required.issubset(catalog.keys())


def test_train_and_serialize_writes_artifacts():
    get_settings.cache_clear()
    settings = get_settings()
    outputs = train_and_serialize(settings)

    assert Path(settings.model_dir / settings.joblib_artifact).exists()
    assert Path(settings.model_dir / settings.pickle_artifact).exists()
    assert Path(settings.model_dir / settings.metadata_artifact).exists()
    assert outputs.metadata.validation_rmse > 0
    assert outputs.metadata.test_rmse > 0


def test_reload_stability_check():
    get_settings.cache_clear()
    settings = get_settings()
    train_and_serialize(settings)

    svc = ModelService(settings)
    svc.load()
    result = svc.reload_stability_check(cycles=3)
    assert result["cycles"] == 3
    assert result["drift"] >= 0
