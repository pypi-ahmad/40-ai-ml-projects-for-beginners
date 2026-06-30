"""Fixtures for FastAPI endpoint contract tests without socket-bound test clients."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
import time

import pytest

from ml_api.app import create_app
from ml_api.core.config import get_settings
from ml_api.core.metrics import MetricsStore
from ml_api.services.model_service import ModelService
from ml_api.training.pipeline import train_and_serialize


@pytest.fixture(scope="session", autouse=True)
def prepared_model_artifacts() -> None:
    """Create deterministic dataset and model artifacts for API tests."""
    module = importlib.import_module("scripts.generate_ames_snapshot")
    module.main()

    get_settings.cache_clear()
    settings = get_settings()
    train_and_serialize(settings)


@pytest.fixture()
def settings():
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture()
def model_service(settings):
    service = ModelService(settings)
    service.load()
    return service


@pytest.fixture()
def metrics_store():
    return MetricsStore()


@pytest.fixture()
def request_stub(settings, model_service, metrics_store):
    app = create_app()
    app.state.settings = settings
    app.state.model_service = model_service
    app.state.metrics_store = metrics_store
    app.state.started_at = time.time() - 2

    return SimpleNamespace(
        app=app,
        state=SimpleNamespace(request_id="test-request-id"),
    )
