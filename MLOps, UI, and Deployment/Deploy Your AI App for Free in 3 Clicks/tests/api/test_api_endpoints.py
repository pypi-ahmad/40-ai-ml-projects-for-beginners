"""Endpoint contract tests executed via direct route calls."""

from __future__ import annotations

import pytest

from ml_api.api.routes import explain, health, metrics, model_info, predict, predict_batch
from ml_api.core.errors import BatchLimitError
from ml_api.schemas.prediction import BatchPredictionRequest, ExplainRequest, HouseFeatures, PredictionRequest
from tests.api.payloads import sample_record


def _house_features() -> HouseFeatures:
    return HouseFeatures(**sample_record())


def test_health_endpoint(request_stub, settings, model_service):
    response = health(request=request_stub, settings=settings, model_service=model_service)
    assert response.status in {"ok", "degraded"}
    assert response.app_name == settings.app_name


def test_model_info_endpoint(model_service):
    response = model_info(model_service=model_service)
    assert response.model_name
    assert response.feature_count > 10


def test_predict_endpoint_valid(request_stub, model_service):
    payload = PredictionRequest(record=_house_features())
    response = predict(payload=payload, request=request_stub, model_service=model_service)
    assert response.prediction > 0
    assert response.request_id == "test-request-id"


def test_predict_batch_endpoint(request_stub, settings, model_service):
    payload = BatchPredictionRequest(records=[_house_features() for _ in range(8)])
    response = predict_batch(
        payload=payload,
        request=request_stub,
        settings=settings,
        model_service=model_service,
    )
    assert response.count == 8
    assert len(response.predictions) == 8


def test_predict_batch_limit_exceeded(request_stub, settings, model_service):
    payload = BatchPredictionRequest(records=[_house_features() for _ in range(settings.max_batch_size)])
    payload.records.append(_house_features())
    with pytest.raises(BatchLimitError):
        predict_batch(
            payload=payload,
            request=request_stub,
            settings=settings,
            model_service=model_service,
        )


def test_explain_endpoint(request_stub, model_service):
    payload = ExplainRequest(record=_house_features(), top_k=5)
    response = explain(payload=payload, request=request_stub, model_service=model_service)
    assert response.prediction > 0
    assert response.method in {"shap", "perturbation"}
    assert len(response.contributions) == 5


def test_metrics_endpoint(metrics_store, model_service):
    metrics_store.record("/health", 200, 1.1)
    metrics_store.record("/predict", 200, 2.4)

    response = metrics(metrics_store=metrics_store, model_service=model_service)
    assert response.uptime_seconds >= 0
    assert response.request_counts["/health"] == 1
    assert response.request_counts["/predict"] == 1


def test_openapi_generation(request_stub):
    schema = request_stub.app.openapi()
    assert "/health" in schema["paths"]
    assert "/predict" in schema["paths"]
    assert request_stub.app.docs_url == "/docs"
    assert request_stub.app.redoc_url == "/redoc"
