"""API contract tests using ASGI transport to avoid flaky TestClient behavior."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

SAMPLE_INPUT = {
    "MedInc": 4.5,
    "HouseAge": 20.0,
    "AveRooms": 5.5,
    "AveBedrms": 1.1,
    "Population": 1200.0,
    "AveOccup": 2.5,
    "Latitude": 34.0,
    "Longitude": -118.0,
}


@pytest.mark.anyio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as api_client:
        response = await api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.anyio
async def test_model_info():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as api_client:
        response = await api_client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "features" in data
    assert "metrics" in data
    assert len(data["features"]) == 8


@pytest.mark.anyio
async def test_single_prediction():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as api_client:
        response = await api_client.post("/predict", json=SAMPLE_INPUT)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_value" in data
    assert isinstance(data["predicted_value"], float)


@pytest.mark.anyio
async def test_batch_prediction_canonical():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as api_client:
        response = await api_client.post("/predict-batch", json={"instances": [SAMPLE_INPUT, SAMPLE_INPUT]})
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2


@pytest.mark.anyio
async def test_batch_prediction_alias():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as api_client:
        response = await api_client.post("/batch-predict", json={"instances": [SAMPLE_INPUT]})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1


@pytest.mark.anyio
async def test_explain():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as api_client:
        response = await api_client.post("/explain", json={"input": SAMPLE_INPUT})
    assert response.status_code == 200
    data = response.json()
    assert "shap_values" in data
    assert "base_value" in data
    assert "prediction" in data
    assert len(data["shap_values"]) == 8


@pytest.mark.anyio
async def test_metrics_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as api_client:
        _ = await api_client.post("/predict", json=SAMPLE_INPUT)
        metrics = await api_client.get("/metrics")
    assert metrics.status_code == 200
    assert "ml_predictions_total" in metrics.text


@pytest.mark.anyio
async def test_invalid_input_returns_422():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as api_client:
        response = await api_client.post("/predict", json={"MedInc": "invalid"})
    assert response.status_code == 422
