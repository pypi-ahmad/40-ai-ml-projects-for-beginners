import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport
from sklearn.linear_model import LogisticRegression

from ml_package.explainability import ModelExplainer
from ml_package.prediction_engine import PredictionEngine

from api.main import app, service


@pytest.fixture(autouse=True)
def setup_model():
    model = LogisticRegression(max_iter=200)
    X = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [7.0, 3.2, 4.7, 1.4],
        [6.3, 3.3, 6.0, 2.5],
    ])
    y = np.array([0, 1, 2])
    model.fit(X, y)
    service.engine = PredictionEngine(model, model_name="iris_classifier")
    service.background_data = X
    service.explainer = ModelExplainer(model, X)
    service.load_error = None


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"healthy", "degraded"}
    assert "load_error" in payload


@pytest.mark.asyncio
async def test_predict_valid(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = await client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "species" in data


@pytest.mark.asyncio
async def test_predict_invalid_negative(client):
    payload = {
        "sepal_length": -5,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = await client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_missing_field(client):
    payload = {"sepal_length": 5.1, "sepal_width": 3.5}
    response = await client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_rejects_extra_field(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
        "unknown_feature": 999,
    }
    response = await client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_model_info(client):
    response = await client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data


@pytest.mark.asyncio
async def test_metrics_json(client):
    response = await client.get("/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert "prediction_stats" in payload
    assert "model_version" in payload


@pytest.mark.asyncio
async def test_metrics_prometheus(client):
    response = await client.get("/metrics/prometheus")
    assert response.status_code == 200
    assert "iris_predictions_total" in response.text
    assert "text/plain" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_explain_local_mode(client):
    payload = {
        "mode": "local",
        "sample": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    }
    response = await client.post("/explain", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "feature_names" in body or "error" in body


@pytest.mark.asyncio
async def test_explain_global_mode(client):
    response = await client.post("/explain", json={"mode": "global"})
    assert response.status_code == 200
    body = response.json()
    assert "global_importance" in body or "error" in body


@pytest.mark.asyncio
async def test_explain_local_mode_requires_sample(client):
    response = await client.post("/explain", json={"mode": "local"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_batch(client):
    payload = {
        "samples": [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
            {
                "sepal_length": 6.2,
                "sepal_width": 2.9,
                "petal_length": 4.3,
                "petal_width": 1.3,
            },
        ]
    }
    response = await client.post("/predict-batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
