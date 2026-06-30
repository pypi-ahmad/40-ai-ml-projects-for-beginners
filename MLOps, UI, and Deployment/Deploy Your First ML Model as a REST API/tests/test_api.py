"""Integration tests for production-style API contract."""
from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.anyio
class TestHealthAndInfo:
    async def test_health(self, client: AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["uptime_seconds"] >= 0.0
        assert "checks" in data
        assert "metrics_db_ready" in data["checks"]

    async def test_model_info_no_model(self, client: AsyncClient, tmp_path):
        from app.config import settings

        settings.model_path = tmp_path / "missing_model.joblib"
        settings.metadata_path = tmp_path / "missing_metadata.json"
        resp = await client.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_loaded"] is False

    async def test_model_info_with_model(self, client: AsyncClient, dummy_model_dir):
        resp = await client.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_loaded"] is True
        assert data["metadata"]["model_name"] == "test-random-forest"


@pytest.mark.anyio
class TestPredict:
    async def test_predict_no_model(self, client: AsyncClient, sample_payload, tmp_path):
        from app.config import settings

        settings.model_path = tmp_path / "missing_model.joblib"
        settings.metadata_path = tmp_path / "missing_metadata.json"
        resp = await client.post("/predict", json=sample_payload)
        assert resp.status_code == 503
        body = resp.json()
        assert body["code"] == "HTTP_503"

    async def test_predict_success(self, client: AsyncClient, dummy_model_dir, sample_payload):
        resp = await client.post("/predict", json=sample_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["prediction"], float)
        assert data["model_name"] == "test-random-forest"
        assert data["feature_schema_version"] == "california-housing-v1"
        assert data["latency_ms"] >= 0.0

    async def test_predict_validation_error(self, client: AsyncClient, dummy_model_dir, sample_payload):
        bad_payload = dict(sample_payload)
        bad_payload.pop("Latitude")
        resp = await client.post("/predict", json=bad_payload)
        assert resp.status_code == 422
        body = resp.json()
        assert body["code"] == "VALIDATION_ERROR"
        assert isinstance(body.get("field_errors"), list)

    async def test_predict_batch_success(self, client: AsyncClient, dummy_model_dir, sample_payload):
        payload = {"records": [sample_payload, sample_payload]}
        resp = await client.post("/predict-batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_records"] == 2
        assert len(data["predictions"]) == 2
        assert data["throughput_records_per_second"] >= 0.0

    async def test_predict_batch_rejects_oversize(self, client: AsyncClient, dummy_model_dir, sample_payload):
        from app.config import settings

        settings.max_batch_size = 1
        payload = {"records": [sample_payload, sample_payload]}
        resp = await client.post("/predict-batch", json=payload)
        assert resp.status_code == 422
        assert "max_batch_size" in resp.json()["detail"]


@pytest.mark.anyio
class TestAuthAndAdmin:
    async def test_api_key_enforced(self, client: AsyncClient, dummy_model_dir, sample_payload):
        from app.config import settings

        settings.api_key = "super-secret"

        missing = await client.post("/predict", json=sample_payload)
        assert missing.status_code == 401
        assert missing.json()["code"] == "UNAUTHORIZED"

        ok = await client.post(
            "/predict",
            json=sample_payload,
            headers={"X-API-Key": "super-secret"},
        )
        assert ok.status_code == 200

    async def test_reload_requires_api_key_when_enabled(self, client: AsyncClient, dummy_model_dir):
        from app.config import settings

        settings.api_key = "admin-key"
        denied = await client.post("/admin/reload")
        assert denied.status_code == 401

        ok = await client.post("/admin/reload", headers={"X-API-Key": "admin-key"})
        assert ok.status_code == 200
        assert ok.json()["status"] == "ok"


@pytest.mark.anyio
class TestExplainAndMetrics:
    async def test_explain_success(self, client: AsyncClient, dummy_model_dir, sample_payload):
        resp = await client.post("/explain", json=sample_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "feature_contributions" in data
        assert len(data["shap_values"]) == 8
        assert data["explainer_type"] in {"TreeExplainer", "LinearExplainer", "KernelExplainer"}

    async def test_metrics_summary(self, client: AsyncClient, dummy_model_dir, sample_payload):
        await client.post("/predict", json=sample_payload)
        await client.post("/predict-batch", json={"records": [sample_payload, sample_payload]})

        resp = await client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requests"] >= 2
        assert "throughput_rps_last_minute" in data
        assert data["model_name"] == "test-random-forest"
        assert data["uptime_seconds"] >= 0.0


@pytest.mark.anyio
class TestDocsAndContract:
    async def test_docs_endpoints(self, client: AsyncClient):
        docs = await client.get("/docs")
        redoc = await client.get("/redoc")
        openapi = await client.get("/openapi.json")
        assert docs.status_code == 200
        assert redoc.status_code == 200
        assert openapi.status_code == 200

    async def test_openapi_contains_examples(self, client: AsyncClient):
        openapi = await client.get("/openapi.json")
        schema = openapi.json()
        props = schema["components"]["schemas"]["HousingFeatures"]
        assert "example" in props
        assert "/predict-batch" in schema["paths"]

    async def test_request_size_limit(self, client: AsyncClient, dummy_model_dir):
        from app.config import settings

        settings.max_request_body_bytes = 16
        payload = {
            "records": [
                {
                    "MedInc": 8.3252,
                    "HouseAge": 41.0,
                    "AveRooms": 6.9841,
                    "AveBedrms": 1.0238,
                    "Population": 322.0,
                    "AveOccup": 2.5556,
                    "Latitude": 37.88,
                    "Longitude": -122.23,
                }
            ]
        }
        resp = await client.post("/predict-batch", json=payload)
        assert resp.status_code == 413
        body = resp.json()
        assert body["code"] == "REQUEST_TOO_LARGE"
        assert body["request_id"] is not None
        assert resp.headers.get("X-Request-ID")
