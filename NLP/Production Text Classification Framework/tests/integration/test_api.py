import pytest
from httpx import ASGITransport, AsyncClient

from textclf_framework.serving.api import APIConfig, create_app
from textclf_framework.serving.inference import Prediction


class DummyEngine:
    async def predict(self, text: str, top_k: int = 3):
        return [Prediction(label_id=1, label_name="positive", confidence=0.91)], 4.2

    async def predict_batch(self, texts: list[str], top_k: int = 3):
        preds = [[Prediction(label_id=1, label_name="positive", confidence=0.91)] for _ in texts]
        return preds, 6.5


@pytest.mark.asyncio
async def test_health_predict_batch() -> None:
    app = create_app(
        APIConfig(model_path="unused", label_names=["negative", "positive"]),
        engine=DummyEngine(),
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        health = await client.get("/health")
        assert health.status_code == 200

        response = await client.post("/predict", json={"text": "great product", "top_k": 1})
        assert response.status_code == 200
        body = response.json()
        assert body["predictions"][0]["label_name"] == "positive"

        batch = await client.post("/predict/batch", json={"texts": ["a", "b"], "top_k": 1})
        assert batch.status_code == 200
        assert len(batch.json()["predictions"]) == 2
