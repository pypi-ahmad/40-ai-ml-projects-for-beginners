from fastapi.testclient import TestClient

from domain_llm_ft.inference.engine import Prediction
from domain_llm_ft.serving import api


class DummyEngine:
    async def predict_async(self, text: str) -> Prediction:
        _ = text
        return Prediction(label="neutral", score=0.9, probabilities=[0.1, 0.9])

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        return [Prediction(label="neutral", score=0.9, probabilities=[0.1, 0.9]) for _ in texts]


def test_health_endpoint() -> None:
    client = TestClient(api.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(api, "_engine", lambda _model_name: DummyEngine())
    client = TestClient(api.app)
    resp = client.post("/predict", json={"text": "hello", "model_name": "distilbert"})
    assert resp.status_code == 200
    assert resp.json()["label"] == "neutral"
