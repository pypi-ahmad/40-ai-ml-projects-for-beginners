from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset

from domain_llm_ft.config.loader import load_config
from domain_llm_ft.data.preprocess import deduplicate_dataset, normalize_text
from domain_llm_ft.evaluation.metrics import classification_metrics
from domain_llm_ft.inference.engine import Prediction
from domain_llm_ft.serving import api
from domain_llm_ft.serving.schemas import PredictRequest


def run() -> None:
    cfg = load_config(Path("configs/baseline.yaml"))
    assert cfg.experiment_name == "domain_llm_ft"
    assert cfg.dataset.name == "ag_news"

    assert normalize_text("  hello   world  ") == "hello world"

    ds = Dataset.from_pandas(
        pd.DataFrame(
            {
                "text": ["a", "a", "b"],
                "label": [0, 0, 1],
            }
        ),
        preserve_index=False,
    )
    assert len(deduplicate_dataset(ds, "text")) == 2

    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_proba = np.array([0.1, 0.9, 0.4, 0.2])
    artifacts = classification_metrics(y_true, y_pred, y_proba, average="binary")
    assert artifacts.metrics["accuracy"] == 0.75

    class DummyEngine:
        async def predict_async(self, text: str) -> Prediction:
            _ = text
            return Prediction(label="neutral", score=0.9, probabilities=[0.1, 0.9])

        def predict_batch(self, texts: list[str]) -> list[Prediction]:
            return [Prediction(label="neutral", score=0.9, probabilities=[0.1, 0.9]) for _ in texts]

    api._engine = lambda _model_name: DummyEngine()  # type: ignore[assignment]

    async def _api_checks() -> None:
        health = await api.health()
        assert health.status == "ok"
        out = await api.predict(PredictRequest(text="hello", model_name="distilbert"))
        assert out.label == "neutral"

    asyncio.run(_api_checks())

    print("manual-tests-passed")


if __name__ == "__main__":
    run()
