import numpy as np

from domain_llm_ft.evaluation.metrics import classification_metrics


def test_classification_metrics_binary() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_proba = np.array([0.1, 0.9, 0.4, 0.2])

    artifacts = classification_metrics(y_true, y_pred, y_proba, average="binary")

    assert artifacts.metrics["accuracy"] == 0.75
    assert "f1" in artifacts.metrics
    assert artifacts.confusion.shape == (2, 2)
