import numpy as np

from src.benchmark import compute_metrics


def test_compute_metrics_multiclass_with_proba():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    y_proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.5, 0.3],
            [0.7, 0.2, 0.1],
            [0.2, 0.3, 0.5],
            [0.1, 0.2, 0.7],
        ]
    )

    metrics = compute_metrics(y_true, y_pred, y_proba=y_proba)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0
