import numpy as np

from textclf_framework.training.metrics import compute_classification_metrics


def test_metrics_core_keys_present() -> None:
    labels = [0, 1, 0, 1]
    preds = [0, 1, 1, 1]
    probs = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
        ]
    )

    metrics = compute_classification_metrics(labels=labels, preds=preds, probs=probs)

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "log_loss" in metrics
    assert "roc_auc" in metrics
