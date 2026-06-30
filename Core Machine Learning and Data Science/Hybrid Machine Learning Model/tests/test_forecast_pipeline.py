from __future__ import annotations

import numpy as np

from src.baseline_models import BaselineResult
from src.forecast_pipeline import ForecastingFramework


def test_train_hybrids_separates_validation_and_test(project_root):
    fw = ForecastingFramework(str(project_root / "config" / "config.yaml"))
    fw.load_data()
    ds = fw.prepare_horizon_dataset(1)

    seq_len = 30
    rng = np.random.default_rng(42)

    baseline_bundle = {
        "results": {
            "Linear Regression": BaselineResult(
                model=None,
                val_pred=ds.y_val + rng.normal(0.0, 0.5, size=len(ds.y_val)),
                test_pred=ds.y_test + rng.normal(0.0, 0.6, size=len(ds.y_test)),
                val_metrics={},
                test_metrics={},
            ),
            "Random Forest": BaselineResult(
                model=None,
                val_pred=ds.y_val + rng.normal(0.0, 0.4, size=len(ds.y_val)),
                test_pred=ds.y_test + rng.normal(0.0, 0.5, size=len(ds.y_test)),
                val_metrics={},
                test_metrics={},
            ),
        }
    }
    deep_bundle = {
        "sequence_length": seq_len,
        "results": {
            "vanilla_lstm": {
                "val_pred": ds.y_val[seq_len:] + rng.normal(0.0, 0.4, size=len(ds.y_val) - seq_len),
                "test_pred": ds.y_test[seq_len:] + rng.normal(0.0, 0.5, size=len(ds.y_test) - seq_len),
            },
            "gru": {
                "val_pred": ds.y_val[seq_len:] + rng.normal(0.0, 0.35, size=len(ds.y_val) - seq_len),
                "test_pred": ds.y_test[seq_len:] + rng.normal(0.0, 0.45, size=len(ds.y_test) - seq_len),
            },
        },
    }

    out = fw.train_hybrids(1, baseline_bundle=baseline_bundle, deep_bundle=deep_bundle)
    assert "val_predictions" in out and "test_predictions" in out
    assert "y_val_true" in out and "y_test_true" in out
    assert len(out["y_val_true"]) == len(next(iter(out["val_predictions"].values())))
    assert len(out["y_test_true"]) == len(next(iter(out["test_predictions"].values())))


def test_optimize_weights_supports_holdout_evaluation(project_root):
    fw = ForecastingFramework(str(project_root / "config" / "config.yaml"))
    preds_val = {
        "a": np.array([1.0, 2.0, 3.0, 4.0]),
        "b": np.array([1.2, 2.1, 2.9, 4.3]),
    }
    y_val = np.array([1.1, 2.0, 3.1, 4.1])
    preds_test = {
        "a": np.array([2.0, 3.0, 4.0]),
        "b": np.array([2.2, 3.1, 4.4]),
    }
    y_test = np.array([2.1, 3.2, 4.1])

    out = fw.optimize_weights(
        horizon=1,
        predictions=preds_val,
        y_true=y_val,
        method="grid",
        evaluation_predictions=preds_test,
        evaluation_y_true=y_test,
    )
    assert "fit_metrics" in out
    assert "test_metrics" in out
    assert "weights" in out
    assert out["metrics"]["rmse"] == out["test_metrics"]["rmse"]
