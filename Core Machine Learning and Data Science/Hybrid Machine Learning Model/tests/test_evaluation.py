from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import evaluate_regression, evaluate_trading, mape, regression_metrics, smape



def test_mape_smape_bounds():
    y_true = np.array([100, 110, 120], dtype=float)
    y_pred = np.array([100, 105, 130], dtype=float)
    assert mape(y_true, y_pred) >= 0
    assert 0 <= smape(y_true, y_pred) <= 200



def test_mape_empty_raises():
    with pytest.raises(ValueError):
        mape(np.array([]), np.array([]))



def test_regression_metrics_keys():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    metrics = regression_metrics(y_true, y_pred, n_features=2)
    for key in ["mae", "mse", "rmse", "mape", "smape", "r2", "adjusted_r2"]:
        assert key in metrics



def test_evaluate_regression_multi_model():
    y_true = np.array([1.0, 2.0, 3.0])
    preds = {
        "a": np.array([1.0, 2.0, 3.0]),
        "b": np.array([0.9, 2.2, 2.8]),
    }
    out = evaluate_regression(y_true, preds)
    assert set(out.keys()) == {"a", "b"}



def test_evaluate_trading_keys():
    y_true = np.array([100, 102, 101, 105, 110], dtype=float)
    y_pred = np.array([100, 101, 102, 104, 111], dtype=float)
    stats = evaluate_trading(y_true, y_pred)
    for key in ["total_return", "sharpe_ratio", "max_drawdown", "hit_rate", "num_trades"]:
        assert key in stats
