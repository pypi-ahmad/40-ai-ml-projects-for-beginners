from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.backtesting import (
    BacktestEngine,
    expanding_window_split,
    rolling_window_split,
    walk_forward_split,
)


@pytest.fixture
def sample_ts():
    n = 220
    X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
    y = X["f1"] * 1.8 + X["f2"] * 0.4 + np.random.randn(n) * 0.1
    return X, y



def test_walk_forward_split(sample_ts):
    X, y = sample_ts
    splits = list(walk_forward_split(X, y, n_splits=4, min_train_size=80, test_size=20))
    assert len(splits) > 0
    for tr, te in splits:
        assert len(tr) >= 80
        assert len(te) == 20



def test_expanding_window_split(sample_ts):
    X, y = sample_ts
    splits = list(expanding_window_split(X, y, n_splits=3, min_train_size=70, test_size=25))
    train_sizes = [len(tr) for tr, _ in splits]
    assert train_sizes == sorted(train_sizes)



def test_rolling_window_split(sample_ts):
    X, y = sample_ts
    splits = list(rolling_window_split(X, y, n_splits=3, window_size=90, test_size=20))
    assert all(len(tr) == 90 for tr, _ in splits)



def test_backtest_engine_walk_forward(sample_ts):
    X, y = sample_ts
    engine = BacktestEngine(model=LinearRegression(), strategy="walk_forward", n_splits=3, min_train_size=80, test_size=20)
    out = engine.run(X, y)
    assert "aggregated_metrics" in out
    assert "mean_rmse" in out["aggregated_metrics"]



def test_backtest_unknown_strategy_raises(sample_ts):
    X, y = sample_ts
    with pytest.raises(ValueError):
        BacktestEngine(model=LinearRegression(), strategy="unknown").run(X, y)
