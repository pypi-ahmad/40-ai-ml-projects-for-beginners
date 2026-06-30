from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.hybrid_models import (
    AdvancedStackingEnsemble,
    WeightedEnsemble,
    inverse_error_weights,
    rank_based_weights,
    weighted_ensemble,
)


@pytest.fixture
def sample_predictions():
    return {
        "A": np.array([1.0, 2.0, 3.0]),
        "B": np.array([1.1, 1.9, 3.1]),
        "C": np.array([0.9, 2.2, 2.8]),
    }


@pytest.fixture
def sample_true():
    return np.array([1.0, 2.0, 3.0])



def test_weighted_ensemble_normalizes_weights(sample_predictions):
    pred = weighted_ensemble(sample_predictions, {"A": 1.0, "B": 1.0, "C": 0.0})
    assert pred.shape == (3,)



def test_weighted_ensemble_empty_raises():
    with pytest.raises(ValueError):
        weighted_ensemble({}, {})



def test_inverse_error_weights(sample_predictions, sample_true):
    w = inverse_error_weights(sample_predictions, sample_true)
    assert abs(sum(w.values()) - 1.0) < 1e-6



def test_rank_based_weights(sample_predictions, sample_true):
    w = rank_based_weights(sample_predictions, sample_true)
    assert abs(sum(w.values()) - 1.0) < 1e-6



def test_weighted_ensemble_class(sample_predictions, sample_true):
    model = WeightedEnsemble(strategy="dynamic", optimizer_method="grid")
    model.fit(sample_predictions, sample_true)
    pred = model.predict(sample_predictions)
    assert pred.shape == sample_true.shape



def test_advanced_stacking_ensemble():
    X = np.random.randn(120, 4)
    y = X[:, 0] * 1.5 - X[:, 1] * 0.7 + np.random.randn(120) * 0.1
    stack = AdvancedStackingEnsemble(
        base_models=[("lr", LinearRegression()), ("rf", RandomForestRegressor(n_estimators=20, random_state=42))]
    )
    stack.fit(X, y)
    pred = stack.predict(X)
    assert pred.shape == (120,)
