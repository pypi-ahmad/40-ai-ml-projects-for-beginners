from __future__ import annotations

import numpy as np
import pytest

from src.weight_optimization import (
    WeightOptimizer,
    brute_force_weights,
    combine_with_weights,
    grid_search_weights,
)


@pytest.fixture
def sample_preds_true():
    preds = {
        "A": np.array([1.0, 2.0, 3.0]),
        "B": np.array([1.2, 1.9, 3.1]),
        "C": np.array([0.8, 2.1, 2.9]),
    }
    true = np.array([1.0, 2.0, 3.0])
    return preds, true



def test_grid_search_returns_valid_weights(sample_preds_true):
    preds, true = sample_preds_true
    weights, score = grid_search_weights(preds, true, step=0.5, metric="rmse")
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert isinstance(score, float)



def test_brute_force_alias(sample_preds_true):
    preds, true = sample_preds_true
    weights, _ = brute_force_weights(preds, true, step=0.5, metric="mae")
    assert abs(sum(weights.values()) - 1.0) < 1e-6



def test_weight_optimizer_grid(sample_preds_true):
    preds, true = sample_preds_true
    opt = WeightOptimizer(method="grid", step=0.2, metric="rmse")
    weights, meta = opt.optimize(preds, true)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert meta["method"] == "grid"



def test_weight_optimizer_unknown_raises(sample_preds_true):
    preds, true = sample_preds_true
    opt = WeightOptimizer(method="unknown")
    with pytest.raises(ValueError):
        opt.optimize(preds, true)



def test_combine_with_weights(sample_preds_true):
    preds, _ = sample_preds_true
    out = combine_with_weights(preds, {"A": 0.6, "B": 0.2, "C": 0.2})
    assert out.shape == (3,)


def test_grid_search_length_mismatch_raises(sample_preds_true):
    preds, _ = sample_preds_true
    with pytest.raises(ValueError):
        grid_search_weights(preds, np.array([1.0, 2.0]), step=0.5, metric="rmse")


def test_grid_search_handles_larger_model_bank():
    rng = np.random.default_rng(42)
    y_true = rng.normal(size=40)
    preds = {
        f"m{i}": y_true + rng.normal(scale=0.1 + i * 0.02, size=40)
        for i in range(6)
    }
    weights, score = grid_search_weights(preds, y_true, step=0.2, metric="rmse")
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert score >= 0.0
