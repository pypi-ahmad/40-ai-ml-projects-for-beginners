from __future__ import annotations

import numpy as np
import pytest

from src.models import MODEL_REGISTRY, PyTorchMLPRegressor, train_model



def make_data(n: int = 120):
    X = np.random.randn(n, 8)
    y = X[:, 0] * 1.2 + X[:, 1] * -0.4 + np.random.randn(n) * 0.1
    return X, y



def test_model_registry_has_required_baselines():
    required = {"Linear Regression", "Random Forest", "Naive Forecast", "Moving Average"}
    assert required.issubset(set(MODEL_REGISTRY.keys()))



def test_train_model_success():
    X, y = make_data()
    model, metrics = train_model(X, y, model_name="Linear Regression")
    assert hasattr(model, "predict")
    assert "train_rmse" in metrics



def test_train_model_unknown_raises():
    X, y = make_data()
    with pytest.raises(ValueError):
        train_model(X, y, model_name="unknown-model")



def test_pytorch_mlp_wrapper_predict_before_fit_raises():
    model = PyTorchMLPRegressor(input_dim=8)
    with pytest.raises(RuntimeError):
        model.predict(np.random.randn(5, 8))



def test_pytorch_mlp_wrapper_fit_predict():
    X, y = make_data()
    model = PyTorchMLPRegressor(input_dim=8, epochs=50)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
