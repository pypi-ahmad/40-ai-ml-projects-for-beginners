from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor

from src.baseline_models import make_baseline_model_registry
from src.evaluation import regression_metrics


MODEL_REGISTRY = make_baseline_model_registry()


class PyTorchMLPRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible neural regressor wrapper.

    Uses sklearn MLP internally for CPU-friendly deterministic behavior.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int] | None = None,
        epochs: int = 200,
        lr: float = 0.001,
        batch_size: int = 64,
        patience: int = 20,
        min_delta: float = 1e-4,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers or [128, 64, 32]
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.min_delta = min_delta

        self._model: MLPRegressor | None = None
        self.best_epoch: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PyTorchMLPRegressor":
        self._model = MLPRegressor(
            hidden_layer_sizes=tuple(self.hidden_layers),
            max_iter=self.epochs,
            random_state=42,
            batch_size=self.batch_size,
            learning_rate_init=self.lr,
            early_stopping=True,
            n_iter_no_change=self.patience,
            tol=self.min_delta,
        )
        self._model.fit(X, y)
        self.best_epoch = int(getattr(self._model, "n_iter_", self.epochs))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        return self._model.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        return float(self._model.score(X, y))



def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "Random Forest",
    params: dict[str, Any] | None = None,
    tune: bool = False,
    param_grid: dict[str, list[Any]] | None = None,
    cv: int = 3,
) -> tuple[Any, dict[str, float]]:
    registry = make_baseline_model_registry()
    if model_name not in registry:
        raise ValueError(f"unknown model: {model_name}")

    model = registry[model_name]
    if params:
        model.set_params(**params)

    if tune and param_grid:
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

        splitter = TimeSeriesSplit(n_splits=max(2, cv))
        grid = GridSearchCV(model, param_grid=param_grid, cv=splitter, scoring="neg_root_mean_squared_error", n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    else:
        model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    metrics = regression_metrics(y_train, train_pred)
    out = {
        "train_mape": metrics["mape"],
        "train_r2": metrics["r2"],
        "train_rmse": metrics["rmse"],
    }
    return model, out



def train_all_models(X_train: np.ndarray, y_train: np.ndarray) -> dict[str, Any]:
    trained: dict[str, Any] = {}
    for name, model in make_baseline_model_registry().items():
        try:
            fitted, _ = train_model(X_train, y_train, model_name=name)
            trained[name] = fitted
        except Exception:
            continue
    return trained



def predict_all_models(models: dict[str, Any], X: np.ndarray) -> dict[str, np.ndarray]:
    return {name: model.predict(X) for name, model in models.items()}
