from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge

from src.evaluation import mape
from src.weight_optimization import WeightOptimizer, combine_with_weights


logger = logging.getLogger(__name__)



def _validate_predictions(predictions: dict[str, np.ndarray]) -> tuple[list[str], dict[str, np.ndarray]]:
    if not predictions:
        raise ValueError("predictions must not be empty")
    names = list(predictions.keys())
    first_len = len(np.asarray(predictions[names[0]]).ravel())
    clean: dict[str, np.ndarray] = {}
    for name, pred in predictions.items():
        arr = np.asarray(pred).ravel()
        if len(arr) != first_len:
            raise ValueError("all prediction arrays must share same length")
        clean[name] = arr
    return names, clean



def weighted_ensemble(
    predictions: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    names, clean = _validate_predictions(predictions)
    if weights is None:
        weights = {name: 1.0 / len(names) for name in names}

    vec = np.array([weights.get(name, 0.0) for name in names], dtype=float)
    if np.allclose(vec.sum(), 0.0):
        vec = np.ones(len(names), dtype=float)

    # Normalize if user-provided weights do not sum to 1.
    vec = vec / vec.sum()

    mat = np.column_stack([clean[name] for name in names])
    return mat @ vec



def inverse_error_weights(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    names, clean = _validate_predictions(predictions)
    y_true = np.asarray(y_true).ravel()

    errors = {}
    for name in names:
        err = float(mape(y_true, clean[name])) + epsilon
        errors[name] = err

    inv = {name: 1.0 / err for name, err in errors.items()}
    total = sum(inv.values())
    return {name: inv[name] / total for name in names}



def rank_based_weights(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    power: float = 1.0,
) -> dict[str, float]:
    names, clean = _validate_predictions(predictions)
    y_true = np.asarray(y_true).ravel()

    ranking = sorted(names, key=lambda name: mape(y_true, clean[name]))
    n = len(ranking)
    score = {name: float((n - idx) ** power) for idx, name in enumerate(ranking)}
    total = sum(score.values())
    return {name: score[name] / total for name in names}


@dataclass(slots=True)
class WeightedEnsemble:
    strategy: str = "static"
    optimizer_method: str = "grid"
    weights: dict[str, float] | None = None

    def fit(self, predictions: dict[str, np.ndarray], y_true: np.ndarray) -> "WeightedEnsemble":
        strategy = self.strategy.lower()
        if strategy == "static":
            self.weights = inverse_error_weights(predictions, y_true)
        elif strategy == "adaptive":
            self.weights = rank_based_weights(predictions, y_true, power=1.5)
        elif strategy == "dynamic":
            optimizer = WeightOptimizer(method=self.optimizer_method, metric="rmse")
            self.weights, _ = optimizer.optimize(predictions, y_true)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        return self

    def predict(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Call fit before predict")
        return weighted_ensemble(predictions, self.weights)


class AdvancedStackingEnsemble:
    """Stacking wrapper for hybrid meta-learning."""

    def __init__(
        self,
        base_models: list[tuple[str, Any]],
        final_estimator: Any | None = None,
        cv: int = 3,
    ) -> None:
        self.base_models = base_models
        self.final_estimator = final_estimator or Ridge(alpha=1.0)
        self.cv = cv
        self.model = StackingRegressor(
            estimators=self.base_models,
            final_estimator=self.final_estimator,
            cv=self.cv,
            n_jobs=-1,
            passthrough=False,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdvancedStackingEnsemble":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)



def build_named_hybrid_predictions(
    ml_predictions: dict[str, np.ndarray],
    dl_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build requested hybrid variants from base ML + DL predictions."""
    out: dict[str, np.ndarray] = {}

    mapping = {
        "Linear Regression + LSTM": ("Linear Regression", "vanilla_lstm"),
        "Random Forest + LSTM": ("Random Forest", "vanilla_lstm"),
        "XGBoost + GRU": ("XGBoost", "gru"),
        "LightGBM + LSTM": ("LightGBM", "vanilla_lstm"),
    }

    for hybrid_name, (ml_key, dl_key) in mapping.items():
        if ml_key in ml_predictions and dl_key in dl_predictions:
            out[hybrid_name] = weighted_ensemble(
                {ml_key: ml_predictions[ml_key], dl_key: dl_predictions[dl_key]},
                {ml_key: 0.5, dl_key: 0.5},
            )

    # Weighted ensemble of all available predictions.
    all_preds = {**ml_predictions, **dl_predictions}
    if all_preds:
        optimizer = WeightOptimizer(method="grid", step=0.1, metric="rmse")
        weights, _ = optimizer.optimize(all_preds, y_true)
        out["Weighted Ensemble"] = combine_with_weights(all_preds, weights)

        static = WeightedEnsemble(strategy="adaptive").fit(all_preds, y_true)
        out["Meta Learner Ensemble"] = static.predict(all_preds)

    return out



def make_stacking_meta_learner() -> Any:
    return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)



def make_default_meta_model() -> Any:
    return LinearRegression()
