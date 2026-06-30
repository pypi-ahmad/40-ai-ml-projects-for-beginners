from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable

import numpy as np

from src.evaluation import mape, smape


logger = logging.getLogger(__name__)



def _metric_fn(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    metric = metric.lower()
    if metric == "rmse":
        return lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))
    if metric == "mae":
        return lambda y, p: float(np.mean(np.abs(y - p)))
    if metric == "mape":
        return lambda y, p: float(mape(y, p))
    if metric == "smape":
        return lambda y, p: float(smape(y, p))
    raise ValueError(f"Unknown metric '{metric}'")



def _normalize(weights: np.ndarray) -> np.ndarray:
    weights = np.clip(weights.astype(float), 0.0, None)
    total = float(weights.sum())
    if total <= 0:
        return np.ones_like(weights) / len(weights)
    return weights / total



def as_matrix(predictions: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    if not predictions:
        raise ValueError("predictions cannot be empty")

    names = list(predictions.keys())
    mat = np.column_stack([np.asarray(predictions[name]).ravel() for name in names])
    if mat.shape[0] == 0:
        raise ValueError("prediction arrays must be non-empty")
    return names, mat



def _simplex_integer_points(dim: int, units: int) -> np.ndarray:
    """Generate integer vectors with nonnegative entries summing to `units`."""
    if dim < 1:
        raise ValueError("dim must be >= 1")
    if units < 1:
        raise ValueError("units must be >= 1")

    current = np.zeros(dim, dtype=int)
    points: list[np.ndarray] = []

    def rec(index: int, remainder: int) -> None:
        if index == dim - 1:
            current[index] = remainder
            points.append(current.copy())
            return
        for value in range(remainder + 1):
            current[index] = value
            rec(index + 1, remainder - value)

    rec(0, units)
    return np.asarray(points, dtype=float)



def combine_with_weights(predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    names, mat = as_matrix(predictions)
    vec = np.array([weights.get(name, 0.0) for name in names], dtype=float)
    vec = _normalize(vec)
    return mat @ vec



def grid_search_weights(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    step: float = 0.05,
    metric: str = "rmse",
) -> tuple[dict[str, float], float]:
    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1]")

    names, mat = as_matrix(predictions)
    y_true = np.asarray(y_true).ravel()
    if y_true.shape[0] != mat.shape[0]:
        raise ValueError("y_true length must match prediction length")

    scorer = _metric_fn(metric)
    n = len(names)
    units = max(1, int(round(1.0 / step)))
    # Maintain exact simplex points even when 1/step is not integer.
    actual_step = 1.0 / units
    best_score = float("inf")
    best_w: dict[str, float] | None = None

    for raw in _simplex_integer_points(n, units) * actual_step:
        pred = mat @ raw
        score = scorer(y_true, pred)
        if score < best_score:
            best_score = score
            best_w = {names[i]: float(raw[i]) for i in range(n)}

    if best_w is None:
        # Fallback if no exact simplex point from floating grid.
        eq = {name: 1.0 / n for name in names}
        return eq, scorer(y_true, mat @ np.array(list(eq.values())))

    return best_w, best_score



def brute_force_weights(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    step: float = 0.05,
    metric: str = "rmse",
) -> tuple[dict[str, float], float]:
    # Alias kept for compatibility.
    return grid_search_weights(predictions, y_true, step=step, metric=metric)



def bayesian_optimize_weights(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    metric: str = "rmse",
    n_calls: int = 80,
    random_state: int = 42,
) -> tuple[dict[str, float], float]:
    names, mat = as_matrix(predictions)
    y_true = np.asarray(y_true).ravel()
    scorer = _metric_fn(metric)

    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except Exception as exc:
        logger.warning("skopt unavailable for bayesian optimization: %s", exc)
        return grid_search_weights(predictions, y_true, step=0.1, metric=metric)

    space = [Real(0.0, 1.0, name=name) for name in names]

    def objective(raw_weights: list[float]) -> float:
        w = _normalize(np.array(raw_weights, dtype=float))
        pred = mat @ w
        return scorer(y_true, pred)

    result = gp_minimize(
        objective,
        dimensions=space,
        n_calls=max(20, n_calls),
        random_state=random_state,
    )

    weights = _normalize(np.array(result.x, dtype=float))
    return {names[i]: float(weights[i]) for i in range(len(names))}, float(result.fun)



def flaml_optimize_weights(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    metric: str = "rmse",
    time_budget_s: int = 120,
    num_samples: int = 200,
) -> tuple[dict[str, float], float]:
    names, mat = as_matrix(predictions)
    y_true = np.asarray(y_true).ravel()
    scorer = _metric_fn(metric)

    try:
        from flaml import tune
    except Exception as exc:
        logger.warning("FLAML tune unavailable: %s", exc)
        return grid_search_weights(predictions, y_true, step=0.1, metric=metric)

    def objective(config: dict[str, float]) -> None:
        raw = np.array([config[name] for name in names], dtype=float)
        w = _normalize(raw)
        pred = mat @ w
        loss = scorer(y_true, pred)
        tune.report(loss=loss)

    search_space = {name: tune.uniform(0.0, 1.0) for name in names}

    analysis = tune.run(
        objective,
        config=search_space,
        metric="loss",
        mode="min",
        num_samples=num_samples,
        time_budget_s=time_budget_s,
        verbose=0,
    )

    best_cfg = analysis.best_config
    raw = np.array([best_cfg[name] for name in names], dtype=float)
    weights = _normalize(raw)
    score = float(analysis.best_result["loss"])
    return {names[i]: float(weights[i]) for i in range(len(names))}, score


@dataclass(slots=True)
class WeightOptimizer:
    method: str = "grid"
    step: float = 0.05
    metric: str = "rmse"
    bayesian_calls: int = 80
    flaml_budget_s: int = 120

    def optimize(
        self,
        predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, float | str]]:
        method = self.method.lower()

        if method in {"grid", "grid_search", "brute", "brute_force"}:
            weights, score = grid_search_weights(predictions, y_true, step=self.step, metric=self.metric)
        elif method in {"bayesian", "nelder_mead"}:
            weights, score = bayesian_optimize_weights(
                predictions,
                y_true,
                metric=self.metric,
                n_calls=self.bayesian_calls,
            )
        elif method == "flaml":
            weights, score = flaml_optimize_weights(
                predictions,
                y_true,
                metric=self.metric,
                time_budget_s=self.flaml_budget_s,
            )
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")

        return weights, {"best_score": float(score), "metric": self.metric, "method": method}
