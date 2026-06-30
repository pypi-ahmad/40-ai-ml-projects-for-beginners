from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


EPS = 1e-8



def _validate_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true and y_pred must be non-empty")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")
    return y_true, y_pred



def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate_arrays(y_true, y_pred)
    denom = np.maximum(np.abs(y_true), EPS)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)



def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate_arrays(y_true, y_pred)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), EPS)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)



def adjusted_r2_score(r2: float, n_samples: int, n_features: int) -> float:
    if n_samples <= n_features + 1:
        return float(r2)
    return float(1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))



def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int | None = None,
) -> dict[str, float]:
    y_true, y_pred = _validate_arrays(y_true, y_pred)
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2,
        # Uppercase aliases for notebook readability.
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "R2": r2,
    }

    if n_features is not None:
        adj = adjusted_r2_score(r2, len(y_true), n_features)
        metrics["adjusted_r2"] = adj
        metrics["Adjusted R2"] = adj

    return metrics



def evaluate_regression(
    y_true: np.ndarray,
    predictions: np.ndarray | dict[str, np.ndarray],
    n_features: int | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate one or many regression prediction vectors."""
    if isinstance(predictions, dict):
        return {
            name: regression_metrics(y_true, pred, n_features=n_features)
            for name, pred in predictions.items()
        }
    return {"model": regression_metrics(y_true, predictions, n_features=n_features)}



def evaluate_trading(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    transaction_cost: float = 0.001,
) -> dict[str, float]:
    """Simple directional strategy diagnostics from forecasts."""
    y_true, y_pred = _validate_arrays(y_true, y_pred)

    true_return = np.diff(y_true) / np.maximum(y_true[:-1], EPS)
    pred_return = np.diff(y_pred)
    signal = np.sign(pred_return)

    strategy_returns = signal * true_return - transaction_cost * np.abs(np.diff(signal, prepend=signal[0]))
    hit_rate = float(np.mean(np.sign(true_return) == signal))

    cumulative = np.cumprod(1.0 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative / np.maximum(running_max, EPS) - 1.0

    return {
        "total_return": float(cumulative[-1] - 1.0) if len(cumulative) else 0.0,
        "annualized_volatility": float(np.std(strategy_returns) * np.sqrt(252)),
        "sharpe_ratio": float(
            (np.mean(strategy_returns) / np.maximum(np.std(strategy_returns), EPS)) * np.sqrt(252)
        ),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "hit_rate": hit_rate,
        "num_trades": float(len(strategy_returns)),
    }



def metrics_table(
    metrics_by_model: dict[str, dict[str, Any]],
    sort_by: str = "rmse",
) -> pd.DataFrame:
    df = pd.DataFrame(metrics_by_model).T
    sort_key = sort_by if sort_by in df.columns else sort_by.upper()
    if sort_key in df.columns:
        df = df.sort_values(sort_key, ascending=True)
    return df
