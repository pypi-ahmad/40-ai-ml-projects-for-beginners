"""Model evaluation helpers and report generation."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


@dataclass(slots=True)
class RegressionMetrics:
    """Standardized regression metrics for model comparison."""

    mae: float
    mse: float
    rmse: float
    r2: float
    mape: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> RegressionMetrics:
    """Compute MAE/MSE/RMSE/R²/MAPE metrics for regression outputs."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    return RegressionMetrics(mae=mae, mse=mse, rmse=rmse, r2=r2, mape=mape)


def rank_models(metrics_by_model: dict[str, RegressionMetrics]) -> pd.DataFrame:
    """Convert metrics dictionary into sorted comparison table."""
    rows = []
    for model_name, metrics in metrics_by_model.items():
        row = metrics.to_dict()
        row["model_name"] = model_name
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    table = table[["model_name", "mae", "mse", "rmse", "r2", "mape"]]
    table = table.sort_values(by=["rmse", "mape", "mae"], ascending=[True, True, True])
    return table.reset_index(drop=True)
