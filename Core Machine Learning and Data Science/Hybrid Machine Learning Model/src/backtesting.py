from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.evaluation import regression_metrics


logger = logging.getLogger(__name__)


IndexPair = tuple[np.ndarray, np.ndarray]



def _to_numpy(X: np.ndarray | pd.DataFrame | pd.Series) -> np.ndarray:
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.values
    return np.asarray(X)



def walk_forward_split(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    n_splits: int = 5,
    min_train_size: int | None = None,
    test_size: int | None = None,
) -> Iterator[IndexPair]:
    n = len(X)
    if n_splits < 1 or n_splits >= n:
        raise ValueError("n_splits must be >=1 and < number of samples")

    min_train = min_train_size or max(30, int(n * 0.5))
    test = test_size or max(1, (n - min_train) // n_splits)

    start = min_train
    for _ in range(n_splits):
        train_end = start
        test_end = min(train_end + test, n)
        if test_end <= train_end:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx
        start = test_end
        if start >= n:
            break



def expanding_window_split(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    n_splits: int = 5,
    min_train_size: int | None = None,
    test_size: int | None = None,
) -> Iterator[IndexPair]:
    n = len(X)
    min_train = min_train_size or max(30, int(n * 0.4))
    test = test_size or max(1, (n - min_train) // n_splits)

    for i in range(n_splits):
        train_end = min_train + i * test
        test_end = min(train_end + test, n)
        if test_end <= train_end:
            break
        yield np.arange(0, train_end), np.arange(train_end, test_end)



def rolling_window_split(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    n_splits: int = 5,
    window_size: int | None = None,
    test_size: int | None = None,
) -> Iterator[IndexPair]:
    n = len(X)
    win = window_size or max(30, int(n * 0.5))
    test = test_size or max(1, (n - win) // n_splits)

    for i in range(n_splits):
        train_start = i * test
        train_end = train_start + win
        test_end = min(train_end + test, n)
        if train_end >= n or test_end <= train_end:
            break
        yield np.arange(train_start, train_end), np.arange(train_end, test_end)



def _aggregate_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = ["mape", "smape", "mae", "mse", "rmse", "r2"]
    out: dict[str, float] = {}
    for key in keys:
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        out[f"mean_{key}"] = float(values.mean())
        out[f"std_{key}"] = float(values.std(ddof=0))
    return out


@dataclass(slots=True)
class BacktestEngine:
    model: any
    strategy: str = "walk_forward"
    n_splits: int = 5
    min_train_size: int | None = None
    test_size: int | None = None
    window_size: int | None = None

    def run(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        return_predictions: bool = True,
    ) -> dict[str, object]:
        if self.model is None:
            raise ValueError("model cannot be None")

        X_np = _to_numpy(X)
        y_np = _to_numpy(y).ravel()

        splitter_map = {
            "walk_forward": walk_forward_split,
            "expanding": expanding_window_split,
            "rolling": rolling_window_split,
        }
        if self.strategy not in splitter_map:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        splitter = splitter_map[self.strategy]
        split_kwargs = {
            "n_splits": self.n_splits,
            "min_train_size": self.min_train_size,
            "test_size": self.test_size,
        }
        if self.strategy == "rolling":
            split_kwargs = {
                "n_splits": self.n_splits,
                "window_size": self.window_size,
                "test_size": self.test_size,
            }

        fold_results: list[dict[str, object]] = []
        stitched_pred = np.full_like(y_np, np.nan, dtype=float)

        for fold_idx, (train_idx, test_idx) in enumerate(splitter(X_np, y_np, **split_kwargs)):
            model = clone(self.model)
            model.fit(X_np[train_idx], y_np[train_idx])
            preds = np.asarray(model.predict(X_np[test_idx])).ravel()
            metrics = regression_metrics(y_np[test_idx], preds)

            fold_results.append(
                {
                    "fold": fold_idx,
                    "train_size": int(len(train_idx)),
                    "test_size": int(len(test_idx)),
                    "metrics": metrics,
                }
            )
            stitched_pred[test_idx] = preds

        if not fold_results:
            raise RuntimeError("Backtest produced zero folds")

        metric_only = [fold["metrics"] for fold in fold_results]
        aggregated = _aggregate_fold_metrics(metric_only)

        payload: dict[str, object] = {
            "fold_results": fold_results,
            "per_fold_metrics": metric_only,
            "aggregated_metrics": aggregated,
        }
        if return_predictions:
            payload["all_predictions"] = stitched_pred
            payload["true_values"] = y_np
        return payload



def backtest_strategy(
    X: np.ndarray,
    y: np.ndarray,
    model: any,
    strategy: str = "walk_forward",
    n_splits: int = 5,
) -> dict[str, object]:
    engine = BacktestEngine(model=model, strategy=strategy, n_splits=n_splits)
    return engine.run(X, y)
