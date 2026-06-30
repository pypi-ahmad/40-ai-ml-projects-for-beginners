from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from src.evaluation import evaluate_regression


logger = logging.getLogger(__name__)


class NaiveForecastRegressor(BaseEstimator, RegressorMixin):
    """Forecast next values as last seen target value."""

    def __init__(self) -> None:
        self.last_value_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveForecastRegressor":
        y = np.asarray(y).ravel()
        if y.size == 0:
            raise ValueError("y cannot be empty")
        self.last_value_ = float(y[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.last_value_ is None:
            raise RuntimeError("Model must be fit before predict")
        return np.full(shape=(len(X),), fill_value=self.last_value_, dtype=float)


class MovingAverageRegressor(BaseEstimator, RegressorMixin):
    """Forecast as moving average of trailing target window."""

    def __init__(self, window: int = 5) -> None:
        self.window = window
        self.mean_value_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MovingAverageRegressor":
        y = np.asarray(y).ravel()
        if y.size < self.window:
            self.mean_value_ = float(np.mean(y))
        else:
            self.mean_value_ = float(np.mean(y[-self.window :]))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.mean_value_ is None:
            raise RuntimeError("Model must be fit before predict")
        return np.full(shape=(len(X),), fill_value=self.mean_value_, dtype=float)


@dataclass(slots=True)
class BaselineResult:
    model: Any
    val_pred: np.ndarray
    test_pred: np.ndarray
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]



def make_baseline_model_registry(random_state: int = 42, n_jobs: int = -1) -> dict[str, Any]:
    """Build required baseline model registry."""
    models: dict[str, Any] = {
        "Naive Forecast": NaiveForecastRegressor(),
        "Moving Average": MovingAverageRegressor(window=5),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=150,
            max_depth=12,
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "SVR": SVR(C=50.0, epsilon=0.05, kernel="rbf"),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=8, weights="distance"),
    }

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=180,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=n_jobs,
            objective="reg:squarederror",
            verbosity=0,
        )
    except Exception as exc:
        logger.warning("XGBoost unavailable: %s", exc)

    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=180,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
        )
    except Exception as exc:
        logger.warning("LightGBM unavailable: %s", exc)

    try:
        from catboost import CatBoostRegressor

        models["CatBoost"] = CatBoostRegressor(
            iterations=180,
            depth=6,
            learning_rate=0.05,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        )
    except Exception as exc:
        logger.warning("CatBoost unavailable: %s", exc)

    return models



def train_single_model(
    name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> BaselineResult:
    mdl = clone(model)
    mdl.fit(X_train, y_train)

    val_pred = mdl.predict(X_val)
    test_pred = mdl.predict(X_test)

    val_metrics = evaluate_regression(y_val, val_pred)["model"]
    test_metrics = evaluate_regression(y_test, test_pred)["model"]
    logger.info("%s val_rmse=%.4f test_rmse=%.4f", name, val_metrics["rmse"], test_metrics["rmse"])
    return BaselineResult(
        model=mdl,
        val_pred=np.asarray(val_pred),
        test_pred=np.asarray(test_pred),
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )



def run_baseline_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[dict[str, BaselineResult], pd.DataFrame]:
    registry = make_baseline_model_registry(random_state=random_state, n_jobs=n_jobs)
    results: dict[str, BaselineResult] = {}

    for name, model in registry.items():
        try:
            results[name] = train_single_model(
                name,
                model,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
            )
        except Exception as exc:
            logger.warning("Failed baseline %s: %s", name, exc)

    leaderboard_rows: list[dict[str, float | str]] = []
    for name, result in results.items():
        row: dict[str, float | str] = {"model": name}
        for key in ["mae", "mse", "rmse", "mape", "smape", "r2"]:
            row[f"val_{key}"] = float(result.val_metrics[key])
            row[f"test_{key}"] = float(result.test_metrics[key])
        leaderboard_rows.append(row)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("test_rmse")
    return results, leaderboard



def run_lazypredict_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Run LazyPredict benchmark for broad baseline scan."""
    from lazypredict.Supervised import LazyRegressor

    reg = LazyRegressor(verbose=0, ignore_warnings=True)
    models_df, _ = reg.fit(
        pd.DataFrame(X_train),
        pd.DataFrame(X_test),
        pd.Series(y_train),
        pd.Series(y_test),
    )
    out = models_df.reset_index().rename(columns={"index": "model"})
    out["tool"] = "LazyPredict"
    return out



def run_pycaret_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
    fold: int = 3,
) -> pd.DataFrame:
    """Run PyCaret comparison on time-series style split.

    PyCaret API changes across versions; wrapper is defensive and returns a normalized table.
    """
    import contextlib
    import io

    from pycaret.regression import RegressionExperiment

    train_df = pd.DataFrame(X_train).copy()
    train_df["target"] = y_train

    # Keep signature stable for callers even though benchmark table is CV-based.
    _ = (X_test, y_test)

    exp = RegressionExperiment(
        target="target",
        session_id=seed,
        fold=fold,
        fold_strategy="timeseries",
        verbose=False,
    )
    exp.fit(train_df)
    # Suppress noisy estimator logs (e.g., LightGBM warnings) during benchmark table creation.
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        _ = exp.compare_models(sort="RMSE", n_select=10, turbo=True)
    table = exp.pull().reset_index(drop=True)

    if "Model" in table.columns:
        table = table.rename(columns={"Model": "model"})
    elif table.columns.size > 0:
        table = table.rename(columns={table.columns[0]: "model"})
    table["tool"] = "PyCaret"
    return table



def run_flaml_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    time_budget_s: int = 300,
    metric: str = "rmse",
    seed: int = 42,
) -> pd.DataFrame:
    """Run FLAML AutoML benchmark and report summary row."""
    from flaml import AutoML

    automl = AutoML()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task="regression",
        time_budget=time_budget_s,
        metric=metric,
        seed=seed,
        verbose=0,
        eval_method="cv",
        n_splits=3,
        split_type="time",
    )
    preds = automl.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    return pd.DataFrame(
        [
            {
                "model": str(automl.best_estimator),
                "tool": "FLAML",
                "rmse": rmse,
                "best_loss": float(automl.best_loss),
                "config": str(automl.best_config),
            }
        ]
    )



def run_automl_suite(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    pycaret_enabled: bool = True,
    lazypredict_enabled: bool = True,
    flaml_enabled: bool = True,
    seed: int = 42,
    flaml_budget_s: int = 300,
) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}

    if lazypredict_enabled:
        try:
            outputs["lazypredict"] = run_lazypredict_benchmark(X_train, y_train, X_test, y_test)
        except Exception as exc:
            logger.warning("LazyPredict failed: %s", exc)

    if pycaret_enabled:
        try:
            outputs["pycaret"] = run_pycaret_benchmark(X_train, y_train, X_test, y_test, seed=seed)
        except Exception as exc:
            logger.warning("PyCaret failed: %s", exc)

    if flaml_enabled:
        try:
            outputs["flaml"] = run_flaml_benchmark(
                X_train,
                y_train,
                X_test,
                y_test,
                time_budget_s=flaml_budget_s,
                seed=seed,
            )
        except Exception as exc:
            logger.warning("FLAML failed: %s", exc)

    return outputs



def expanding_time_series_cv_scores(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> list[float]:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    scores: list[float] = []
    for train_idx, test_idx in splitter.split(X):
        fold_model = clone(model)
        fold_model.fit(X[train_idx], y[train_idx])
        preds = fold_model.predict(X[test_idx])
        rmse = float(np.sqrt(mean_squared_error(y[test_idx], preds)))
        scores.append(rmse)
    return scores
