"""Optional AutoML benchmark runners for LazyPredict, PyCaret, and FLAML."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
import time

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoMLResult:
    framework: str
    status: str
    best_model: str
    metric_name: str
    metric_value: float
    runtime_seconds: float
    details: dict[str, str | float | int]


def run_automl_benchmarks(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    random_seed: int,
) -> list[AutoMLResult]:
    """Run AutoML frameworks if installed; otherwise return explicit unavailable status."""
    return [
        _run_lazypredict(x_train, x_val, y_train, y_val),
        _run_flaml(x_train, x_val, y_train, y_val, random_seed=random_seed),
        _run_pycaret(x_train, x_val, y_train, y_val, random_seed=random_seed),
    ]


def _run_lazypredict(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> AutoMLResult:
    module = _try_import("lazypredict.Supervised")
    if module is None:
        return _missing("LazyPredict")

    started = time.perf_counter()
    try:
        regressor = module.LazyRegressor(verbose=0, ignore_warnings=True)
        models_frame, _ = regressor.fit(x_train, x_val, y_train, y_val)
        best = models_frame.sort_values("RMSE").iloc[0]
        return AutoMLResult(
            framework="LazyPredict",
            status="ok",
            best_model=str(best.name),
            metric_name="rmse",
            metric_value=round(float(best["RMSE"]), 4),
            runtime_seconds=round(time.perf_counter() - started, 4),
            details={"models_evaluated": int(models_frame.shape[0])},
        )
    except Exception as exc:  # pragma: no cover - optional dependency path
        logger.warning("LazyPredict run failed: %s", exc)
        return _failed("LazyPredict", started, str(exc))


def _run_flaml(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    random_seed: int,
) -> AutoMLResult:
    module = _try_import("flaml")
    if module is None:
        return _missing("FLAML")

    started = time.perf_counter()
    try:
        automl = module.AutoML()
        automl.fit(
            X_train=x_train,
            y_train=y_train,
            task="regression",
            metric="rmse",
            time_budget=60,
            seed=random_seed,
            verbose=0,
        )
        pred = automl.predict(x_val)
        rmse = float((((pred - y_val.to_numpy()) ** 2).mean()) ** 0.5)
        return AutoMLResult(
            framework="FLAML",
            status="ok",
            best_model=str(automl.best_estimator),
            metric_name="rmse",
            metric_value=round(rmse, 4),
            runtime_seconds=round(time.perf_counter() - started, 4),
            details={"time_budget": 60},
        )
    except Exception as exc:  # pragma: no cover - optional dependency path
        logger.warning("FLAML run failed: %s", exc)
        return _failed("FLAML", started, str(exc))


def _run_pycaret(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    random_seed: int,
) -> AutoMLResult:
    regression = _try_import("pycaret.regression")
    if regression is None:
        return _missing("PyCaret")

    started = time.perf_counter()
    try:
        train_df = x_train.copy()
        train_df["sale_price"] = y_train.to_numpy()
        valid_df = x_val.copy()
        valid_df["sale_price"] = y_val.to_numpy()

        regression.setup(
            train_df,
            target="sale_price",
            session_id=random_seed,
            verbose=False,
            html=False,
            fold=3,
            train_size=0.8,
        )
        best = regression.compare_models(n_select=1, sort="RMSE")
        best_name = str(type(best).__name__)

        pred_df = regression.predict_model(best, data=valid_df)
        rmse = float((((pred_df["prediction_label"] - valid_df["sale_price"]) ** 2).mean()) ** 0.5)

        return AutoMLResult(
            framework="PyCaret",
            status="ok",
            best_model=best_name,
            metric_name="rmse",
            metric_value=round(rmse, 4),
            runtime_seconds=round(time.perf_counter() - started, 4),
            details={"rows": int(train_df.shape[0])},
        )
    except Exception as exc:  # pragma: no cover - optional dependency path
        logger.warning("PyCaret run failed: %s", exc)
        return _failed("PyCaret", started, str(exc))


def _try_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


def _missing(framework: str) -> AutoMLResult:
    return AutoMLResult(
        framework=framework,
        status="unavailable",
        best_model="",
        metric_name="rmse",
        metric_value=0.0,
        runtime_seconds=0.0,
        details={"reason": "dependency not installed"},
    )


def _failed(framework: str, started: float, reason: str) -> AutoMLResult:
    return AutoMLResult(
        framework=framework,
        status="failed",
        best_model="",
        metric_name="rmse",
        metric_value=0.0,
        runtime_seconds=round(time.perf_counter() - started, 4),
        details={"reason": reason[:300]},
    )
