"""Benchmark adapters for LazyPredict, FLAML, and PyCaret.

Adapters are optional and return structured status metadata when skipped.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from src.constants import TARGET_NAME
from src.evaluation import compute_regression_metrics


@dataclass(slots=True)
class BenchmarkResult:
    """Container for benchmark output table and status metadata."""

    tool: str
    status: str
    table: pd.DataFrame
    notes: str = ""


def _empty_result(tool: str, status: str, notes: str) -> BenchmarkResult:
    return BenchmarkResult(tool=tool, status=status, table=pd.DataFrame(), notes=notes)


def _sample_training_rows(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_train_rows: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Optionally downsample training rows for faster, reproducible benchmark tools."""
    if max_train_rows < 1 or len(X_train) <= max_train_rows:
        return X_train, y_train
    sampled = X_train.sample(n=max_train_rows, random_state=random_state)
    return sampled, y_train.loc[sampled.index]


def run_lazypredict_benchmark(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    *,
    enabled: bool = False,
    max_train_rows: int = 4000,
) -> BenchmarkResult:
    """Run LazyPredict regression benchmark on train/validation split."""
    if not enabled:
        return _empty_result(
            "LazyPredict",
            "skipped",
            "Quick profile active. Use full profile to run LazyPredict benchmark.",
        )

    try:
        from lazypredict.Supervised import LazyRegressor
    except Exception as exc:
        return _empty_result("LazyPredict", "skipped", f"Import failed: {exc}")

    try:
        X_train_fit, y_train_fit = _sample_training_rows(X_train, y_train, max_train_rows)
        reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
        models, _ = reg.fit(X_train_fit, X_val, y_train_fit, y_val)
        table = models.reset_index().rename(columns={"index": "model_name"})
        keep_cols = [c for c in ["model_name", "MAE", "RMSE", "R-Squared", "Time Taken"] if c in table]
        table = table[keep_cols]
        table["train_rows_used"] = len(X_train_fit)
        return BenchmarkResult(tool="LazyPredict", status="ok", table=table)
    except Exception as exc:
        return _empty_result("LazyPredict", "failed", f"Execution failed: {exc}")


def run_flaml_benchmark(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    *,
    time_budget_seconds: int = 90,
    enabled: bool = False,
    max_train_rows: int = 5000,
) -> BenchmarkResult:
    """Run FLAML AutoML regression and return validation metrics."""
    if not enabled:
        return _empty_result(
            "FLAML",
            "skipped",
            "Quick profile active. Use full profile to run FLAML benchmark.",
        )

    try:
        from flaml import AutoML
    except Exception as exc:
        return _empty_result("FLAML", "skipped", f"Import failed: {exc}")

    try:
        X_train_fit, y_train_fit = _sample_training_rows(X_train, y_train, max_train_rows)
        automl = AutoML()
        automl.fit(
            X_train=X_train_fit,
            y_train=y_train_fit,
            task="regression",
            time_budget=time_budget_seconds,
            metric="rmse",
            log_file_name="artifacts/reports/flaml.log",
            seed=42,
            estimator_list=["lgbm", "xgboost", "rf", "extra_tree"],
        )

        preds = automl.predict(X_val)
        metrics = compute_regression_metrics(y_val, preds)

        table = pd.DataFrame(
            [
                {
                    "model_name": str(automl.model.estimator),
                    "mae": metrics.mae,
                    "mse": metrics.mse,
                    "rmse": metrics.rmse,
                    "r2": metrics.r2,
                    "mape": metrics.mape,
                    "best_config": str(automl.best_config),
                    "time_budget_seconds": time_budget_seconds,
                    "train_rows_used": len(X_train_fit),
                }
            ]
        )
        return BenchmarkResult(tool="FLAML", status="ok", table=table)
    except Exception as exc:
        return _empty_result("FLAML", "failed", f"Execution failed: {exc}")


def run_pycaret_benchmark(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    *,
    include_models: Sequence[str] | None = None,
    enabled: bool = False,
    max_train_rows: int = 3000,
) -> BenchmarkResult:
    """Run PyCaret regression compare workflow on train data."""
    if not enabled:
        return _empty_result(
            "PyCaret",
            "skipped",
            "Quick profile active. Use full profile to run PyCaret benchmark.",
        )

    try:
        from pycaret.regression import compare_models, pull, setup
    except Exception as exc:
        return _empty_result("PyCaret", "skipped", f"Import failed: {exc}")

    X_train_fit, y_train_fit = _sample_training_rows(X_train, y_train, max_train_rows)
    train_df = X_train_fit.copy()
    train_df[TARGET_NAME] = y_train_fit.values

    try:
        setup(
            data=train_df,
            target=TARGET_NAME,
            session_id=42,
            fold=3,
            verbose=False,
            html=False,
            use_gpu=False,
        )
        compare_models(
            n_select=5,
            include=list(include_models) if include_models else None,
            turbo=True,
            errors="ignore",
            verbose=False,
        )
        results = pull().reset_index().rename(columns={"index": "model_name"})

        if "RMSE" in results.columns:
            results = results.sort_values("RMSE", ascending=True)

        results["train_rows_used"] = len(X_train_fit)
        results["validation_rows"] = len(X_val)
        results["validation_target_mean"] = float(y_val.mean())

        return BenchmarkResult(tool="PyCaret", status="ok", table=results)
    except Exception as exc:
        return _empty_result("PyCaret", "failed", f"Execution failed: {exc}")
