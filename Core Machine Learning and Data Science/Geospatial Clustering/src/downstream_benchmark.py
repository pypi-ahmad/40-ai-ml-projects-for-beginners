"""Downstream supervised benchmark (manual ML vs LazyPredict vs FLAML vs PyCaret).

Clustering itself is unsupervised, but real business programs often feed cluster
features into supervised tasks such as ETA prediction. This module benchmarks
those downstream workflows.
"""

from __future__ import annotations

import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    BENCHMARK_REPORT_PATH,
    COL_CLUSTER,
    COL_DELIVERY_PERSON_ID,
    COL_ID,
    COL_OUTLIER,
    COL_TIME_TAKEN,
    COL_ZONE_LABEL,
    PROJECT_ROOT,
    RANDOM_SEED,
)

# Avoid matplotlib cache permission warnings in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))


@dataclass
class BenchmarkResult:
    """One benchmark row in a normalized output format."""

    framework: str
    model: str
    rmse: float
    mae: float
    r2: float
    runtime_sec: float
    status: str
    note: str = ""


def _prepare_supervised_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Prepare feature matrix and target for delivery-time regression."""
    data = df.copy()
    y = pd.to_numeric(data[COL_TIME_TAKEN], errors="coerce")

    drop_cols = [COL_TIME_TAKEN, COL_CLUSTER, COL_ZONE_LABEL, COL_OUTLIER]
    for col in [COL_ID, COL_DELIVERY_PERSON_ID]:
        if col in data.columns:
            drop_cols.append(col)

    X = data.drop(columns=[c for c in drop_cols if c in data.columns])
    # Normalize pandas nullable missing markers for sklearn compatibility.
    X = X.replace({pd.NA: np.nan})

    # Only keep rows with valid target.
    mask = y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # Remove high-cardinality categorical identifiers that explode one-hot features.
    # This keeps AutoML comparisons fast and stable for notebook/runtime use.
    high_cardinality: list[str] = []
    cat_candidates = X.select_dtypes(include=["object", "category"]).columns.tolist()
    n_rows = max(len(X), 1)
    for col in cat_candidates:
        nunique = int(X[col].nunique(dropna=True))
        unique_ratio = nunique / n_rows
        if nunique > 120 or unique_ratio > 0.20:
            high_cardinality.append(col)
    if high_cardinality:
        X = X.drop(columns=high_cardinality)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    return X, y, numeric_cols, categorical_cols


def _build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> Any:
    """Build a robust preprocessing pipeline."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=20,
                    max_categories=30,
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ]
    )


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Compute RMSE, MAE, and R2."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def _compact_feature_matrix(X_matrix: Any, *, max_features: int = 140) -> np.ndarray:
    """Convert transformed matrix to dense finite array and optionally reduce dimensions."""
    if hasattr(X_matrix, "toarray"):
        arr = X_matrix.toarray()
    else:
        arr = np.asarray(X_matrix)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    arr = np.asarray(arr, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.shape[1] > max_features and arr.shape[0] > 2:
        from sklearn.decomposition import PCA

        n_components = min(max_features, arr.shape[0] - 1, arr.shape[1])
        if n_components >= 2:
            arr = PCA(n_components=n_components, random_state=RANDOM_SEED).fit_transform(arr)

    return arr


def run_manual_benchmark(df: pd.DataFrame) -> list[BenchmarkResult]:
    """Benchmark manually chosen sklearn regressors."""
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    X, y, numeric_cols, categorical_cols = _prepare_supervised_data(df)
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    model_specs = [
        ("DummyRegressor", DummyRegressor(strategy="mean")),
        ("LinearRegression", LinearRegression()),
        ("RandomForestRegressor", RandomForestRegressor(n_estimators=120, random_state=RANDOM_SEED, n_jobs=-1)),
    ]

    results: list[BenchmarkResult] = []
    for name, estimator in model_specs:
        start = time.perf_counter()
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        rmse, mae, r2 = _reg_metrics(y_test.to_numpy(), preds)
        runtime = time.perf_counter() - start

        results.append(
            BenchmarkResult(
                framework="manual",
                model=name,
                rmse=rmse,
                mae=mae,
                r2=r2,
                runtime_sec=runtime,
                status="ok",
            )
        )

    return results


def run_lazypredict_benchmark(df: pd.DataFrame) -> list[BenchmarkResult]:
    """Run LazyPredict benchmark on a moderate subsample for practical runtime."""
    from sklearn.model_selection import train_test_split

    X, y, numeric_cols, categorical_cols = _prepare_supervised_data(df)

    # LazyPredict is runtime-heavy on full large data with many estimators.
    sample_n = min(len(X), 3_000)
    if len(X) > sample_n:
        sample_idx = np.random.default_rng(RANDOM_SEED).choice(len(X), size=sample_n, replace=False)
        X = X.iloc[sample_idx].reset_index(drop=True)
        y = y.iloc[sample_idx].reset_index(drop=True)

    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
    X_matrix = _compact_feature_matrix(preprocessor.fit_transform(X), max_features=120)

    X_train, X_test, y_train, y_test = train_test_split(
        X_matrix,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )

    start = time.perf_counter()
    try:
        from lazypredict.Supervised import LazyRegressor

        reg = LazyRegressor(
            verbose=0,
            ignore_warnings=True,
            predictions=False,
            random_state=RANDOM_SEED,
            timeout=45,
            max_models=10,
            n_jobs=-1,
        )
        models_df, _ = reg.fit(X_train, X_test, y_train, y_test)
        elapsed = time.perf_counter() - start

        top_rows = models_df.head(8).reset_index()
        results = [
            BenchmarkResult(
                framework="lazypredict",
                model=str(row["Model"]),
                rmse=float(row.get("RMSE", np.nan)),
                mae=float(row.get("MAE", np.nan)),
                r2=float(row.get("R-Squared", np.nan)),
                runtime_sec=float(elapsed),
                status="ok",
            )
            for _, row in top_rows.iterrows()
        ]
        return results
    except Exception as exc:
        return [
            BenchmarkResult(
                framework="lazypredict",
                model="n/a",
                rmse=np.nan,
                mae=np.nan,
                r2=np.nan,
                runtime_sec=time.perf_counter() - start,
                status="failed",
                note=str(exc),
            )
        ]


def run_flaml_benchmark(df: pd.DataFrame) -> list[BenchmarkResult]:
    """Run FLAML AutoML benchmark."""
    from sklearn.model_selection import train_test_split

    X, y, numeric_cols, categorical_cols = _prepare_supervised_data(df)
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
    X_matrix = _compact_feature_matrix(preprocessor.fit_transform(X), max_features=140)

    X_train, X_test, y_train, y_test = train_test_split(
        X_matrix,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )

    start = time.perf_counter()
    try:
        AutoML = None
        try:
            from flaml import AutoML
        except Exception:
            try:
                from flaml.automl import AutoML  # type: ignore
            except Exception as import_exc:
                return [
                    BenchmarkResult(
                        framework="flaml",
                        model="n/a",
                        rmse=np.nan,
                        mae=np.nan,
                        r2=np.nan,
                        runtime_sec=time.perf_counter() - start,
                        status="skipped",
                        note=f"FLAML AutoML extras missing: {import_exc}",
                    )
                ]

        automl = AutoML()
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            task="regression",
            time_budget=30,
            metric="rmse",
            seed=RANDOM_SEED,
        )
        preds = automl.predict(X_test)
        rmse, mae, r2 = _reg_metrics(y_test.to_numpy(), preds)

        return [
            BenchmarkResult(
                framework="flaml",
                model=str(automl.best_estimator),
                rmse=rmse,
                mae=mae,
                r2=r2,
                runtime_sec=time.perf_counter() - start,
                status="ok",
                note=f"best_config={automl.best_config}",
            )
        ]
    except Exception as exc:
        return [
            BenchmarkResult(
                framework="flaml",
                model="n/a",
                rmse=np.nan,
                mae=np.nan,
                r2=np.nan,
                runtime_sec=time.perf_counter() - start,
                status="failed",
                note=str(exc),
            )
        ]


def _run_pycaret_job(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[str, float, float, float]:
    """Execute PyCaret workflow in a worker process."""
    from pycaret.regression import compare_models, predict_model, pull, setup

    setup(
        data=train_df,
        target=COL_TIME_TAKEN,
        session_id=RANDOM_SEED,
        verbose=False,
        fold=3,
        html=False,
        n_jobs=1,
    )
    best_model = compare_models(
        n_select=1,
        include=["lr", "ridge", "rf", "et", "lightgbm"],
        turbo=True,
        budget_time=1,
    )
    pred_df = predict_model(best_model, data=test_df.assign(**{COL_TIME_TAKEN: y_test}), verbose=False)

    rmse, mae, r2 = _reg_metrics(y_test.to_numpy(), pred_df["prediction_label"].to_numpy())
    leaderboard = pull()
    best_name = str(leaderboard.iloc[0]["Model"]) if not leaderboard.empty else str(type(best_model).__name__)
    return best_name, rmse, mae, r2


def run_pycaret_benchmark(df: pd.DataFrame) -> list[BenchmarkResult]:
    """Run PyCaret benchmark when Python runtime is supported."""
    # PyCaret 3.x currently supports Python <=3.11.
    if tuple(platform.python_version_tuple()[:2]) > ("3", "11"):
        return [
            BenchmarkResult(
                framework="pycaret",
                model="n/a",
                rmse=np.nan,
                mae=np.nan,
                r2=np.nan,
                runtime_sec=0.0,
                status="skipped",
                note="PyCaret requires Python <=3.11.",
            )
        ]

    start = time.perf_counter()
    try:
        from sklearn.model_selection import train_test_split

        X, y, _, _ = _prepare_supervised_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_SEED,
        )
        train_df = X_train.copy()
        train_df[COL_TIME_TAKEN] = y_train

        # Constrain runtime for reproducible local execution.
        sample_n = min(len(train_df), 5_000)
        if len(train_df) > sample_n:
            sample_idx = np.random.default_rng(RANDOM_SEED).choice(
                len(train_df),
                size=sample_n,
                replace=False,
            )
            train_df = train_df.iloc[np.sort(sample_idx)].reset_index(drop=True)
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_pycaret_job, train_df, X_test, y_test)
            try:
                best_name, rmse, mae, r2 = future.result(timeout=180)
            except TimeoutError:
                return [
                    BenchmarkResult(
                        framework="pycaret",
                        model="n/a",
                        rmse=np.nan,
                        mae=np.nan,
                        r2=np.nan,
                        runtime_sec=time.perf_counter() - start,
                        status="skipped",
                        note="PyCaret benchmark timed out after 180s.",
                    )
                ]

        return [
            BenchmarkResult(
                framework="pycaret",
                model=best_name,
                rmse=rmse,
                mae=mae,
                r2=r2,
                runtime_sec=time.perf_counter() - start,
                status="ok",
            )
        ]
    except Exception as exc:
        return [
            BenchmarkResult(
                framework="pycaret",
                model="n/a",
                rmse=np.nan,
                mae=np.nan,
                r2=np.nan,
                runtime_sec=time.perf_counter() - start,
                status="failed",
                note=str(exc),
            )
        ]


def run_full_benchmark(df: pd.DataFrame, *, save_path: Path | None = None) -> pd.DataFrame:
    """Run full downstream benchmark suite and persist a normalized table."""
    rows: list[BenchmarkResult] = []

    rows.extend(run_manual_benchmark(df))
    rows.extend(run_lazypredict_benchmark(df))
    rows.extend(run_flaml_benchmark(df))
    rows.extend(run_pycaret_benchmark(df))

    out_df = pd.DataFrame([r.__dict__ for r in rows])
    target_path = save_path or BENCHMARK_REPORT_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(target_path, index=False)

    return out_df
