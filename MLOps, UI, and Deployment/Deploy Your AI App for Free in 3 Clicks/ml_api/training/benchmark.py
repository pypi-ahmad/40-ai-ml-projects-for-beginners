"""Model benchmarking and selection logic for API model serving."""

from __future__ import annotations

from dataclasses import dataclass
import time

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # pragma: no cover
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) ** 0.5
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_api.training.data import DatasetBundle
from ml_api.training.feature_spec import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@dataclass(frozen=True)
class ModelScore:
    model_name: str
    val_rmse: float
    val_mae: float
    val_r2: float
    test_rmse: float
    test_mae: float
    test_r2: float
    fit_seconds: float


@dataclass(frozen=True)
class BenchmarkResult:
    best_model_name: str
    best_pipeline: Pipeline
    scores: list[ModelScore]


def benchmark_models(model_catalog: dict[str, object], dataset: DatasetBundle) -> BenchmarkResult:
    """Train and compare candidate regressors using fixed splits."""
    preprocessor = build_preprocessor()
    split = dataset.split

    scored: list[tuple[ModelScore, Pipeline]] = []

    for name, estimator in model_catalog.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", estimator),
            ]
        )

        started = time.perf_counter()
        pipeline.fit(split.x_train, split.y_train)
        fit_seconds = time.perf_counter() - started

        val_pred = pipeline.predict(split.x_val)
        test_pred = pipeline.predict(split.x_test)

        score = ModelScore(
            model_name=name,
            val_rmse=round(float(root_mean_squared_error(split.y_val, val_pred)), 4),
            val_mae=round(float(mean_absolute_error(split.y_val, val_pred)), 4),
            val_r2=round(float(r2_score(split.y_val, val_pred)), 4),
            test_rmse=round(float(root_mean_squared_error(split.y_test, test_pred)), 4),
            test_mae=round(float(mean_absolute_error(split.y_test, test_pred)), 4),
            test_r2=round(float(r2_score(split.y_test, test_pred)), 4),
            fit_seconds=round(fit_seconds, 4),
        )
        scored.append((score, pipeline))

    scored.sort(key=lambda item: item[0].val_rmse)
    best_score, best_pipeline = scored[0]
    return BenchmarkResult(
        best_model_name=best_score.model_name,
        best_pipeline=best_pipeline,
        scores=[item[0] for item in scored],
    )


def build_preprocessor() -> ColumnTransformer:
    """Build train-only fitted preprocessing graph for fair model comparison."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def scores_to_frame(scores: list[ModelScore]) -> pd.DataFrame:
    """Convert scores into tabular artifact for reports and notebooks."""
    return pd.DataFrame([
        {
            "model_name": score.model_name,
            "val_rmse": score.val_rmse,
            "val_mae": score.val_mae,
            "val_r2": score.val_r2,
            "test_rmse": score.test_rmse,
            "test_mae": score.test_mae,
            "test_r2": score.test_r2,
            "fit_seconds": score.fit_seconds,
        }
        for score in scores
    ])
