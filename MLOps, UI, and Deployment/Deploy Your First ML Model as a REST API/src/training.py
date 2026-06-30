"""Core training pipeline for California Housing regression."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import BENCHMARKS_DIR, FEATURE_NAMES, TARGET_NAME
from src.evaluation import RegressionMetrics, compute_regression_metrics, rank_models


@dataclass(slots=True)
class TrainedModel:
    """Result object returned by train_best_model."""

    model_name: str
    model: Any
    val_metrics: RegressionMetrics
    test_metrics: RegressionMetrics
    train_seconds: float


def _linear_pipeline(model: Any) -> Pipeline:
    """Build scaler + regressor pipeline for linear models."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", model),
        ]
    )


def build_candidate_models(random_state: int = 42) -> dict[str, Any]:
    """Return baseline+tree+boosting candidate regressors."""
    models: dict[str, Any] = {
        "LinearRegression": _linear_pipeline(LinearRegression()),
        "Ridge": _linear_pipeline(Ridge(alpha=1.0, random_state=random_state)),
        "Lasso": _linear_pipeline(Lasso(alpha=0.0005, random_state=random_state, max_iter=10000)),
        "ElasticNet": _linear_pipeline(
            ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=random_state, max_iter=10000)
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=120,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=160,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=48,
            random_state=random_state,
        )
    except Exception:
        pass

    try:
        from catboost import CatBoostRegressor

        models["CatBoost"] = CatBoostRegressor(
            iterations=250,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            random_state=random_state,
            verbose=False,
        )
    except Exception:
        pass

    return models


def train_and_rank_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train all candidates and return trained models + sorted ranking."""
    models = build_candidate_models(random_state=random_state)
    trained: dict[str, Any] = {}
    metrics: dict[str, RegressionMetrics] = {}

    for name, model in models.items():
        print(f"Training candidate model: {name}", flush=True)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        metrics[name] = compute_regression_metrics(y_val, preds)
        trained[name] = model

    ranking = rank_models(metrics)
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(BENCHMARKS_DIR / "manual_model_ranking.csv", index=False)
    return trained, ranking


def train_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> tuple[TrainedModel, pd.DataFrame]:
    """Select top validation model then refit on train+val and evaluate test."""
    trained, ranking = train_and_rank_models(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        random_state=random_state,
    )

    if ranking.empty:
        raise RuntimeError("No models were trained. Install required training extras.")

    best_name = str(ranking.iloc[0]["model_name"])
    candidate = trained[best_name]

    X_fit = pd.concat([X_train, X_val], axis=0)
    y_fit = pd.concat([y_train, y_val], axis=0)

    start = time.perf_counter()
    candidate.fit(X_fit, y_fit)
    train_seconds = time.perf_counter() - start

    val_preds = candidate.predict(X_val)
    test_preds = candidate.predict(X_test)

    val_metrics = compute_regression_metrics(y_val, val_preds)
    test_metrics = compute_regression_metrics(y_test, test_preds)

    result = TrainedModel(
        model_name=best_name,
        model=candidate,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_seconds=train_seconds,
    )
    return result, ranking


def build_metadata(
    trained: TrainedModel,
    ranking: pd.DataFrame,
    n_train: int,
    n_val: int,
    n_test: int,
    *,
    feature_schema_version: str,
) -> dict[str, Any]:
    """Create metadata payload shared by serving + docs."""
    top_table = ranking.head(5).to_dict(orient="records")
    return {
        "problem_type": "regression",
        "dataset_name": "California Housing",
        "target_name": TARGET_NAME,
        "feature_names": FEATURE_NAMES,
        "feature_schema_version": feature_schema_version,
        "model_name": trained.model_name,
        "model_version": "1.0.0",
        "model_library": type(trained.model).__module__,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "mae": trained.test_metrics.mae,
        "mse": trained.test_metrics.mse,
        "rmse": trained.test_metrics.rmse,
        "r2": trained.test_metrics.r2,
        "mape": trained.test_metrics.mape,
        "validation_mae": trained.val_metrics.mae,
        "validation_mse": trained.val_metrics.mse,
        "validation_rmse": trained.val_metrics.rmse,
        "validation_r2": trained.val_metrics.r2,
        "validation_mape": trained.val_metrics.mape,
        "train_seconds": trained.train_seconds,
        "top_models": top_table,
        "assumptions": [
            "Train/validation/test split done with random_state=42",
            "Features expected in canonical California Housing order",
            "Model trained for local CPU inference",
        ],
        "limitations": [
            "Dataset represents 1990 California census block groups",
            "No external drift monitoring in local mode",
        ],
    }


def dump_training_summary(metadata: dict[str, Any], output_path: Path) -> None:
    """Write human-readable training summary for README linking."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
