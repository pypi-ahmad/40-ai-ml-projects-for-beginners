"""Model catalog for benchmarking and model selection."""

from __future__ import annotations

import importlib
import logging

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

logger = logging.getLogger(__name__)


def build_model_catalog(random_seed: int) -> dict[str, object]:
    """Build deterministic model candidates, including optional extras when installed."""
    models: dict[str, object] = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.01, random_state=random_seed, max_iter=80_000, tol=1e-3),
        "elasticnet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=random_seed, max_iter=80_000, tol=1e-3),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=random_seed,
            n_jobs=-1,
            min_samples_leaf=1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            random_state=random_seed,
            n_jobs=-1,
            min_samples_leaf=1,
        ),
    }

    optional_factories: dict[str, tuple[str, str, dict[str, int | float]]] = {
        "xgboost": (
            "xgboost",
            "XGBRegressor",
            {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": random_seed,
                "objective": "reg:squarederror",
                "n_jobs": -1,
            },
        ),
        "lightgbm": (
            "lightgbm",
            "LGBMRegressor",
            {
                "n_estimators": 600,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": random_seed,
                "n_jobs": -1,
            },
        ),
        "catboost": (
            "catboost",
            "CatBoostRegressor",
            {
                "iterations": 600,
                "learning_rate": 0.05,
                "depth": 6,
                "random_seed": random_seed,
                "verbose": 0,
            },
        ),
    }

    for name, (module_name, class_name, kwargs) in optional_factories.items():
        model_class = _load_optional_model(module_name, class_name)
        if model_class is None:
            continue
        models[name] = model_class(**kwargs)

    return models


def _load_optional_model(module_name: str, class_name: str):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        logger.info("Optional model %s not available", module_name)
        return None

    return getattr(module, class_name)
