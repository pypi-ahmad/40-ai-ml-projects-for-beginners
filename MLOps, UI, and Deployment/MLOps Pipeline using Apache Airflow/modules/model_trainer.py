"""Model benchmarking, AutoML integration, champion selection, and MLflow logging."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit

from .data_loader import save_csv, save_json
from .feature_engineering import temporal_train_test_split
from .settings import load_config, resolve_path

warnings.filterwarnings("ignore")


@dataclass
class CandidateResult:
    """Model candidate with comparable evaluation metrics."""

    name: str
    model: Any
    metrics: dict[str, float]
    source: str


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    denom = np.clip(np.abs(y_true), 1e-6, None)
    mape = float(np.mean(np.abs(err) / denom) * 100)
    y_var = float(np.var(y_true))
    r2 = float(1.0 - (np.var(err) / y_var)) if y_var > 0 else 0.0
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "mape": mape}


def _clone_estimator(estimator: Any) -> Any:
    try:
        return clone(estimator)
    except Exception:
        cls = estimator.__class__
        if hasattr(estimator, "get_params"):
            return cls(**estimator.get_params())
        return estimator


def _model_catalog(random_state: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "Lasso": Lasso(alpha=0.01, random_state=random_state),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=random_state),
        "RandomForest": RandomForestRegressor(n_estimators=220, random_state=random_state, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=220, random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
    }

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=260,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            objective="reg:squarederror",
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=260,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            verbose=-1,
        )
    except Exception:
        pass

    try:
        from catboost import CatBoostRegressor

        models["CatBoost"] = CatBoostRegressor(
            iterations=260,
            learning_rate=0.05,
            depth=6,
            random_seed=random_state,
            verbose=False,
        )
    except Exception:
        pass

    return models


def _optional_dependency_status() -> dict[str, str]:
    """Report optional benchmarking dependency availability."""
    status: dict[str, str] = {}
    try:
        from lazypredict.Supervised import LazyRegressor  # noqa: F401

        status["lazypredict"] = "available"
    except Exception as exc:
        status["lazypredict"] = f"unavailable ({exc.__class__.__name__})"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from flaml import AutoML  # noqa: F401

        status["flaml"] = "available"
    except Exception as exc:
        status["flaml"] = f"unavailable ({exc.__class__.__name__})"

    try:
        from pycaret.regression import setup  # noqa: F401

        status["pycaret"] = "available"
    except Exception as exc:
        status["pycaret"] = f"unavailable ({exc.__class__.__name__})"
    return status


def _time_series_cv(estimator: Any, X: pd.DataFrame, y: pd.Series, n_splits: int) -> tuple[float, float]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rmses = []
    fold_r2s = []
    for train_idx, val_idx in tscv.split(X):
        est = _clone_estimator(estimator)
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        est.fit(X_tr, y_tr)
        y_hat = est.predict(X_val)
        m = _metrics(y_val.to_numpy(), np.asarray(y_hat))
        fold_rmses.append(m["rmse"])
        fold_r2s.append(m["r2"])
    return float(np.mean(fold_rmses)), float(np.mean(fold_r2s))


def benchmark_core_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
    cv_splits: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Benchmark required regression model suite."""
    rows = []
    fitted_models: dict[str, Any] = {}

    for name, estimator in _model_catalog(random_state=random_state).items():
        model = _clone_estimator(estimator)
        model.fit(X_train, y_train)
        pred_test = np.asarray(model.predict(X_test))
        test_metrics = _metrics(y_test.to_numpy(), pred_test)
        cv_rmse, cv_r2 = _time_series_cv(estimator, X_train, y_train, n_splits=cv_splits)

        row = {
            "model": name,
            "cv_rmse": cv_rmse,
            "cv_r2": cv_r2,
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_r2": test_metrics["r2"],
            "test_mape": test_metrics["mape"],
            "source": "core_benchmark",
        }
        rows.append(row)
        fitted_models[name] = model

    leaderboard = pd.DataFrame(rows).sort_values("test_rmse").reset_index(drop=True)
    return leaderboard, fitted_models


def run_lazypredict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    max_rows: int = 1200,
) -> pd.DataFrame:
    """Run LazyPredict quick benchmark and normalize output."""
    try:
        from lazypredict.Supervised import LazyRegressor

        # Bound runtime by sampling the latest training/test windows.
        if len(X_train) > max_rows:
            X_train = X_train.tail(max_rows)
            y_train = y_train.tail(max_rows)
        test_rows = max(120, min(len(X_test), max_rows // 3))
        X_test = X_test.tail(test_rows)
        y_test = y_test.tail(test_rows)

        reg = LazyRegressor(verbose=0, ignore_warnings=True)
        results, _ = reg.fit(X_train, X_test, y_train, y_test)
        out = results.reset_index().rename(columns={"index": "model"})
        cols = [c for c in out.columns if c in ["model", "RMSE", "MAE", "R-Squared", "Time Taken"]]
        out = out[cols]
        rename_map = {
            "RMSE": "lazy_rmse",
            "MAE": "lazy_mae",
            "R-Squared": "lazy_r2",
            "Time Taken": "lazy_time_sec",
        }
        out = out.rename(columns=rename_map)
        return out
    except Exception:
        return pd.DataFrame()


def run_flaml(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    time_budget_sec: int,
    random_state: int,
) -> CandidateResult | None:
    """Run FLAML AutoML and return evaluated candidate."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from flaml import AutoML

        automl = AutoML()
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            task="regression",
            time_budget=time_budget_sec,
            seed=random_state,
            verbose=0,
        )
        model = automl.model.estimator if hasattr(automl.model, "estimator") else automl.model
        y_pred = np.asarray(model.predict(X_test))
        metrics = _metrics(y_test.to_numpy(), y_pred)
        metrics["best_loss"] = float(automl.best_loss)
        return CandidateResult(name="FLAML", model=model, metrics=metrics, source="flaml")
    except Exception:
        return None


def run_pycaret(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    fold: int,
    session_id: int,
) -> CandidateResult | None:
    """Run PyCaret compare+tune+finalize workflow and return candidate."""
    try:
        from pycaret.regression import compare_models, finalize_model, predict_model, setup, tune_model

        setup(
            data=train_df,
            target=target_col,
            fold=fold,
            session_id=session_id,
            verbose=False,
            html=False,
            n_jobs=-1,
            data_split_shuffle=False,
            fold_strategy="timeseries",
            fold_shuffle=False,
        )
        best = compare_models(sort="RMSE", n_select=1)
        tuned = tune_model(best, optimize="RMSE", choose_better=True)
        final = finalize_model(tuned)

        pred_df = predict_model(final, data=test_df)
        pred_col = "prediction_label" if "prediction_label" in pred_df.columns else "Label"
        y_pred = pred_df[pred_col].to_numpy()
        y_true = test_df[target_col].to_numpy()

        return CandidateResult(name="PyCaret", model=final, metrics=_metrics(y_true, y_pred), source="pycaret")
    except Exception:
        return None


def _fit_for_selection(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    fitted = _clone_estimator(model)
    fitted.fit(X_train, y_train)
    return fitted


def _log_mlflow_run(
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifact_paths: list[str],
    mlruns_dir: Path,
) -> str | None:
    try:
        import mlflow

        mlflow.set_tracking_uri(f"file://{mlruns_dir}")
        mlflow.set_experiment("screentime-forecasting")
        with mlflow.start_run(run_name=run_name) as run:
            for key, value in params.items():
                mlflow.log_param(key, value)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))
            for path in artifact_paths:
                p = Path(path)
                if p.exists():
                    mlflow.log_artifact(str(p))
            return run.info.run_id
    except Exception:
        return None


def run_training_pipeline(
    feature_df: pd.DataFrame,
    selected_features: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train/evaluate/register-ready model bundle.

    Returns dict with champion model, metrics, leaderboards, split data, and run metadata.
    """
    config = config or load_config()
    project_cfg = config["project"]
    train_cfg = config["training"]

    target_col = "target_next_day"
    date_col = str(project_cfg["date_col"])
    dependency_status = _optional_dependency_status()

    train_df, test_df = temporal_train_test_split(
        feature_df,
        date_col=date_col,
        test_size=float(train_cfg["test_size"]),
    )

    if selected_features:
        raw_feature_cols = [col for col in selected_features if col in feature_df.columns]
    else:
        blocked = {target_col, str(project_cfg["target_col"]), date_col, str(project_cfg["group_col"])}
        raw_feature_cols = [
            col
            for col in feature_df.select_dtypes(include=[np.number]).columns
            if col not in blocked
        ]

    if not raw_feature_cols:
        raise ValueError("No feature columns available for training")

    X_train = train_df[raw_feature_cols].replace([np.inf, -np.inf], np.nan)
    y_train = train_df[target_col].astype(float)
    X_test = test_df[raw_feature_cols].replace([np.inf, -np.inf], np.nan)
    y_test = test_df[target_col].astype(float)

    # Normalize feature names for model backends that reject spaces/symbols.
    cleaned_names = {
        col: (
            col.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
            .replace("/", "_")
        )
        for col in X_train.columns
    }
    X_train = X_train.rename(columns=cleaned_names)
    X_test = X_test.rename(columns=cleaned_names)
    feature_cols = [cleaned_names[c] for c in raw_feature_cols]

    # Leakage-safe imputation: fit on train, apply to test.
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_cols, index=X_test.index)

    leaderboard, fitted_models = benchmark_core_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=int(project_cfg["random_state"]),
        cv_splits=int(train_cfg["cv_splits"]),
    )

    lazy_df = pd.DataFrame()
    if bool(train_cfg["lazy_enabled"]):
        lazy_df = run_lazypredict(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            max_rows=int(train_cfg.get("lazy_max_rows", 1200)),
        )

    candidates: list[CandidateResult] = []

    best_core = leaderboard.iloc[0]
    best_core_name = str(best_core["model"])
    best_core_model = fitted_models[best_core_name]
    best_core_metrics = {
        "mae": float(best_core["test_mae"]),
        "mse": float(best_core["test_rmse"] ** 2),
        "rmse": float(best_core["test_rmse"]),
        "r2": float(best_core["test_r2"]),
        "mape": float(best_core["test_mape"]),
    }
    candidates.append(CandidateResult(name=best_core_name, model=best_core_model, metrics=best_core_metrics, source="core_benchmark"))

    if bool(train_cfg["flaml_enabled"]):
        flaml_candidate = run_flaml(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            time_budget_sec=int(train_cfg["flaml_time_budget_sec"]),
            random_state=int(project_cfg["random_state"]),
        )
        if flaml_candidate:
            candidates.append(flaml_candidate)

    if bool(train_cfg["pycaret_enabled"]):
        train_pycaret = X_train.copy()
        train_pycaret[target_col] = y_train.to_numpy()
        test_pycaret = X_test.copy()
        test_pycaret[target_col] = y_test.to_numpy()
        pycaret_candidate = run_pycaret(
            train_df=train_pycaret,
            test_df=test_pycaret,
            target_col=target_col,
            fold=int(train_cfg["pycaret_fold"]),
            session_id=int(train_cfg["pycaret_session_id"]),
        )
        if pycaret_candidate:
            candidates.append(pycaret_candidate)

    champion = min(candidates, key=lambda c: c.metrics["rmse"])

    final_model = _fit_for_selection(champion.model, X_train, y_train)
    final_pred = np.asarray(final_model.predict(X_test))
    final_metrics = _metrics(y_test.to_numpy(), final_pred)

    scoreboard_rows = []
    for candidate in candidates:
        row = {"candidate": candidate.name, "source": candidate.source}
        row.update(candidate.metrics)
        scoreboard_rows.append(row)
    candidate_scoreboard = pd.DataFrame(scoreboard_rows).sort_values("rmse", ascending=True).reset_index(drop=True)

    leaderboard_path = resolve_path(config, "model_leaderboard")
    metrics_path = resolve_path(config, "best_metrics")
    save_csv(leaderboard, leaderboard_path)
    save_json(
        {
            "champion": champion.name,
            "champion_source": champion.source,
            "champion_metrics": final_metrics,
            "features": feature_cols,
            "dependency_status": dependency_status,
        },
        metrics_path,
    )

    mlruns_dir = resolve_path(config, "mlruns_dir")
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    run_id = _log_mlflow_run(
        run_name=f"champion-{champion.name}",
        params={
            "champion_model": champion.name,
            "champion_source": champion.source,
            "feature_count": len(feature_cols),
            "test_size": float(train_cfg["test_size"]),
        },
        metrics=final_metrics,
        artifact_paths=[str(leaderboard_path), str(metrics_path)],
        mlruns_dir=mlruns_dir,
    )

    return {
        "model": final_model,
        "model_name": champion.name,
        "model_source": champion.source,
        "metrics": final_metrics,
        "leaderboard": leaderboard,
        "candidate_scoreboard": candidate_scoreboard,
        "lazy_leaderboard": lazy_df,
        "feature_columns": feature_cols,
        "train_df": train_df,
        "test_df": test_df,
        "X_test": X_test,
        "y_test": y_test,
        "mlflow_run_id": run_id,
        "dependency_status": dependency_status,
        "leaderboard_path": str(leaderboard_path),
        "metrics_path": str(metrics_path),
    }
