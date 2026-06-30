"""Training and benchmarking pipeline for California Housing deployment project."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from pipeline.config import (
    FEATURE_COLUMNS,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
    get_project_paths,
)

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@dataclass
class TrainingArtifacts:
    """Container for produced benchmark outputs."""

    ranking: pd.DataFrame
    lazy_ranking: pd.DataFrame
    metadata: dict[str, Any]
    model: Any
    model_name: str


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute numerically stable MAPE for near-zero targets."""
    epsilon = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))))


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return standard regression metrics used by the project."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R²": float(r2_score(y_true, y_pred)),
        "MAPE": float(_safe_mape(y_true, y_pred)),
    }


def _build_manual_models(profile: str) -> dict[str, Any]:
    """Build required model family with profile-specific budgets."""
    if profile == "deep":
        n_estimators = 600
    elif profile == "fast":
        n_estimators = 180
    else:
        n_estimators = 350

    models: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=10_000),
        "Random Forest": RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=max(100, n_estimators // 2),
            random_state=RANDOM_SEED,
        ),
    }

    try:
        import xgboost as xgb

        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=0,
        )
    except Exception as exc:  # pragma: no cover - dependency optional runtime
        LOGGER.warning("Skipping XGBoost benchmark: %s", exc)

    try:
        import lightgbm as lgb

        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1,
        )
    except Exception as exc:  # pragma: no cover - dependency optional runtime
        LOGGER.warning("Skipping LightGBM benchmark: %s", exc)

    try:
        from catboost import CatBoostRegressor

        models["CatBoost"] = CatBoostRegressor(
            iterations=n_estimators,
            depth=8,
            learning_rate=0.05,
            random_seed=RANDOM_SEED,
            verbose=0,
            loss_function="RMSE",
        )
    except Exception as exc:  # pragma: no cover - dependency optional runtime
        LOGGER.warning("Skipping CatBoost benchmark: %s", exc)

    return models


def _run_lazypredict(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Run LazyPredict for broad baseline comparison table."""
    try:
        from lazypredict.Supervised import LazyRegressor

        lazy = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
        models_df, _ = lazy.fit(x_train, x_test, y_train, y_test)
    except Exception as exc:  # pragma: no cover - dependency/runtime variability
        LOGGER.warning("LazyPredict failed; returning empty table: %s", exc)
        return pd.DataFrame(columns=["Model", "Adjusted R-Squared", "R-Squared", "RMSE", "Time Taken"])

    if "Model" not in models_df.columns:
        models_df = models_df.reset_index().rename(columns={"index": "Model"})
    return models_df


def _run_flaml(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    time_budget_seconds: int,
) -> tuple[pd.DataFrame, Any | None]:
    """Run FLAML AutoML and return metrics table + fitted estimator."""
    try:
        from flaml import AutoML
    except Exception as exc:  # pragma: no cover - dependency optional runtime
        LOGGER.warning("FLAML unavailable: %s", exc)
        return pd.DataFrame(), None

    automl = AutoML()
    settings = {
        "time_budget": time_budget_seconds,
        "task": "regression",
        "metric": "r2",
        "seed": RANDOM_SEED,
        "log_file_name": str(get_project_paths().benchmarks / "flaml.log"),
    }
    automl.fit(X_train=x_train, y_train=y_train, **settings)
    preds = automl.predict(x_test)
    metrics = _compute_metrics(y_test.to_numpy(), np.asarray(preds))
    result = pd.DataFrame(
        [
            {
                "Model": "FLAML AutoML",
                "Source": "FLAML",
                **metrics,
            }
        ]
    )
    return result, automl


def _run_pycaret(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, Any | None]:
    """Run PyCaret comparison workflow and return metrics + best model."""
    try:
        from pycaret.regression import compare_models, predict_model, pull, setup
    except Exception as exc:  # pragma: no cover - dependency optional runtime
        LOGGER.warning("PyCaret unavailable: %s", exc)
        return pd.DataFrame(), None

    # PyCaret mutates global experiment state by design.
    setup(
        data=train_df,
        target=TARGET_COLUMN,
        session_id=RANDOM_SEED,
        fold=3,
        html=False,
        verbose=False,
        n_jobs=-1,
    )
    best_model = compare_models(sort="R2", n_select=1)
    leaderboard = pull()
    if isinstance(best_model, list):
        best_model = best_model[0]
    pycaret_preds = predict_model(best_model, data=test_df)["prediction_label"].to_numpy()
    metrics = _compute_metrics(test_df[TARGET_COLUMN].to_numpy(), pycaret_preds)
    result = pd.DataFrame(
        [
            {
                "Model": f"PyCaret {leaderboard.iloc[0]['Model']}" if not leaderboard.empty else "PyCaret Best",
                "Source": "PyCaret",
                **metrics,
            }
        ]
    )
    return result, best_model


def _create_dataset_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Load California Housing dataset and split with reproducible seed."""
    frame = fetch_california_housing(as_frame=True).frame
    x = frame[FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )
    return x_train, x_test, y_train, y_test, frame


def _plot_eda(frame: pd.DataFrame, paths: Path) -> None:
    """Generate basic EDA visuals used in notebooks and README."""
    plt.figure(figsize=(10, 6))
    corr = frame.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.2)
    plt.title("California Housing Correlation Matrix")
    plt.tight_layout()
    plt.savefig(paths / "correlation-matrix.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for idx, column in enumerate(FEATURE_COLUMNS):
        ax = axes[idx // 4, idx % 4]
        sns.histplot(frame[column], kde=True, ax=ax)
        ax.set_title(column)
    fig.suptitle("Feature Distributions")
    fig.tight_layout()
    fig.savefig(paths / "feature-distributions.png", dpi=150)
    plt.close(fig)


def _plot_model_comparison(ranking: pd.DataFrame, paths: Path) -> None:
    """Create ranking chart for R² comparison."""
    plot_df = ranking.sort_values("R²", ascending=False).copy()
    plt.figure(figsize=(12, 5))
    bars = plt.bar(plot_df["Model"], plot_df["R²"], color=sns.color_palette("viridis", len(plot_df)))
    plt.xticks(rotation=40, ha="right")
    plt.ylabel("R²")
    plt.title("Model Comparison by R²")
    for bar, value in zip(bars, plot_df["R²"]):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(paths / "model-comparison-r2.png", dpi=150)
    plt.close()


def _plot_shap(
    best_model: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    paths: Path,
    profile: str,
) -> np.ndarray | None:
    """Create SHAP summary and feature-importance plots if backend supports model."""
    try:
        import shap
    except Exception as exc:  # pragma: no cover - dependency optional runtime
        LOGGER.warning("SHAP unavailable, skipping explainability plots: %s", exc)
        return None

    train_sample = x_train.sample(min(300, len(x_train)), random_state=RANDOM_SEED)
    shap_test_size = {"fast": 80, "balanced": 180, "deep": 300}[profile]
    test_sample = x_test.sample(min(shap_test_size, len(x_test)), random_state=RANDOM_SEED)

    if hasattr(best_model, "feature_importances_"):
        try:
            explainer = shap.TreeExplainer(best_model)
            values = np.asarray(explainer.shap_values(test_sample))
            if values.ndim == 3:
                values = values[0]
            plt.figure()
            shap.summary_plot(values, test_sample, show=False)
        except Exception:
            fallback_sample = test_sample.sample(min(40, len(test_sample)), random_state=RANDOM_SEED)
            explainer = shap.Explainer(best_model.predict, train_sample, algorithm="permutation")
            shap_values = explainer(fallback_sample, max_evals=(2 * len(FEATURE_COLUMNS) + 1))
            values = np.asarray(shap_values.values)
            test_sample = fallback_sample
            plt.figure()
            shap.summary_plot(shap_values, test_sample, show=False)
    else:
        explainer = shap.Explainer(best_model.predict, train_sample)
        shap_values = explainer(test_sample, max_evals=(2 * len(FEATURE_COLUMNS) + 1))
        values = np.asarray(shap_values.values)
        plt.figure()
        shap.summary_plot(shap_values, test_sample, show=False)

    plt.tight_layout()
    plt.savefig(paths / "shap-summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    importance = np.mean(np.abs(values), axis=0)
    order = np.argsort(importance)
    sorted_importance = importance[order]
    sorted_features = np.array(FEATURE_COLUMNS)[order]

    plt.figure(figsize=(8, 5))
    plt.barh(sorted_features, sorted_importance, color=sns.color_palette("mako", len(sorted_features)))
    plt.xlabel("Mean |SHAP value|")
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(paths / "shap-importance.png", dpi=150)
    plt.close()

    return train_sample.to_numpy(dtype=np.float32)


def _try_export_onnx(model: Any, output_path: Path) -> tuple[bool, str]:
    """Export ONNX demo artifact when estimator is converter-compatible."""
    try:
        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import FloatTensorType
    except Exception as exc:  # pragma: no cover - dependency optional runtime
        return False, f"skl2onnx unavailable: {exc}"

    try:
        onnx_model = to_onnx(
            model,
            initial_types=[("float_input", FloatTensorType([None, len(FEATURE_COLUMNS)]))],
            target_opset=17,
        )
        output_path.write_bytes(onnx_model.SerializeToString())
        return True, "Exported successfully"
    except Exception as exc:  # pragma: no cover - model compatibility runtime
        return False, f"ONNX export skipped: {exc}"


def run_training_pipeline(
    profile: str = "balanced",
    include_pycaret: bool | None = None,
) -> TrainingArtifacts:
    """Execute complete benchmark flow and save artifacts for serving and tutorials.

    Args:
        profile: Runtime budget profile. One of "fast", "balanced", "deep".
        include_pycaret: Override for PyCaret run behavior. If None, defaults to
            False for fast profile and True otherwise.

    Returns:
        TrainingArtifacts with ranking tables, chosen model, and metadata.
    """
    if profile not in {"fast", "balanced", "deep"}:
        raise ValueError("profile must be one of: fast, balanced, deep")
    if include_pycaret is None:
        include_pycaret = profile in {"balanced", "deep"}

    paths = get_project_paths()
    x_train, x_test, y_train, y_test, frame = _create_dataset_splits()
    _plot_eda(frame, paths.figures)

    manual_models = _build_manual_models(profile)
    manual_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    for name, model in manual_models.items():
        LOGGER.info("Training manual model: %s", name)
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        fitted_models[name] = model
        manual_rows.append(
            {
                "Model": name,
                "Source": "Manual",
                **_compute_metrics(y_test.to_numpy(), np.asarray(preds)),
            }
        )

    manual_ranking = pd.DataFrame(manual_rows)

    include_lazy = profile in {"balanced", "deep"}
    if include_lazy:
        lazy_ranking = _run_lazypredict(x_train, x_test, y_train, y_test)
    else:
        lazy_ranking = pd.DataFrame(columns=["Model", "Adjusted R-Squared", "R-Squared", "RMSE", "Time Taken"])
    lazy_ranking.to_csv(paths.benchmarks / "lazypredict_ranking.csv", index=False)

    if profile in {"balanced", "deep"}:
        flaml_budget = {"balanced": 120, "deep": 1800}[profile]
        flaml_ranking, flaml_model = _run_flaml(x_train, y_train, x_test, y_test, flaml_budget)
        if flaml_model is not None:
            fitted_models["FLAML AutoML"] = flaml_model
    else:
        flaml_ranking = pd.DataFrame()

    pycaret_ranking = pd.DataFrame()
    if include_pycaret:
        pycaret_ranking, pycaret_model = _run_pycaret(
            train_df=pd.concat([x_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1),
            test_df=pd.concat([x_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1),
        )
        if pycaret_model is not None and not pycaret_ranking.empty:
            fitted_models[str(pycaret_ranking.iloc[0]["Model"])] = pycaret_model
            pycaret_ranking.to_csv(paths.benchmarks / "pycaret_ranking.csv", index=False)

    ranking = pd.concat([manual_ranking, flaml_ranking, pycaret_ranking], ignore_index=True)
    ranking = ranking.sort_values(["R²", "RMSE"], ascending=[False, True]).reset_index(drop=True)
    ranking.to_csv(paths.benchmarks / "model_ranking.csv", index=False)
    _plot_model_comparison(ranking, paths.figures)

    overall_best_row = ranking.iloc[0]
    overall_best_name = str(overall_best_row["Model"])

    serving_allowed_models = {
        "Linear Regression",
        "Ridge",
        "Lasso",
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting",
        "LightGBM",
        "XGBoost",
    }
    serving_candidates = ranking[ranking["Model"].isin(serving_allowed_models)]
    if serving_candidates.empty:
        serving_row = manual_ranking.sort_values(["R²", "RMSE"], ascending=[False, True]).iloc[0]
    else:
        serving_row = serving_candidates.sort_values(["R²", "RMSE"], ascending=[False, True]).iloc[0]

    best_name = str(serving_row["Model"])
    if best_name not in fitted_models:
        # Fallback for label mismatch in optional tool outputs.
        best_name = manual_ranking.sort_values(["R²", "RMSE"], ascending=[False, True]).iloc[0]["Model"]
    best_model = fitted_models[best_name]

    model_path = paths.models / "best_model.joblib"
    joblib.dump(best_model, model_path)

    background = _plot_shap(best_model, x_train, x_test, paths.figures, profile=profile)
    if background is not None:
        np.save(paths.models / "background_sample.npy", background)

    onnx_model_name = best_name
    onnx_ok, onnx_message = _try_export_onnx(best_model, paths.models / "model_demo.onnx")
    if not onnx_ok:
        # Demonstrate ONNX conversion with converter-compatible baseline model.
        onnx_demo_model = Ridge(alpha=1.0).fit(x_train, y_train)
        onnx_model_name = "Ridge (ONNX demo fallback)"
        onnx_ok, onnx_message = _try_export_onnx(onnx_demo_model, paths.models / "model_demo.onnx")
        if not onnx_ok:
            LOGGER.warning(onnx_message)

    metadata = {
        "overall_best_model": overall_best_name,
        "overall_best_source": str(overall_best_row["Source"]),
        "best_model": best_name,
        "best_model_source": str(serving_row["Source"]),
        "features": FEATURE_COLUMNS,
        "profile": profile,
        "metrics": {k: float(serving_row[k]) for k in ["MAE", "MSE", "RMSE", "R²", "MAPE"]},
        "overall_ranking_top3": ranking.head(3).to_dict(orient="records"),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "n_features": len(FEATURE_COLUMNS),
        "seed": RANDOM_SEED,
        "onnx_export": {"success": onnx_ok, "model": onnx_model_name, "message": onnx_message},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (paths.models / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    return TrainingArtifacts(
        ranking=ranking,
        lazy_ranking=lazy_ranking,
        metadata=metadata,
        model=best_model,
        model_name=best_name,
    )
