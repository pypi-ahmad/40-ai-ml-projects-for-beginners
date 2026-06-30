from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.backtesting import BacktestEngine
from src.baseline_models import BaselineResult, run_automl_suite, run_baseline_benchmark
from src.data_loader import build_horizon_target, load_stock_data, split_data
from src.deep_learning import DeepLearningConfig, run_deep_learning_benchmark
from src.evaluation import evaluate_regression, metrics_table, regression_metrics
from src.feature_importance import FeatureImportanceAnalyzer, importance_to_frame, shap_values
from src.features import FeaturePipeline
from src.hybrid_models import (
    AdvancedStackingEnsemble,
    make_stacking_meta_learner,
    weighted_ensemble,
)
from src.utils import ensure_dir, set_global_seed
from src.visualization import (
    plot_ensemble_contributions,
    plot_error_distribution,
    plot_forecast_with_interval,
    plot_model_comparison,
    plot_predictions,
    plot_residuals,
)
from src.weight_optimization import WeightOptimizer, combine_with_weights


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HorizonDataset:
    horizon: int
    feature_columns: list[str]
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


class ForecastingFramework:
    """End-to-end reusable forecasting framework for hybrid time-series modeling."""

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.project_root = Path(config_path).resolve().parent.parent
        self.paths = self.config.get("paths", {})
        self.output_dirs = {
            "plots": ensure_dir(self.project_root / self.config["visualization"]["output_dir"]),
            "artifacts": ensure_dir(self.project_root / self.paths.get("artifacts_dir", "outputs/artifacts")),
            "tables": ensure_dir(self.project_root / self.paths.get("tables_dir", "outputs/tables")),
            "predictions": ensure_dir(self.project_root / self.paths.get("predictions_dir", "outputs/predictions")),
            "models": ensure_dir(self.project_root / self.paths.get("models_dir", "outputs/models")),
        }

        set_global_seed(int(self.config.get("runtime", {}).get("seed", 42)))
        self.df_raw: pd.DataFrame | None = None
        self.horizon_cache: dict[int, HorizonDataset] = {}

    def load_data(self) -> pd.DataFrame:
        data_path = self.project_root / self.config["data"]["path"]
        self.df_raw = load_stock_data(data_path)
        return self.df_raw

    def build_features(self) -> pd.DataFrame:
        if self.df_raw is None:
            self.load_data()
        assert self.df_raw is not None

        feat_cfg = self.config["features"]
        pipe = FeaturePipeline(
            lags=feat_cfg.get("lags"),
            rolling_windows=feat_cfg.get("rolling_windows"),
            ema_windows=feat_cfg.get("ema_windows"),
            wma_windows=feat_cfg.get("wma_windows"),
            momentum_windows=feat_cfg.get("momentum_windows"),
            include_technical=feat_cfg.get("include_technical", True),
            include_date_features=feat_cfg.get("include_date_features", True),
            include_price_derived=feat_cfg.get("include_price_derived", True),
            dropna=False,
        )
        features = pipe.fit_transform(self.df_raw)
        return features

    def prepare_horizon_dataset(self, horizon: int) -> HorizonDataset:
        if horizon in self.horizon_cache:
            return self.horizon_cache[horizon]

        features = self.build_features()
        target_col = self.config["features"].get("target_col", "Close")
        labeled = build_horizon_target(features, target_col=target_col, horizon=horizon)
        labeled = labeled.dropna().copy()

        train_df, val_df, test_df = split_data(
            labeled,
            train_end=self.config["data"].get("train_end"),
            val_end=self.config["data"].get("val_end"),
        )

        feature_cols = [c for c in labeled.columns if c != "target"]
        X_train = train_df[feature_cols].values
        y_train = train_df["target"].values
        X_val = val_df[feature_cols].values
        y_val = val_df["target"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["target"].values

        data = HorizonDataset(
            horizon=horizon,
            feature_columns=feature_cols,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )
        self.horizon_cache[horizon] = data
        return data

    def train_baselines(self, horizon: int) -> dict[str, Any]:
        ds = self.prepare_horizon_dataset(horizon)
        cfg = self.config["models"]

        results, leaderboard = run_baseline_benchmark(
            ds.X_train,
            ds.y_train,
            ds.X_val,
            ds.y_val,
            ds.X_test,
            ds.y_test,
            random_state=cfg.get("random_state", 42),
            n_jobs=cfg.get("n_jobs", -1),
        )

        automl_cfg = cfg.get("automl", {})
        automl_results = run_automl_suite(
            ds.X_train,
            ds.y_train,
            ds.X_test,
            ds.y_test,
            pycaret_enabled=automl_cfg.get("pycaret", True),
            lazypredict_enabled=automl_cfg.get("lazypredict", True),
            flaml_enabled=automl_cfg.get("flaml", True),
            seed=cfg.get("random_state", 42),
            flaml_budget_s=automl_cfg.get("flaml_time_budget_s", 300),
        )

        leaderboard_path = self.output_dirs["tables"] / f"h{horizon}_baseline_leaderboard.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
        plot_model_comparison(leaderboard, metric_col="test_rmse", path=self.output_dirs["plots"] / f"h{horizon}_baseline_comparison.png")

        prediction_payload = {
            name: {
                "val_pred": result.val_pred.tolist(),
                "test_pred": result.test_pred.tolist(),
                "val_metrics": result.val_metrics,
                "test_metrics": result.test_metrics,
            }
            for name, result in results.items()
        }

        with open(self.output_dirs["predictions"] / f"h{horizon}_baseline_predictions.json", "w", encoding="utf-8") as f:
            json.dump(prediction_payload, f, indent=2)

        return {"results": results, "leaderboard": leaderboard, "automl": automl_results}

    def train_deep_models(self, horizon: int) -> dict[str, Any]:
        ds = self.prepare_horizon_dataset(horizon)
        deep_cfg = self.config["models"]["deep_learning"]

        config = DeepLearningConfig(
            sequence_length=deep_cfg.get("sequence_length", 30),
            hidden_size=deep_cfg.get("hidden_size", 64),
            dropout=deep_cfg.get("dropout", 0.2),
            learning_rate=deep_cfg.get("learning_rate", 1e-3),
            batch_size=deep_cfg.get("batch_size", 64),
            epochs=deep_cfg.get("epochs", 25),
            early_stopping_patience=deep_cfg.get("early_stopping_patience", 5),
            device="cpu",
        )

        architectures = deep_cfg.get("architectures", ["vanilla_lstm", "gru"])
        outputs = run_deep_learning_benchmark(
            X_train=ds.X_train,
            y_train=ds.y_train,
            X_val=ds.X_val,
            y_val=ds.y_val,
            X_test=ds.X_test,
            y_test=ds.y_test,
            architectures=architectures,
            config=config,
        )

        rows = []
        for name, payload in outputs.items():
            row = {"model": name}
            row.update({f"val_{k}": v for k, v in payload["val_metrics"].items() if k in {"rmse", "mae", "mape", "smape", "r2"}})
            row.update({f"test_{k}": v for k, v in payload["test_metrics"].items() if k in {"rmse", "mae", "mape", "smape", "r2"}})
            rows.append(row)

            plot_predictions(
                ds.y_test[config.sequence_length :],
                payload["test_pred"],
                title=f"{name} Horizon {horizon}",
                path=self.output_dirs["plots"] / f"h{horizon}_{name}_predictions.png",
            )

        deep_df = pd.DataFrame(rows).sort_values("test_rmse")
        deep_df.to_csv(self.output_dirs["tables"] / f"h{horizon}_deep_leaderboard.csv", index=False)

        return {"results": outputs, "leaderboard": deep_df, "sequence_length": config.sequence_length}

    @staticmethod
    def _align_tail(reference: np.ndarray, target_len: int) -> np.ndarray:
        if target_len > len(reference):
            raise ValueError("target_len exceeds reference length")
        return np.asarray(reference)[-target_len:]

    def train_hybrids(
        self,
        horizon: int,
        baseline_bundle: dict[str, Any] | None = None,
        deep_bundle: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        baseline_bundle = baseline_bundle or self.train_baselines(horizon)
        deep_bundle = deep_bundle or self.train_deep_models(horizon)
        ds = self.prepare_horizon_dataset(horizon)
        seq_len = deep_bundle["sequence_length"]

        deep_results: dict[str, dict[str, Any]] = deep_bundle["results"]
        if not deep_results:
            raise ValueError("Deep learning bundle has no models; cannot build hybrids")

        deep_anchor = next(iter(deep_results.values()))
        val_len = len(np.asarray(deep_anchor["val_pred"]).ravel())
        test_len = len(np.asarray(deep_anchor["test_pred"]).ravel())
        y_val_true = self._align_tail(ds.y_val, val_len)
        y_test_true = self._align_tail(ds.y_test, test_len)

        baseline_results: dict[str, BaselineResult] = baseline_bundle["results"]
        ml_val_preds = {name: self._align_tail(res.val_pred, val_len) for name, res in baseline_results.items()}
        ml_test_preds = {name: self._align_tail(res.test_pred, test_len) for name, res in baseline_results.items()}
        dl_val_preds = {name: np.asarray(payload["val_pred"]).ravel() for name, payload in deep_results.items()}
        dl_test_preds = {name: np.asarray(payload["test_pred"]).ravel() for name, payload in deep_results.items()}

        mapping = {
            "Linear Regression + LSTM": ("Linear Regression", "vanilla_lstm"),
            "Random Forest + LSTM": ("Random Forest", "vanilla_lstm"),
            "XGBoost + GRU": ("XGBoost", "gru"),
            "LightGBM + LSTM": ("LightGBM", "vanilla_lstm"),
        }

        val_hybrids: dict[str, np.ndarray] = {}
        test_hybrids: dict[str, np.ndarray] = {}
        for hybrid_name, (ml_key, dl_key) in mapping.items():
            if ml_key in ml_val_preds and dl_key in dl_val_preds:
                val_hybrids[hybrid_name] = weighted_ensemble(
                    {ml_key: ml_val_preds[ml_key], dl_key: dl_val_preds[dl_key]},
                    {ml_key: 0.5, dl_key: 0.5},
                )
                test_hybrids[hybrid_name] = weighted_ensemble(
                    {ml_key: ml_test_preds[ml_key], dl_key: dl_test_preds[dl_key]},
                    {ml_key: 0.5, dl_key: 0.5},
                )

        all_val_preds = {**ml_val_preds, **dl_val_preds}
        all_test_preds = {**ml_test_preds, **dl_test_preds}
        ordered_names = sorted(all_val_preds.keys())
        if ordered_names:
            val_matrix = np.column_stack([all_val_preds[name] for name in ordered_names])
            test_matrix = np.column_stack([all_test_preds[name] for name in ordered_names])

            # Learned validation weights are applied to holdout test predictions (no test leakage).
            optimizer = WeightOptimizer(method="grid", step=0.1, metric="rmse")
            learned_weights, _ = optimizer.optimize(all_val_preds, y_val_true)
            val_hybrids["Weighted Ensemble"] = combine_with_weights(all_val_preds, learned_weights)
            test_hybrids["Weighted Ensemble"] = combine_with_weights(all_test_preds, learned_weights)

            stack = AdvancedStackingEnsemble(
                base_models=[
                    ("ridge", Ridge(alpha=1.0)),
                    ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
                ],
                final_estimator=Ridge(alpha=0.5),
                cv=3,
            )
            stack.fit(val_matrix, y_val_true)
            val_hybrids["Stacking Ensemble"] = stack.predict(val_matrix)
            test_hybrids["Stacking Ensemble"] = stack.predict(test_matrix)

            meta_model = make_stacking_meta_learner()
            meta_model.fit(val_matrix, y_val_true)
            val_hybrids["Meta Learner Ensemble"] = meta_model.predict(val_matrix)
            test_hybrids["Meta Learner Ensemble"] = meta_model.predict(test_matrix)

        val_metrics = evaluate_regression(y_val_true, val_hybrids)
        test_metrics = evaluate_regression(y_test_true, test_hybrids)
        leaderboard = metrics_table(test_metrics, sort_by="rmse").reset_index().rename(columns={"index": "model"})
        val_leaderboard = metrics_table(val_metrics, sort_by="rmse").reset_index().rename(columns={"index": "model"})
        leaderboard.to_csv(self.output_dirs["tables"] / f"h{horizon}_hybrid_leaderboard.csv", index=False)
        val_leaderboard.to_csv(self.output_dirs["tables"] / f"h{horizon}_hybrid_validation_leaderboard.csv", index=False)

        for name, pred in test_hybrids.items():
            stem = name.replace(" ", "_").lower()
            plot_predictions(
                y_test_true,
                pred,
                title=f"{name} (H{horizon})",
                path=self.output_dirs["plots"] / f"h{horizon}_{stem}_pred.png",
            )
            plot_residuals(y_test_true, pred, path=self.output_dirs["plots"] / f"h{horizon}_{stem}_residuals.png")
            plot_error_distribution(
                y_test_true - pred,
                path=self.output_dirs["plots"] / f"h{horizon}_{stem}_errors.png",
            )

        return {
            "val_predictions": val_hybrids,
            "test_predictions": test_hybrids,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "val_leaderboard": val_leaderboard,
            "leaderboard": leaderboard,
            "y_val_true": y_val_true,
            "y_test_true": y_test_true,
            "sequence_length": seq_len,
        }

    def optimize_weights(
        self,
        horizon: int,
        predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
        method: str,
        evaluation_predictions: dict[str, np.ndarray] | None = None,
        evaluation_y_true: np.ndarray | None = None,
    ) -> dict[str, Any]:
        cfg = self.config["weight_optimization"]
        optimizer = WeightOptimizer(
            method=method,
            step=cfg.get("step", 0.05),
            metric=cfg.get("metric", "rmse"),
            bayesian_calls=cfg.get("bayesian_calls", 80),
            flaml_budget_s=cfg.get("flaml_budget_s", 120),
        )
        weights, diagnostics = optimizer.optimize(predictions, y_true)
        fit_pred = combine_with_weights(predictions, weights)
        fit_metrics = regression_metrics(y_true, fit_pred)

        out = {
            "weights": weights,
            "diagnostics": diagnostics,
            "fit_metrics": fit_metrics,
            "fit_predictions": fit_pred,
            # Backward-compatible aliases expected by app/notebooks.
            "metrics": fit_metrics,
            "predictions": fit_pred,
        }

        if evaluation_predictions is not None and evaluation_y_true is not None:
            eval_pred = combine_with_weights(evaluation_predictions, weights)
            eval_metrics = regression_metrics(evaluation_y_true, eval_pred)
            out["test_predictions"] = eval_pred
            out["test_metrics"] = eval_metrics
            # Prefer holdout metrics for downstream ranking/reporting.
            out["metrics"] = eval_metrics
            out["predictions"] = eval_pred

        plot_ensemble_contributions(weights, path=self.output_dirs["plots"] / f"h{horizon}_{method}_weights.png")
        return out

    def backtest(
        self,
        horizon: int,
        model: Any,
        strategy: str,
    ) -> dict[str, Any]:
        ds = self.prepare_horizon_dataset(horizon)
        X = np.vstack([ds.X_train, ds.X_val, ds.X_test])
        y = np.concatenate([ds.y_train, ds.y_val, ds.y_test])

        back_cfg = self.config["backtesting"]
        engine = BacktestEngine(
            model=model,
            strategy=strategy,
            n_splits=back_cfg.get("n_splits", 5),
            min_train_size=max(30, int(len(X) * back_cfg.get("min_train_size", 0.55))),
            test_size=max(1, int(len(X) * back_cfg.get("test_size", 0.1))),
            window_size=max(30, int(len(X) * back_cfg.get("rolling_window_size", 0.6))),
        )
        result = engine.run(X, y)

        fold_rows: list[dict[str, float | int]] = []
        for fold in result["fold_results"]:
            row: dict[str, float | int] = {
                "fold": int(fold["fold"]),
                "train_size": int(fold["train_size"]),
                "test_size": int(fold["test_size"]),
            }
            row.update({k: float(v) for k, v in fold["metrics"].items()})
            fold_rows.append(row)

        pd.DataFrame(fold_rows).to_csv(self.output_dirs["tables"] / f"h{horizon}_{strategy}_backtest.csv", index=False)
        return result

    def explain(
        self,
        horizon: int,
        model: Any,
    ) -> dict[str, Any]:
        ds = self.prepare_horizon_dataset(horizon)
        analyzer = FeatureImportanceAnalyzer(model=model, X_train=ds.X_train, y_train=ds.y_train)
        importance = analyzer.compute_all()

        perm_df = importance_to_frame(importance["permutation"])
        perm_df.to_csv(self.output_dirs["tables"] / f"h{horizon}_permutation_importance.csv", index=False)

        explain_rows = min(200, len(ds.X_test))
        try:
            explainer, shap_vals = shap_values(model, ds.X_train[:300], ds.X_test[:explain_rows])
            shap_summary_available = True
        except Exception as exc:
            logger.warning("SHAP computation failed: %s", exc)
            explainer = None
            shap_vals = np.array([])
            shap_summary_available = False

        return {
            "importance": importance,
            "summary": analyzer.summary(),
            "shap_available": shap_summary_available,
            "shap_values": shap_vals,
            "explainer": explainer,
        }

    def forecast(
        self,
        horizon: int,
        predictions: np.ndarray,
        y_true: np.ndarray,
    ) -> pd.DataFrame:
        error_std = float(np.std(y_true - predictions))
        lower = predictions - 1.96 * error_std
        upper = predictions + 1.96 * error_std

        idx = self.prepare_horizon_dataset(horizon).test_df.index[-len(predictions) :]
        forecast_df = pd.DataFrame(
            {
                "date": idx,
                "forecast": predictions,
                "lower_95": lower,
                "upper_95": upper,
            }
        )

        plot_forecast_with_interval(
            index=forecast_df["date"],
            forecast=forecast_df["forecast"].values,
            lower=forecast_df["lower_95"].values,
            upper=forecast_df["upper_95"].values,
            title=f"Horizon {horizon} Forecast",
            path=self.output_dirs["plots"] / f"h{horizon}_forecast_interval.png",
        )

        forecast_df.to_csv(self.output_dirs["predictions"] / f"h{horizon}_forecast.csv", index=False)
        return forecast_df

    def run_horizon(self, horizon: int) -> dict[str, Any]:
        baseline_bundle = self.train_baselines(horizon)
        deep_bundle = self.train_deep_models(horizon)
        hybrid_bundle = self.train_hybrids(horizon, baseline_bundle=baseline_bundle, deep_bundle=deep_bundle)

        y_val_true = hybrid_bundle["y_val_true"]
        y_test_true = hybrid_bundle["y_test_true"]
        prediction_bank_val = hybrid_bundle["val_predictions"]
        prediction_bank_test = hybrid_bundle["test_predictions"]

        opt_outputs: dict[str, Any] = {}
        for method in self.config["weight_optimization"].get("methods", ["grid", "bayesian", "flaml"]):
            try:
                opt_outputs[method] = self.optimize_weights(
                    horizon,
                    prediction_bank_val,
                    y_val_true,
                    method=method,
                    evaluation_predictions=prediction_bank_test,
                    evaluation_y_true=y_test_true,
                )
            except Exception as exc:
                logger.warning("Weight optimization %s failed on horizon %d: %s", method, horizon, exc)

        best_method = None
        best_val_rmse = float("inf")
        best_payload = None
        for method, payload in opt_outputs.items():
            rmse = payload["fit_metrics"]["rmse"]
            if rmse < best_val_rmse:
                best_val_rmse = rmse
                best_method = method
                best_payload = payload

        if best_payload is not None:
            test_pred = best_payload.get("test_predictions", best_payload["predictions"])
            self.forecast(horizon, test_pred, y_test_true)

        return {
            "baseline": baseline_bundle,
            "deep": deep_bundle,
            "hybrid": hybrid_bundle,
            "weight_optimization": opt_outputs,
            "best_weight_method": best_method,
        }

    def run_all_horizons(self) -> dict[int, dict[str, Any]]:
        horizons = self.config["features"].get("horizons", [1, 5, 10, 30])
        outputs: dict[int, dict[str, Any]] = {}
        for horizon in horizons:
            logger.info("Running complete pipeline for horizon=%d", horizon)
            outputs[horizon] = self.run_horizon(horizon)
        return outputs
