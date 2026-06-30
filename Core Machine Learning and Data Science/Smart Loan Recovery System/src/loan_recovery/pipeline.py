"""End-to-end pipeline orchestration for Smart Loan Recovery System."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    DATA_PATH,
    FIGURES_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    TABLES_DIR,
    TARGET_COLUMN,
)
from .dashboard import DashboardBuilder
from .data_loader import LoanDataLoader
from .eda import LoanEDA
from .evaluation import ModelEvaluator
from .explainability import ModelExplainer
from .features import FeatureEngineer
from .flaml_optimizer import FLAMLOptimizer
from .lazy_predict import LazyPredictBenchmark
from .models import ModelTrainer
from .pycaret_workflow import PyCaretWorkflow
from .segmentation import BorrowerSegmenter
from .strategy import RecoveryStrategyEngine
from .utils import cleanup_runtime_artifacts, ensure_output_dirs, save_dataframe, save_json, save_model, set_global_seed


@dataclass(slots=True)
class PipelineArtifacts:
    """Top-level pipeline outputs returned to notebooks and app."""

    leaderboard: pd.DataFrame
    lazypredict_table: pd.DataFrame
    pycaret_table: pd.DataFrame
    flaml_metrics: dict[str, Any]
    segmentation_metrics: pd.DataFrame
    segment_profiles: pd.DataFrame
    assigned_portfolio: pd.DataFrame
    scenario_table: pd.DataFrame
    evaluation_summary: dict[str, Any]
    best_model_name: str


class SmartLoanRecoveryPipeline:
    """Run complete production workflow with reproducible artifacts."""

    def __init__(self, data_path: str | Path = DATA_PATH, random_state: int = 42, strict_mode: bool = False) -> None:
        self.data_path = str(data_path)
        self.random_state = random_state
        self.strict_mode = strict_mode

    def run(self) -> PipelineArtifacts:
        """Execute all pipeline stages and persist intermediate outputs."""
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        set_global_seed(self.random_state)
        ensure_output_dirs()

        loader = LoanDataLoader(self.data_path)
        df = loader.load(add_target_derivatives=False)

        quality = loader.quality_report(df)
        save_json(quality.to_dict(), REPORTS_DIR / "data_quality_report.json")
        if self.strict_mode and quality.blocking_issues:
            raise ValueError(f"Data quality gate failed: {quality.blocking_issues}")

        feature_engineer = FeatureEngineer()
        enriched = feature_engineer.engineer(df, include_target_derivatives=False)
        save_dataframe(enriched, TABLES_DIR / "feature_enriched_data.csv")

        model_feature_cols = feature_engineer.model_feature_columns(
            list(enriched.columns),
            include_sensitive=False,
            early_warning=True,
            drop_cols=["Borrower_ID", TARGET_COLUMN],
        )
        feature_audit = pd.DataFrame(
            {
                "feature": list(enriched.columns),
                "used_for_modeling": [col in model_feature_cols for col in enriched.columns],
                "is_target": [col == TARGET_COLUMN for col in enriched.columns],
                "is_sensitive": [col in {"Marital_Status", "Residence_Type"} for col in enriched.columns],
                "is_collection_stage_proxy": [col in {"Collection_Attempts", "Recovery_Efficiency_Ratio", "Collection_Intensity_Score"} for col in enriched.columns],
                "is_target_derived": [col == "High_Risk_Flag" for col in enriched.columns],
            }
        )
        save_dataframe(feature_audit, TABLES_DIR / "feature_audit.csv")

        # EDA stage
        eda = LoanEDA(enriched)
        eda.generate_all_plots(FIGURES_DIR)
        save_dataframe(eda.summary_table(), TABLES_DIR / "eda_summary_table.csv")
        save_dataframe(eda.relationship_analysis(), TABLES_DIR / "eda_relationship_analysis.csv")

        split = feature_engineer.train_val_test_split(
            enriched,
            target_col=TARGET_COLUMN,
            drop_cols=["Borrower_ID"],
            include_sensitive=False,
            early_warning=True,
        )

        # Segmentation stage
        segmenter = BorrowerSegmenter(random_state=self.random_state)
        segmentation = segmenter.run(enriched)
        save_dataframe(segmentation.metrics_table, TABLES_DIR / "segmentation_metrics.csv")
        save_dataframe(segmentation.profile_table, TABLES_DIR / "segment_profiles.csv")
        save_json({str(k): v for k, v in segmentation.named_segments.items()}, REPORTS_DIR / "segment_names.json")

        # Add best-segment labels to enriched portfolio for downstream strategy and dashboards.
        best_labels = segmentation.labels_by_algorithm[segmentation.best_algorithm]
        enriched_with_segments = enriched.copy()
        enriched_with_segments["segment"] = best_labels
        enriched_with_segments["segment_name"] = enriched_with_segments["segment"].map(segmentation.named_segments)

        # Manual benchmark models
        trainer = ModelTrainer(random_state=self.random_state)
        model_artifacts = trainer.train_baselines(split.x_train, split.y_train, split.x_val, split.y_val)
        leaderboard = model_artifacts.leaderboard
        save_dataframe(leaderboard, TABLES_DIR / "manual_model_leaderboard.csv")
        imbalance_table = trainer.imbalance_experiments(split.x_train, split.y_train, split.x_val, split.y_val)
        save_dataframe(imbalance_table, TABLES_DIR / "imbalance_experiments.csv")

        best_model_name, best_model = trainer.best_model()
        save_model(best_model, MODELS_DIR / "best_manual_model.joblib")

        # Threshold optimization on validation data for written-off class decisions.
        evaluator = ModelEvaluator(FIGURES_DIR)
        y_prob_val = best_model.predict_proba(split.x_val)
        threshold_result = evaluator.optimize_high_risk_threshold(split.y_val, y_prob_val)
        save_json(
            {
                "optimized_threshold": threshold_result.threshold,
                "total_cost": threshold_result.total_cost,
                "false_positive_cost": threshold_result.false_positive_cost,
                "false_negative_cost": threshold_result.false_negative_cost,
                "high_risk_recall": threshold_result.recall_high_risk,
                "high_risk_precision": threshold_result.precision_high_risk,
            },
            REPORTS_DIR / "threshold_optimization.json",
        )

        # Final evaluation on untouched test data.
        y_prob = best_model.predict_proba(split.x_test)
        y_pred_default = best_model.predict(split.x_test)
        y_pred_cost_sensitive = evaluator.apply_high_risk_threshold(y_prob, threshold=threshold_result.threshold)

        eval_results = evaluator.evaluate(split.y_test, y_pred_default, y_prob, enriched.loc[split.x_test.index])
        cost_sensitive_business_metrics = evaluator.business_metrics(
            split.y_test,
            y_pred_cost_sensitive,
            y_prob,
            enriched.loc[split.x_test.index],
            false_positive_cost=250.0,
            false_negative_cost=2500.0,
        )

        evaluator.plot_confusion_matrix(split.y_test, y_pred_default, filename="confusion_matrix_default.png")
        evaluator.plot_confusion_matrix(split.y_test, y_pred_default, filename="confusion_matrix.png")
        evaluator.plot_confusion_matrix(split.y_test, y_pred_cost_sensitive, filename="confusion_matrix_cost_sensitive.png")
        evaluator.plot_roc_curves(split.y_test, y_prob)
        evaluator.plot_pr_curve_high_risk(split.y_test, y_prob)
        evaluator.plot_calibration_high_risk(split.y_test, y_prob)
        evaluator.plot_model_comparison(leaderboard)

        evaluation_summary = {
            "classification_metrics": eval_results.classification_metrics,
            "business_metrics_default_threshold": eval_results.business_metrics,
            "business_metrics_cost_sensitive_threshold": cost_sensitive_business_metrics,
            "threshold_metrics": {
                "optimized_threshold": threshold_result.threshold,
                "high_risk_recall": threshold_result.recall_high_risk,
                "high_risk_precision": threshold_result.precision_high_risk,
                "total_cost": threshold_result.total_cost,
            },
            "optimized_high_risk_threshold": threshold_result.threshold,
            "best_model": best_model_name,
        }
        save_json(evaluation_summary, REPORTS_DIR / "evaluation_summary.json")

        # LazyPredict baseline (numeric only).
        lazy_df = self._run_lazypredict(
            split.x_train,
            split.y_train,
            split.x_test,
            split.y_test,
        )
        save_dataframe(lazy_df, TABLES_DIR / "lazypredict_results.csv")

        # PyCaret workflow.
        pycaret_df = self._run_pycaret(enriched, feature_engineer)
        if not pycaret_df.empty:
            save_dataframe(pycaret_df, TABLES_DIR / "pycaret_comparison.csv")

        # FLAML optimization.
        flaml_metrics = self._run_flaml(split.x_train, split.y_train, split.x_test, split.y_test)
        save_json(flaml_metrics, REPORTS_DIR / "flaml_summary.json")

        # Strategy assignment and scenarios.
        strategy_engine = RecoveryStrategyEngine(best_model, feature_engineer)
        scored = strategy_engine.score_portfolio(enriched_with_segments)
        assigned = strategy_engine.assign_strategies(scored)
        segment_recos = strategy_engine.segment_recommendations(assigned)
        scenarios = strategy_engine.what_if_scenarios(enriched_with_segments)

        save_dataframe(assigned, TABLES_DIR / "portfolio_with_strategies.csv")
        save_dataframe(segment_recos, TABLES_DIR / "segment_strategy_recommendations.csv")
        save_dataframe(scenarios, TABLES_DIR / "scenario_analysis.csv")

        # SHAP explainability.
        explainer = ModelExplainer(best_model, FIGURES_DIR)
        shap_outputs = explainer.explain(split.x_test)
        save_json(
            {
                "shap_summary": str(shap_outputs.shap_summary_path) if shap_outputs.shap_summary_path else None,
                "shap_waterfall": str(shap_outputs.shap_waterfall_path) if shap_outputs.shap_waterfall_path else None,
                "shap_force": str(shap_outputs.shap_force_path) if shap_outputs.shap_force_path else None,
                "shap_feature_importance": str(shap_outputs.feature_importance_path)
                if shap_outputs.feature_importance_path
                else None,
            },
            REPORTS_DIR / "shap_outputs.json",
        )

        # Dashboard artifact generation.
        self._build_dashboard_assets(assigned, scenarios)
        cleanup_runtime_artifacts(PROJECT_ROOT)

        return PipelineArtifacts(
            leaderboard=leaderboard,
            lazypredict_table=lazy_df,
            pycaret_table=pycaret_df,
            flaml_metrics=flaml_metrics,
            segmentation_metrics=segmentation.metrics_table,
            segment_profiles=segmentation.profile_table,
            assigned_portfolio=assigned,
            scenario_table=scenarios,
            evaluation_summary=evaluation_summary,
            best_model_name=best_model_name,
        )

    def _run_lazypredict(
        self,
        x_train_input: pd.DataFrame,
        y_train: pd.Series,
        x_test_input: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        lazy = LazyPredictBenchmark(random_state=self.random_state)

        x_train = pd.get_dummies(x_train_input, drop_first=False)
        x_test = pd.get_dummies(x_test_input, drop_first=False)
        x_train, x_test = x_train.align(x_test, join="left", axis=1, fill_value=0)

        results = lazy.run(x_train, y_train, x_test, y_test)
        required = lazy.required_model_snapshot()
        if not required.empty:
            return required
        return results

    def _run_pycaret(self, enriched: pd.DataFrame, feature_engineer: FeatureEngineer) -> pd.DataFrame:
        pycaret_error_path = REPORTS_DIR / "pycaret_error.json"
        try:
            x_py, y_py = feature_engineer.split_features_target(
                enriched,
                target_col=TARGET_COLUMN,
                drop_cols=["Borrower_ID"],
                include_sensitive=False,
                early_warning=True,
            )
            pycaret_data = pd.concat([x_py, y_py], axis=1)
            pycaret = PyCaretWorkflow(random_state=self.random_state)
            artifacts = pycaret.run(pycaret_data, target_col=TARGET_COLUMN)
            if pycaret_error_path.exists():
                pycaret_error_path.unlink()
            return artifacts.comparison_table
        except Exception as exc:
            save_json({"pycaret_error": str(exc)}, pycaret_error_path)
            if self.strict_mode:
                raise
            return pd.DataFrame()

    def _run_flaml(
        self,
        x_train_input: pd.DataFrame,
        y_train: pd.Series,
        x_test_input: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, Any]:
        try:
            optimizer = FLAMLOptimizer(time_budget=20, random_state=self.random_state)

            x_train = pd.get_dummies(x_train_input, drop_first=False)
            x_test = pd.get_dummies(x_test_input, drop_first=False)
            x_train, x_test = x_train.align(x_test, join="left", axis=1, fill_value=0)

            artifacts = optimizer.run(x_train, y_train, x_test, y_test)
            return {
                "estimator_name": artifacts.estimator_name,
                "best_config": artifacts.best_config,
                "best_loss": artifacts.best_loss,
                "metrics": artifacts.metrics,
            }
        except Exception as exc:
            if self.strict_mode:
                raise
            return {"flaml_error": str(exc)}

    def _build_dashboard_assets(self, assigned: pd.DataFrame, scenarios: pd.DataFrame) -> None:
        builder = DashboardBuilder()
        risk_fig = builder.risk_distribution(assigned)
        recovery_fig = builder.recovery_probability_distribution(assigned)
        segment_fig = builder.segment_distribution(assigned)
        strategy_fig = builder.strategy_recommendations(assigned)
        scenario_fig = builder.scenario_comparison(scenarios)

        builder.save_html(risk_fig, REPORTS_DIR / "dashboard_risk_distribution.html")
        builder.save_html(recovery_fig, REPORTS_DIR / "dashboard_recovery_probability.html")
        builder.save_html(segment_fig, REPORTS_DIR / "dashboard_segment_distribution.html")
        builder.save_html(strategy_fig, REPORTS_DIR / "dashboard_strategy_recommendations.html")
        builder.save_html(scenario_fig, REPORTS_DIR / "dashboard_scenarios.html")
