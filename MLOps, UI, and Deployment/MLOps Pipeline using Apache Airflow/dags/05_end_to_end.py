"""DAG 05: Full end-to-end MLOps orchestration DAG.

This DAG intentionally runs the full lifecycle in one graph so it is testable with
`airflow dags test` and operationally visible in a single run:
validation -> features -> training -> reporting.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator

import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.data_generator import run_data_augmentation
from modules.data_loader import load_dataset, load_json
from modules.data_validator import detect_drift, run_full_validation, save_validation_report
from modules.feature_engineering import run_feature_pipeline
from modules.feature_selector import run_feature_selection
from modules.model_evaluator import evaluate_model
from modules.model_registry import register_model
from modules.model_trainer import run_training_pipeline
from modules.monitoring import pipeline_runtime_report, run_data_drift_monitoring, save_monitoring_snapshot
from modules.reporting import build_report_html, save_report
from modules.settings import load_config, resolve_path

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="05_end_to_end",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["mlops", "orchestration", "end_to_end"],
)
def end_to_end_dag() -> None:
    """Run the complete ML lifecycle in one DAG."""

    @task(task_id="run_validation_stage")
    def run_validation_stage() -> dict[str, str | int]:
        cfg = load_config()
        augmented = run_data_augmentation(config=cfg)
        quality = run_full_validation(augmented, config=cfg)
        if not bool(quality.get("checks_passed", False)):
            raise ValueError(f"Validation gate failed: {quality}")
        numeric_cols = augmented.select_dtypes(include="number").columns.tolist()
        split_idx = int(len(augmented) * 0.7)
        drift = detect_drift(
            baseline_df=augmented.iloc[:split_idx],
            current_df=augmented.iloc[split_idx:],
            numeric_cols=numeric_cols,
            psi_threshold=float(cfg["validation"]["psi_threshold"]),
            ks_pvalue_threshold=float(cfg["validation"]["ks_pvalue_threshold"]),
        )
        report_path = save_validation_report({"quality_report": quality, "drift_report": drift}, config=cfg)
        return {"rows": len(augmented), "report_path": report_path}

    @task(task_id="run_feature_stage")
    def run_feature_stage(_: dict[str, str | int]) -> dict[str, object]:
        cfg = load_config()
        validated_df = load_dataset("validated", config=cfg)
        featured = run_feature_pipeline(validated_df, config=cfg)
        ranking = run_feature_selection(featured, config=cfg)
        top_features = ranking.head(15)["feature"].tolist()
        return {"rows": len(featured), "top_features": top_features}

    @task(task_id="run_training_stage")
    def run_training_stage(feature_info: dict[str, object]) -> dict[str, object]:
        cfg = load_config()
        feature_df = load_dataset("features", config=cfg)
        selected = [str(x) for x in feature_info.get("top_features", [])]
        training = run_training_pipeline(feature_df=feature_df, selected_features=selected, config=cfg)

        evaluation = evaluate_model(
            model=training["model"],
            X_test=training["X_test"],
            y_test=training["y_test"],
            figures_dir=resolve_path(cfg, "figures_dir"),
            prefix="champion",
        )

        registry = register_model(
            model=training["model"],
            model_name="screentime_predictor",
            base_dir=resolve_path(cfg, "model_registry_dir"),
            metrics=evaluation["metrics"],
            hyperparameters={"model_name": training["model_name"], "source": training["model_source"]},
            feature_names=training["feature_columns"],
            stage="staging",
            mlflow_run_id=training.get("mlflow_run_id"),
        )

        baseline = feature_df.iloc[: int(len(feature_df) * 0.7)]
        current = feature_df.iloc[int(len(feature_df) * 0.7) :]
        numeric_cols = [c for c in feature_df.select_dtypes(include="number").columns if c != "target_next_day"]
        drift = run_data_drift_monitoring(baseline, current, numeric_cols=numeric_cols, config=cfg)
        runtime = pipeline_runtime_report(
            {
                "feature_selection": float(training["candidate_scoreboard"].shape[0]),
                "training_bundle": 1.0,
                "evaluation_registry": 1.0,
            }
        )
        monitoring_path = save_monitoring_snapshot(
            metrics=evaluation["metrics"],
            drift_report=drift,
            runtime_report=runtime,
            config=cfg,
        )
        return {
            "model_name": str(training["model_name"]),
            "model_source": str(training["model_source"]),
            "metrics": evaluation["metrics"],
            "registry_version": int(registry["version"]),
            "monitoring_snapshot": monitoring_path,
        }

    @task(task_id="run_reporting_stage")
    def run_reporting_stage(training_info: dict[str, object]) -> dict[str, str]:
        cfg = load_config()
        metrics_payload = load_json(resolve_path(cfg, "best_metrics"))
        metrics = metrics_payload.get("champion_metrics", {})
        leaderboard = pd.read_csv(resolve_path(cfg, "model_leaderboard"))
        ranking = pd.read_csv(resolve_path(cfg, "feature_rankings"))
        monitor_payload = load_json(str(training_info["monitoring_snapshot"]))
        figure_dir = resolve_path(cfg, "figures_dir")
        figures = {
            "residual_plot": str(figure_dir / "champion_residuals.png"),
            "prediction_plot": str(figure_dir / "champion_prediction_vs_actual.png"),
            "error_dist_plot": str(figure_dir / "champion_error_distribution.png"),
        }

        html = build_report_html(
            title="MLOps Pipeline Report",
            summary={
                "project": cfg["project"]["name"],
                "champion": training_info["model_name"],
                "source": training_info["model_source"],
                "registry_version": training_info["registry_version"],
            },
            metrics=metrics,
            model_leaderboard=leaderboard,
            candidate_scoreboard=leaderboard[["model", "test_rmse", "test_mae", "test_r2", "source"]],
            feature_ranking=ranking,
            monitoring_snapshot=monitor_payload,
            figure_paths=figures,
        )

        report_path = save_report(html, resolve_path(cfg, "reports_dir"), filename="pipeline_report_latest.html")
        latest_path = save_report(html, PROJECT_ROOT / "outputs", filename="pipeline_report.html")
        return {"report_path": report_path, "latest_path": latest_path}

    complete = BashOperator(
        task_id="end_to_end_complete",
        bash_command="echo 'End-to-end pipeline complete'",
    )

    validation = run_validation_stage()
    features = run_feature_stage(validation)
    training = run_training_stage(features)
    report = run_reporting_stage(training)
    validation >> features >> training >> report >> complete


dag_obj = end_to_end_dag()
