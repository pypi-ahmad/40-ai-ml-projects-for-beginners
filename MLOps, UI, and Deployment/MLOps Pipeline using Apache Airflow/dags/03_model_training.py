"""DAG 03: Model benchmarking, training, evaluation, registry, and tracking."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.data_loader import load_dataset
from modules.feature_selector import run_feature_selection
from modules.model_evaluator import evaluate_model
from modules.model_registry import register_model
from modules.model_trainer import run_training_pipeline
from modules.monitoring import pipeline_runtime_report, run_data_drift_monitoring, save_monitoring_snapshot
from modules.settings import load_config, resolve_path

CONFIG = load_config()
SCHEDULING = CONFIG.get("scheduling", {})


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}


@dag(
    dag_id="03_model_training",
    start_date=datetime(2026, 1, 1),
    schedule=SCHEDULING.get("training", "0 7 * * 1"),
    catchup=bool(SCHEDULING.get("catchup", False)),
    default_args=default_args,
    tags=["mlops", "training", "registry"],
)
def model_training_dag() -> None:
    """Training DAG."""
    config = load_config()

    wait_for_features = FileSensor(
        task_id="wait_for_feature_data",
        filepath=str(resolve_path(config, "data_features")),
        poke_interval=30,
        timeout=60 * 20,
        mode="poke",
    )

    @task(task_id="train_and_select_model")
    def train_and_select_model() -> dict:
        cfg = load_config()
        timing: dict[str, float] = {}

        t0 = time.perf_counter()
        feature_df = load_dataset("features", config=cfg)
        rankings = run_feature_selection(feature_df, config=cfg)
        selected = rankings.head(15)["feature"].tolist()
        timing["feature_selection"] = round(time.perf_counter() - t0, 4)

        t1 = time.perf_counter()
        training = run_training_pipeline(feature_df=feature_df, selected_features=selected, config=cfg)
        timing["benchmark_training"] = round(time.perf_counter() - t1, 4)

        t2 = time.perf_counter()
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
        timing["evaluation_registry"] = round(time.perf_counter() - t2, 4)

        baseline = feature_df.iloc[: int(len(feature_df) * 0.7)]
        current = feature_df.iloc[int(len(feature_df) * 0.7) :]
        numeric_cols = [c for c in feature_df.select_dtypes(include="number").columns if c != "target_next_day"]
        drift = run_data_drift_monitoring(baseline, current, numeric_cols=numeric_cols, config=cfg)

        runtime = pipeline_runtime_report(timing)
        monitoring_path = save_monitoring_snapshot(
            metrics=evaluation["metrics"],
            drift_report=drift,
            runtime_report=runtime,
            config=cfg,
        )

        return {
            "champion_model": training["model_name"],
            "champion_source": training["model_source"],
            "metrics": evaluation["metrics"],
            "registry": registry,
            "figures": evaluation["figures"],
            "monitoring_snapshot": monitoring_path,
            "leaderboard_path": training["leaderboard_path"],
        }

    complete = BashOperator(
        task_id="model_training_complete",
        bash_command="echo 'Model training pipeline complete'",
    )

    training_info = train_and_select_model()
    wait_for_features >> training_info >> complete


dag_obj = model_training_dag()
