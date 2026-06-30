"""DAG 02: Feature engineering and feature selection pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.data_loader import load_dataset, save_dataset
from modules.feature_engineering import run_feature_pipeline, temporal_train_test_split
from modules.feature_selector import run_feature_selection
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
    dag_id="02_feature_engineering",
    start_date=datetime(2026, 1, 1),
    schedule=SCHEDULING.get("feature_engineering", "15 6 * * *"),
    catchup=bool(SCHEDULING.get("catchup", False)),
    default_args=default_args,
    tags=["mlops", "features"],
)
def feature_engineering_dag() -> None:
    """Feature engineering DAG."""
    config = load_config()

    wait_for_validated = FileSensor(
        task_id="wait_for_validated_data",
        filepath=str(resolve_path(config, "data_validated")),
        poke_interval=30,
        timeout=60 * 20,
        mode="poke",
    )

    @task(task_id="build_features")
    def build_features() -> dict:
        cfg = load_config()
        validated_df = load_dataset("validated", config=cfg)
        featured_df = run_feature_pipeline(validated_df, config=cfg)
        return {"rows": len(featured_df), "columns": len(featured_df.columns)}

    @task(task_id="run_feature_selection")
    def run_selection() -> dict:
        cfg = load_config()
        featured_df = load_dataset("features", config=cfg)
        ranking_df = run_feature_selection(featured_df, config=cfg)
        return {
            "top_features": ranking_df.head(10)["feature"].tolist(),
            "ranking_rows": len(ranking_df),
        }

    @task(task_id="split_train_test")
    def split_train_test() -> dict:
        cfg = load_config()
        featured_df = load_dataset("features", config=cfg)
        train_df, test_df = temporal_train_test_split(
            featured_df,
            date_col=str(cfg["project"]["date_col"]),
            test_size=float(cfg["training"]["test_size"]),
        )
        train_path = save_dataset(train_df, "train", config=cfg)
        test_path = save_dataset(test_df, "test", config=cfg)
        return {
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_path": str(train_path),
            "test_path": str(test_path),
        }

    complete = BashOperator(
        task_id="feature_engineering_complete",
        bash_command="echo 'Feature engineering pipeline complete'",
    )

    feature_info = build_features()
    selection = run_selection()
    split_info = split_train_test()

    wait_for_validated >> feature_info >> selection >> split_info >> complete


dag_obj = feature_engineering_dag()
