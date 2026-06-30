"""DAG 01: Data validation pipeline.

Flow:
raw load -> synthetic augmentation -> quality checks -> drift check -> validation report.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.data_generator import run_data_augmentation
from modules.data_loader import load_dataset, save_dataset
from modules.data_validator import detect_drift, run_full_validation, save_validation_report
from modules.settings import load_config

CONFIG = load_config()
SCHEDULING = CONFIG.get("scheduling", {})


def _generate_synthetic_callable() -> dict[str, str | int]:
    config = load_config()
    df = run_data_augmentation(config=config)
    validated_path = save_dataset(df, "validated", config=config)
    return {"rows": len(df), "validated_path": str(validated_path)}


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}


@dag(
    dag_id="01_data_validation",
    start_date=datetime(2026, 1, 1),
    schedule=SCHEDULING.get("validation", "0 6 * * *"),
    catchup=bool(SCHEDULING.get("catchup", False)),
    default_args=default_args,
    tags=["mlops", "validation", "data_quality"],
)
def data_validation_dag() -> None:
    """Data validation DAG."""

    generate_synthetic = PythonOperator(
        task_id="generate_synthetic_data",
        python_callable=_generate_synthetic_callable,
    )

    @task(task_id="run_quality_checks")
    def run_quality_checks() -> dict:
        config = load_config()
        validated_df = load_dataset("validated", config=config)
        report = run_full_validation(validated_df, config=config)
        if not bool(report.get("checks_passed", False)):
            raise ValueError(f"Data quality checks failed: {report}")
        return report

    @task(task_id="run_drift_detection")
    def run_drift_detection() -> dict:
        config = load_config()
        validated_df = load_dataset("validated", config=config)
        numeric_cols = validated_df.select_dtypes(include="number").columns.tolist()
        split_idx = int(len(validated_df) * 0.7)
        baseline_df = validated_df.iloc[:split_idx]
        current_df = validated_df.iloc[split_idx:]
        val_cfg = config["validation"]
        drift = detect_drift(
            baseline_df=baseline_df,
            current_df=current_df,
            numeric_cols=numeric_cols,
            psi_threshold=float(val_cfg["psi_threshold"]),
            ks_pvalue_threshold=float(val_cfg["ks_pvalue_threshold"]),
        )
        return drift

    @task(task_id="persist_validation_report")
    def persist_validation_report(report: dict, drift_report: dict) -> dict:
        config = load_config()
        payload = {
            "quality_report": report,
            "drift_report": drift_report,
            "generated_at": datetime.utcnow().isoformat(),
        }
        report_path = save_validation_report(payload, config=config)
        return {"report_path": report_path}

    complete = BashOperator(
        task_id="validation_complete",
        bash_command="echo 'Data validation pipeline complete'",
    )

    quality_report = run_quality_checks()
    drift_report = run_drift_detection()
    persisted = persist_validation_report(quality_report, drift_report)

    generate_synthetic >> quality_report
    generate_synthetic >> drift_report
    [quality_report, drift_report] >> persisted >> complete


dag_obj = data_validation_dag()
