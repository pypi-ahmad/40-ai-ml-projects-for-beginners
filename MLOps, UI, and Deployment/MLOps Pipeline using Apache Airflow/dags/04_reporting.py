"""DAG 04: Reporting pipeline for metrics, rankings, monitoring, and diagnostics."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor

import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.data_loader import load_json, save_json
from modules.reporting import build_report_html, save_report
from modules.settings import load_config, resolve_path

CONFIG = load_config()
SCHEDULING = CONFIG.get("scheduling", {})


def _latest_monitoring_snapshot(monitoring_dir: Path) -> Path | None:
    files = sorted(monitoring_dir.glob("monitoring_snapshot_*.json"))
    return files[-1] if files else None


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}


@dag(
    dag_id="04_reporting",
    start_date=datetime(2026, 1, 1),
    schedule=SCHEDULING.get("reporting", "30 7 * * 1"),
    catchup=bool(SCHEDULING.get("catchup", False)),
    default_args=default_args,
    tags=["mlops", "reporting"],
)
def reporting_dag() -> None:
    """Reporting DAG."""
    config = load_config()

    wait_for_metrics = FileSensor(
        task_id="wait_for_metrics_file",
        filepath=str(resolve_path(config, "best_metrics")),
        poke_interval=30,
        timeout=60 * 20,
        mode="poke",
    )

    @task(task_id="generate_pipeline_report")
    def generate_pipeline_report() -> dict:
        cfg = load_config()
        report_dir = resolve_path(cfg, "reports_dir")

        metrics_payload = load_json(resolve_path(cfg, "best_metrics"))
        metrics = metrics_payload.get("champion_metrics", {})

        leaderboard = pd.read_csv(resolve_path(cfg, "model_leaderboard"))
        ranking = pd.read_csv(resolve_path(cfg, "feature_rankings"))

        monitoring_path = _latest_monitoring_snapshot(resolve_path(cfg, "monitoring_dir"))
        monitoring = load_json(monitoring_path) if monitoring_path else None

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
                "forecast_horizon_days": cfg["project"]["forecast_horizon_days"],
                "champion": metrics_payload.get("champion"),
                "champion_source": metrics_payload.get("champion_source"),
            },
            metrics=metrics,
            model_leaderboard=leaderboard,
            candidate_scoreboard=leaderboard[["model", "test_rmse", "test_mae", "test_r2", "source"]],
            feature_ranking=ranking,
            monitoring_snapshot=monitoring,
            figure_paths=figures,
        )

        timestamped = save_report(
            report_html=html,
            output_dir=report_dir,
            filename=f"pipeline_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html",
        )

        latest = save_report(report_html=html, output_dir=PROJECT_ROOT / "outputs", filename="pipeline_report.html")

        manifest = {
            "timestamped_report": timestamped,
            "latest_report": latest,
            "generated_at": datetime.utcnow().isoformat(),
        }
        manifest_path = resolve_path(cfg, "reports_dir") / "report_manifest.json"
        save_json(manifest, manifest_path)
        return manifest

    complete = BashOperator(
        task_id="reporting_complete",
        bash_command="echo 'Reporting pipeline complete'",
    )

    report_info = generate_pipeline_report()
    wait_for_metrics >> report_info >> complete


dag_obj = reporting_dag()
