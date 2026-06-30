#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/.tmp/matplotlib}"
export AIRFLOW_HOME="${AIRFLOW_HOME:-$PROJECT_ROOT/airflow_config}"
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_ROOT/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
mkdir -p "$MPLCONFIGDIR"

# Run each stage directly via python for deterministic local execution.
.venv/bin/python - <<'PY'
import logging
import time
import tracemalloc
from pathlib import Path

from modules.data_generator import run_data_augmentation
from modules.data_loader import load_dataset
from modules.data_validator import detect_drift, run_full_validation, save_validation_report
from modules.feature_engineering import run_feature_pipeline
from modules.feature_selector import run_feature_selection
from modules.model_evaluator import evaluate_model
from modules.model_registry import register_model
from modules.model_trainer import run_training_pipeline
from modules.monitoring import pipeline_runtime_report, run_data_drift_monitoring, save_monitoring_snapshot
from modules.reporting import build_report_html, save_report
from modules.settings import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("local_pipeline")

cfg = load_config()
timings: dict[str, float] = {}
tracemalloc.start()

start = time.perf_counter()
aug = run_data_augmentation(cfg)
timings["data_augmentation"] = round(time.perf_counter() - start, 4)

start = time.perf_counter()
val = run_full_validation(aug, cfg)
if not bool(val.get("checks_passed", False)):
    raise RuntimeError(f"Validation checks failed: {val}")
val_drift = detect_drift(
    baseline_df=aug.iloc[: int(len(aug) * 0.7)],
    current_df=aug.iloc[int(len(aug) * 0.7) :],
    numeric_cols=aug.select_dtypes(include='number').columns.tolist(),
    psi_threshold=float(cfg['validation']['psi_threshold']),
    ks_pvalue_threshold=float(cfg['validation']['ks_pvalue_threshold']),
)
save_validation_report({'quality_report': val, 'drift_report': val_drift}, cfg)
timings["data_validation"] = round(time.perf_counter() - start, 4)

start = time.perf_counter()
featured = run_feature_pipeline(aug, cfg)
ranking = run_feature_selection(featured, cfg)
selected = ranking.head(15)['feature'].tolist()
timings["feature_pipeline"] = round(time.perf_counter() - start, 4)

start = time.perf_counter()
training = run_training_pipeline(featured, selected_features=selected, config=cfg)
timings["model_training"] = round(time.perf_counter() - start, 4)

start = time.perf_counter()
eval_result = evaluate_model(
    model=training['model'],
    X_test=training['X_test'],
    y_test=training['y_test'],
    figures_dir=resolve_path(cfg, 'figures_dir'),
    prefix='champion',
)
timings["model_evaluation"] = round(time.perf_counter() - start, 4)

start = time.perf_counter()
registry = register_model(
    model=training['model'],
    model_name='screentime_predictor',
    base_dir=resolve_path(cfg, 'model_registry_dir'),
    metrics=eval_result['metrics'],
    hyperparameters={'model_name': training['model_name'], 'source': training['model_source']},
    feature_names=training['feature_columns'],
    stage='staging',
    mlflow_run_id=training.get('mlflow_run_id'),
)
timings["model_registry"] = round(time.perf_counter() - start, 4)

start = time.perf_counter()
drift = run_data_drift_monitoring(
    baseline_df=featured.iloc[: int(len(featured) * 0.7)],
    current_df=featured.iloc[int(len(featured) * 0.7) :],
    numeric_cols=[c for c in featured.select_dtypes(include='number').columns if c != 'target_next_day'],
    config=cfg,
)
timings["monitoring"] = round(time.perf_counter() - start, 4)
runtime = pipeline_runtime_report(timings)
runtime["peak_memory_mb"] = round(tracemalloc.get_traced_memory()[1] / (1024 * 1024), 4)
monitoring_path = save_monitoring_snapshot(eval_result['metrics'], drift, runtime, cfg)

start = time.perf_counter()
report_html = build_report_html(
    title='MLOps Pipeline Report',
    summary={
        'project': cfg['project']['name'],
        'champion': training['model_name'],
        'source': training['model_source'],
        'registry_version': registry['version'],
    },
    metrics=eval_result['metrics'],
    model_leaderboard=training['leaderboard'],
    candidate_scoreboard=training['candidate_scoreboard'],
    feature_ranking=ranking,
    monitoring_snapshot={'alerts': [], 'drift_report': drift, 'runtime_report': runtime},
    figure_paths=eval_result['figures'],
)

save_report(report_html, resolve_path(cfg, 'reports_dir'), filename='pipeline_report_latest.html')
save_report(report_html, Path('outputs'), filename='pipeline_report.html')
timings["reporting"] = round(time.perf_counter() - start, 4)
runtime["task_runtime_seconds"] = timings
runtime["total_runtime_seconds"] = round(sum(timings.values()), 4)
tracemalloc.stop()

logger.info("Pipeline complete")
logger.info("Monitoring snapshot: %s", monitoring_path)
PY

echo "Local pipeline run finished"
