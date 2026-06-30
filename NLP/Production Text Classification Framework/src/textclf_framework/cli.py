"""Typer CLI for training, evaluation, optimization and serving."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import mlflow
import pandas as pd
import typer
import uvicorn
from loguru import logger

from textclf_framework.benchmarking.runner import BenchmarkMatrixRunner
from textclf_framework.data.loader import DATASET_REGISTRY, DatasetLoader
from textclf_framework.data.preprocessing import PreprocessConfig
from textclf_framework.data.profiling import build_dataset_profile
from textclf_framework.data.versioning import build_manifest, save_manifest
from textclf_framework.logging_utils import configure_logging
from textclf_framework.models.registry import required_models, resolve_model_id
from textclf_framework.models.versioning import new_version, save_model_version
from textclf_framework.reporting.report_builder import ReportBuilder
from textclf_framework.serving.api import APIConfig, create_app
from textclf_framework.settings import AppConfig, load_config
from textclf_framework.training.engine import TextClassificationTrainer, TrainingRunConfig

app = typer.Typer(help="Production text classification framework CLI")


def _bootstrap(config_path: str | None) -> AppConfig:
    config = load_config(config_path)
    configure_logging()
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    return config


@app.command("profile-data")
def profile_data(
    dataset: str = typer.Option("setfit_20_newsgroups", help="Dataset key."),
    config_path: str | None = typer.Option(None, help="YAML config file."),
) -> None:
    """Profile dataset and write profile + version manifest."""
    config = _bootstrap(config_path)
    loader = DatasetLoader(seed=config.seed, val_size=config.datasets.validation_size)
    preprocess_cfg = PreprocessConfig()
    dataset_dict = loader.load(dataset, preprocess_config=preprocess_cfg)

    profile = build_dataset_profile(dataset, dataset_dict)
    profile_path = config.paths.report_dir / f"{dataset}_profile.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(asdict(profile), default=str, indent=2), encoding="utf-8")

    manifest = build_manifest(
        dataset_name=dataset,
        source=DATASET_REGISTRY[dataset].hf_id,
        dataset_dict=dataset_dict,
        split_seed=config.seed,
        preprocessing_config=asdict(preprocess_cfg),
        label_names=loader.label_names(dataset_dict),
    )
    save_manifest(manifest, config.paths.artifact_dir / f"{dataset}_manifest.json")
    logger.info(f"Profile saved to {profile_path}")


@app.command("train-run")
def train_run(
    dataset: str = typer.Option("setfit_20_newsgroups"),
    model: str = typer.Option("distilbert"),
    strategy: str = typer.Option("full"),
    config_path: str | None = typer.Option(None),
) -> None:
    """Train one model on one dataset and log to MLflow."""
    config = _bootstrap(config_path)
    loader = DatasetLoader(seed=config.seed, val_size=config.datasets.validation_size)
    dataset_dict = loader.load(dataset, preprocess_config=PreprocessConfig())
    labels = loader.label_names(dataset_dict)

    run_config = TrainingRunConfig(
        output_dir=config.paths.artifact_dir / "models" / dataset / model,
        model_name=model,
        strategy=strategy,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        epochs=config.training.epochs,
        train_batch_size=config.training.train_batch_size,
        eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        warmup_ratio=config.training.warmup_ratio,
        max_length=config.training.max_length,
        early_stopping_patience=config.training.early_stopping_patience,
        gradient_checkpointing=config.training.gradient_checkpointing,
        fp16=config.training.fp16,
    )

    with mlflow.start_run(run_name=f"{dataset}-{model}-{strategy}"):
        mlflow.log_params(
            {
                "dataset": dataset,
                "model": model,
                "resolved_model": resolve_model_id(model),
                "strategy": strategy,
                "max_length": config.training.max_length,
            }
        )
        result = TextClassificationTrainer(run_config).train(dataset_dict, num_labels=len(labels), label_names=labels)
        mlflow.log_metrics(result.metrics)
        mlflow.log_param("trainable_params", result.trainable_params)
        mlflow.log_param("total_params", result.total_params)

        metrics_path = config.paths.report_dir / f"metrics_{dataset}_{model}.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(metrics_path))
        version = new_version(
            name=model,
            dataset=dataset,
            strategy=result.strategy_used,
            metrics=result.metrics,
            artifact_path=str(result.model_dir),
        )
        save_model_version(version, config.paths.artifact_dir / "model_registry.json")
        logger.info(f"Training complete model_dir={result.model_dir}")


@app.command("benchmark-run")
def benchmark_run(config_path: str | None = typer.Option(None)) -> None:
    """Run benchmark matrix across required models and selected datasets."""
    config = _bootstrap(config_path)
    datasets = [config.datasets.primary, *config.datasets.additional]
    models = required_models()

    runner = BenchmarkMatrixRunner()

    def evaluator(dataset_name: str, model_name: str) -> dict[str, float]:
        start = time.perf_counter()
        try:
            loader = DatasetLoader(seed=config.seed, val_size=config.datasets.validation_size)
            ds = loader.load(dataset_name, preprocess_config=PreprocessConfig())
            labels = loader.label_names(ds)
            train_cfg = TrainingRunConfig(
                output_dir=config.paths.artifact_dir / "benchmark" / dataset_name / model_name,
                model_name=model_name,
                strategy=config.models.strategies.get(model_name, "full"),
                learning_rate=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
                epochs=config.training.epochs,
                train_batch_size=config.training.train_batch_size,
                eval_batch_size=config.training.eval_batch_size,
                gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                warmup_ratio=config.training.warmup_ratio,
                max_length=config.training.max_length,
                early_stopping_patience=config.training.early_stopping_patience,
                gradient_checkpointing=config.training.gradient_checkpointing,
                fp16=config.training.fp16,
            )
            result = TextClassificationTrainer(train_cfg).train(ds, num_labels=len(labels), label_names=labels)
            elapsed = time.perf_counter() - start

            return {
                "accuracy": result.metrics.get("eval_accuracy", 0.0),
                "macro_f1": result.metrics.get("eval_macro_f1", 0.0),
                "train_seconds": elapsed,
                "inference_latency_ms": 0.0,
                "gpu_memory_mb": 0.0,
                "cpu_latency_ms": 0.0,
                "model_size_mb": 0.0,
            }
        except Exception as exc:
            if model_name == "modernbert":
                logger.warning(f"Skipping modernbert for dataset={dataset_name}. reason={exc}")
                return {
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                    "train_seconds": 0.0,
                    "inference_latency_ms": 0.0,
                    "gpu_memory_mb": 0.0,
                    "cpu_latency_ms": 0.0,
                    "model_size_mb": 0.0,
                }
            raise

    result_df = runner.run(
        datasets=datasets,
        models=models,
        evaluator=evaluator,
        strategy_resolver=lambda m: config.models.strategies.get(m, "full"),
    )

    output_csv = config.paths.report_dir / "benchmark_matrix.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    logger.info(f"Benchmark matrix saved to {output_csv}")


@app.command("build-report")
def build_report(config_path: str | None = typer.Option(None)) -> None:
    """Generate markdown/HTML benchmark reports."""
    config = _bootstrap(config_path)
    builder = ReportBuilder(config.paths.report_dir)
    csv_path = config.paths.report_dir / "benchmark_matrix.csv"
    md = builder.build_benchmark_summary(csv_path)
    html = builder.build_benchmark_plot(csv_path)
    logger.info(f"Generated reports: {md}, {html}")


@app.command("serve-api")
def serve_api(
    model_path: str = typer.Option("artifacts/models/champion"),
    labels: str = typer.Option("class_0,class_1"),
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
) -> None:
    """Launch FastAPI service."""
    label_names = [x.strip() for x in labels.split(",") if x.strip()]
    api = create_app(APIConfig(model_path=model_path, label_names=label_names))
    uvicorn.run(api, host=host, port=port)


@app.command("run-ui")
def run_ui() -> None:
    """Launch Streamlit dashboard."""
    subprocess.run(["streamlit", "run", "apps/streamlit_app.py"], check=True)


@app.command("quick-all")
def quick_all(config_path: str | None = typer.Option("configs/quick.yaml")) -> None:
    """Quick smoke execution for data profiling + one train run."""
    profile_data(dataset="setfit_20_newsgroups", config_path=config_path)
    train_run(
        dataset="setfit_20_newsgroups",
        model="distilbert",
        strategy="full",
        config_path=config_path,
    )


if __name__ == "__main__":
    app()
