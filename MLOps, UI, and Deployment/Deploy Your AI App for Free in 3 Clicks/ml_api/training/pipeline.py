"""End-to-end training pipeline for FastAPI serving artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from pathlib import Path

import numpy as np

from ml_api.core.config import Settings
from ml_api.training.automl import run_automl_benchmarks
from ml_api.training.benchmark import benchmark_models, scores_to_frame
from ml_api.training.data import load_dataset
from ml_api.training.models import build_model_catalog
from ml_api.training.persistence import ModelMetadata, save_artifacts


@dataclass(frozen=True)
class TrainingOutputs:
    metadata: ModelMetadata
    benchmark_csv: Path
    benchmark_json: Path
    automl_json: Path


def train_and_serialize(settings: Settings) -> TrainingOutputs:
    """Train candidate models, persist winner, and write benchmark artifacts."""
    _set_seed(settings.random_seed)

    dataset = load_dataset(settings.data_path, random_seed=settings.random_seed)
    models = build_model_catalog(settings.random_seed)
    benchmark = benchmark_models(models, dataset)

    metadata = save_artifacts(
        output_dir=settings.model_dir,
        pipeline=benchmark.best_pipeline,
        model_name=benchmark.best_model_name,
        dataset_hash=dataset.dataset_hash,
        training_rows=dataset.split.x_train.shape[0],
        scores=benchmark.scores,
        baseline_row=dataset.split.x_train.iloc[0],
        joblib_name=settings.joblib_artifact,
        pickle_name=settings.pickle_artifact,
        metadata_name=settings.metadata_artifact,
    )

    settings.benchmark_dir.mkdir(parents=True, exist_ok=True)

    benchmark_df = scores_to_frame(benchmark.scores)
    benchmark_csv = settings.benchmark_dir / "model_benchmark.csv"
    benchmark_json = settings.benchmark_dir / "model_benchmark.json"
    benchmark_df.to_csv(benchmark_csv, index=False)
    benchmark_json.write_text(benchmark_df.to_json(orient="records", indent=2))

    automl_results = run_automl_benchmarks(
        x_train=dataset.split.x_train,
        x_val=dataset.split.x_val,
        y_train=dataset.split.y_train,
        y_val=dataset.split.y_val,
        random_seed=settings.random_seed,
    )
    automl_json = settings.benchmark_dir / "automl_benchmark.json"
    automl_json.write_text(json.dumps([asdict(item) for item in automl_results], indent=2))

    return TrainingOutputs(
        metadata=metadata,
        benchmark_csv=benchmark_csv,
        benchmark_json=benchmark_json,
        automl_json=automl_json,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
