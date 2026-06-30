"""Model persistence and metadata artifact helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from ml_api.training.benchmark import ModelScore
from ml_api.training.feature_spec import ALL_FEATURES, TARGET_COLUMN


@dataclass(frozen=True)
class ModelMetadata:
    model_name: str
    model_type: str
    model_version: str
    feature_columns: list[str]
    target_column: str
    training_rows: int
    validation_rmse: float
    test_rmse: float
    dataset_hash: str
    created_at_utc: str
    benchmark_scores: list[dict[str, float | str]]
    baseline_row: dict[str, float | str | int]


def save_artifacts(
    output_dir: Path,
    pipeline: Pipeline,
    model_name: str,
    dataset_hash: str,
    training_rows: int,
    scores: list[ModelScore],
    baseline_row: pd.Series,
    joblib_name: str,
    pickle_name: str,
    metadata_name: str,
) -> ModelMetadata:
    """Persist model artifacts and metadata JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib_path = output_dir / joblib_name
    pickle_path = output_dir / pickle_name
    metadata_path = output_dir / metadata_name

    joblib.dump(pipeline, joblib_path)
    with pickle_path.open("wb") as f:
        pickle.dump(pipeline, f)

    best = scores[0]
    metadata = ModelMetadata(
        model_name=model_name,
        model_type=type(pipeline.named_steps["model"]).__name__,
        model_version="v1",
        feature_columns=ALL_FEATURES,
        target_column=TARGET_COLUMN,
        training_rows=training_rows,
        validation_rmse=best.val_rmse,
        test_rmse=best.test_rmse,
        dataset_hash=dataset_hash,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        benchmark_scores=[asdict(score) for score in scores],
        baseline_row={k: _to_json_scalar(v) for k, v in baseline_row.to_dict().items()},
    )

    metadata_path.write_text(json.dumps(asdict(metadata), indent=2))
    return metadata


def load_model(joblib_path: Path):
    """Load model artifact from joblib path."""
    return joblib.load(joblib_path)


def load_metadata(path: Path) -> dict[str, object]:
    """Load model metadata JSON."""
    return json.loads(path.read_text())


def _to_json_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value
