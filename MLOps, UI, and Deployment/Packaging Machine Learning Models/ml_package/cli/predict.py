"""CLI interface for prediction, batch scoring, and API serving."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ml_package import ModelLoader, PredictionEngine, VersionRegistry, setup_logging
from ml_package.settings import PackageSettings
from ml_package.validation import IrisFeatures

logger = setup_logging("iris_cli")

FEATURE_KEYS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def load_model(
    model_path: str | Path,
    settings: PackageSettings,
) -> PredictionEngine:
    loader = ModelLoader(
        model_path,
        verify_integrity=settings.verify_artifacts,
        require_manifest=True,
        trusted_digests=settings.resolved_trusted_digests(model_path),
        allow_unsafe_deserialization=settings.allow_unsafe_deserialization,
    )
    model = loader.load()
    engine = PredictionEngine(model, model_name="iris_classifier")
    if settings.registry_path.exists():
        registry = VersionRegistry(str(settings.registry_path))
        active = registry.get_active()
        if active is not None:
            engine.model_version = active.version_id
    return engine


def _ensure_sample_dict(sample: dict[str, Any], index: int) -> dict[str, float]:
    missing = [key for key in FEATURE_KEYS if key not in sample]
    if missing:
        raise ValueError(f"Sample {index} missing fields: {missing}")

    validated = IrisFeatures(
        sepal_length=float(sample["sepal_length"]),
        sepal_width=float(sample["sepal_width"]),
        petal_length=float(sample["petal_length"]),
        petal_width=float(sample["petal_width"]),
    )
    return {
        "sepal_length": validated.sepal_length,
        "sepal_width": validated.sepal_width,
        "petal_length": validated.petal_length,
        "petal_width": validated.petal_width,
    }


def load_samples_from_json(path: str | Path) -> list[dict[str, float]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Batch file not found: {file_path}")

    with file_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        samples = payload
    elif isinstance(payload, dict):
        if "samples" in payload and isinstance(payload["samples"], list):
            samples = payload["samples"]
        else:
            samples = [payload]
    else:
        raise ValueError("JSON batch payload must be object or list")

    return [_ensure_sample_dict(sample, idx) for idx, sample in enumerate(samples)]


def load_samples_from_csv(path: str | Path) -> list[dict[str, float]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Batch file not found: {file_path}")

    with file_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV batch file is empty")

    return [_ensure_sample_dict(row, idx) for idx, row in enumerate(rows)]


def load_batch_samples(path: str | Path) -> list[dict[str, float]]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        return load_samples_from_json(file_path)
    if suffix == ".csv":
        return load_samples_from_csv(file_path)
    raise ValueError("Batch file must use .json or .csv extension")


def emit_payload(payload: dict[str, Any], output_path: str | None = None) -> None:
    if output_path is None:
        print(json.dumps(payload, indent=2))
        return

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved output to {destination}")


def cmd_predict(args: argparse.Namespace, settings: PackageSettings) -> int:
    engine = load_model(args.model_path, settings)
    features = np.array(
        [[args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]],
        dtype=float,
    )
    result = engine.predict(features)
    emit_payload(result, args.output)
    return 0


def cmd_batch(args: argparse.Namespace, settings: PackageSettings) -> int:
    samples = load_batch_samples(args.file)
    engine = load_model(args.model_path, settings)
    features = np.array([[sample[key] for key in FEATURE_KEYS] for sample in samples], dtype=float)
    results = engine.predict_batch(features)
    payload = {"predictions": results, "count": len(results)}
    emit_payload(payload, args.output)
    return 0


def cmd_serve(args: argparse.Namespace, _: PackageSettings) -> int:
    from api.main import app
    import uvicorn

    logger.info("Starting API server from CLI")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_info(args: argparse.Namespace, settings: PackageSettings) -> int:
    engine = load_model(args.model_path, settings)
    info = engine.get_model_info()
    info["model_path"] = str(args.model_path)
    emit_payload(info, args.output)
    return 0


def build_parser(default_model_path: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Iris Classifier CLI — predict, batch, serve, info"
    )
    parser.add_argument(
        "--model-path",
        default=default_model_path,
        help="Path to serialized model",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON file path",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    predict_parser = subparsers.add_parser("predict", help="Single prediction")
    predict_parser.add_argument("--sepal-length", type=float, required=True)
    predict_parser.add_argument("--sepal-width", type=float, required=True)
    predict_parser.add_argument("--petal-length", type=float, required=True)
    predict_parser.add_argument("--petal-width", type=float, required=True)
    predict_parser.set_defaults(func=cmd_predict)

    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch prediction from JSON/CSV file",
    )
    batch_parser.add_argument("file", help="Batch input file (.json or .csv)")
    batch_parser.set_defaults(func=cmd_batch)

    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.set_defaults(func=cmd_serve)

    info_parser = subparsers.add_parser("info", help="Show model metadata")
    info_parser.set_defaults(func=cmd_info)

    return parser


def main() -> int:
    settings = PackageSettings.from_env()
    parser = build_parser(str(settings.model_path))
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        return int(args.func(args, settings))
    except Exception as exc:
        logger.error(f"CLI command failed: {exc}")
        print(json.dumps({"error": "command_failed", "detail": str(exc)}, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())
