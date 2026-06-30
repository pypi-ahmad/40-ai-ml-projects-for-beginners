"""Serialization benchmark utilities for model artifact formats."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ml_package.model_loader import ModelLoader


def benchmark_serialization(
    model: Any,
    *,
    artifact_stem: str,
    output_dir: str | Path,
    sample: np.ndarray | None = None,
    torchscript_model: Any | None = None,
) -> list[dict[str, Any]]:
    """Benchmark save/load/inference latency across supported formats."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if sample is None:
        sample = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=float)

    rows: list[dict[str, Any]] = []
    for extension in (".pkl", ".joblib", ".onnx", ".pt"):
        artifact_path = output_path / f"{artifact_stem}{extension}"
        target_model = torchscript_model if extension == ".pt" else model
        row = {
            "format": extension,
            "artifact_path": str(artifact_path),
            "status": "ok",
            "save_time_ms": None,
            "load_time_ms": None,
            "predict_time_ms": None,
            "size_bytes": None,
            "error": None,
        }

        try:
            if target_model is None:
                raise RuntimeError(f"No compatible model provided for format {extension}")

            save_loader = ModelLoader(artifact_path)
            save_start = time.perf_counter()
            save_loader.save(
                target_model,
                create_manifest=True,
                metadata={"benchmark": "serialization", "format": extension},
            )
            row["save_time_ms"] = round((time.perf_counter() - save_start) * 1000, 3)
            row["size_bytes"] = artifact_path.stat().st_size

            load_loader = ModelLoader(
                artifact_path,
                verify_integrity=True,
                require_manifest=True,
            )
            load_start = time.perf_counter()
            loaded = load_loader.load()
            row["load_time_ms"] = round((time.perf_counter() - load_start) * 1000, 3)

            pred_start = time.perf_counter()
            if extension == ".onnx":
                loaded.run(None, {loaded.get_inputs()[0].name: sample.astype(np.float32)})
            elif extension == ".pt":
                try:
                    import torch
                except ImportError as exc:
                    raise RuntimeError(
                        "torch is required for TorchScript inference benchmark"
                    ) from exc
                with torch.no_grad():
                    loaded(torch.as_tensor(sample, dtype=torch.float32))
            else:
                loaded.predict(sample)
            row["predict_time_ms"] = round((time.perf_counter() - pred_start) * 1000, 3)
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)

        rows.append(row)

    return rows


def write_serialization_benchmark(rows: list[dict[str, Any]], output_file: str | Path) -> Path:
    """Write serialization benchmark rows as JSON."""
    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    return destination
