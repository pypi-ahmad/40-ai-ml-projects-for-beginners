"""Model and metadata serialization helpers."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib


def save_model(model: Any, path: Path) -> None:
    """Persist model artifact to disk via joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_metadata(metadata: dict[str, Any], path: Path) -> None:
    """Persist JSON metadata with UTF-8 encoding and deterministic ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload["saved_at_utc"] = datetime.now(UTC).isoformat()
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_metadata(path: Path) -> dict[str, Any]:
    """Load metadata JSON from disk."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def export_onnx_if_available(model: Any, output_path: Path, n_features: int) -> tuple[bool, str]:
    """Try ONNX export; return status + message.

    This function is optional so training can run even when ONNX tooling
    is not installed in the active environment.
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except Exception as exc:
        return False, f"ONNX export skipped: {exc}"

    try:
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            f.write(onnx_model.SerializeToString())
        return True, f"ONNX saved to {output_path}"
    except Exception as exc:
        return False, f"ONNX export failed: {exc}"
