"""Environment verification utility for local project readiness.

Usage:
    .venv/bin/python scripts/verify_environment.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

REQUIRED_PACKAGES = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "sklearn",
    "joblib",
    "numpy",
    "pandas",
    "httpx",
    "shap",
]

OPTIONAL_TRAIN_PACKAGES = [
    "lazypredict",
    "flaml",
    "pycaret",
    "xgboost",
    "lightgbm",
    "catboost",
]


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except Exception:
        return False


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    artifacts_dir = project_root / "artifacts"

    report = {
        "python_version": sys.version.split()[0],
        "project_root": str(project_root),
        "required_packages": {pkg: _has_module(pkg) for pkg in REQUIRED_PACKAGES},
        "optional_train_packages": {pkg: _has_module(pkg) for pkg in OPTIONAL_TRAIN_PACKAGES},
        "paths": {
            "models_dir_exists": models_dir.exists(),
            "artifacts_dir_exists": artifacts_dir.exists(),
            "artifacts_writable": _is_writable(artifacts_dir),
            "tmp_writable": _is_writable(Path("/tmp") / "ml_api_project9"),
        },
        "artifacts": {
            "model_joblib": (models_dir / "model.joblib").exists(),
            "metadata_json": (models_dir / "metadata.json").exists(),
        },
        "env": {
            "api_key_set": bool(os.getenv("API_KEY")),
            "max_batch_size": os.getenv("MAX_BATCH_SIZE", "(default)"),
            "max_request_body_bytes": os.getenv("MAX_REQUEST_BODY_BYTES", "(default)"),
        },
    }

    required_ok = all(report["required_packages"].values())
    report["status"] = "pass" if required_ok else "fail"

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
