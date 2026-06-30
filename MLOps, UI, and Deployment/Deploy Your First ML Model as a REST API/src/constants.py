"""Shared constants used by training, API, and notebook workflows."""
from __future__ import annotations

from pathlib import Path

# California Housing canonical feature order.
FEATURE_NAMES: list[str] = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET_NAME: str = "MedHouseVal"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BENCHMARKS_DIR = ARTIFACTS_DIR / "benchmarks"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
PERFORMANCE_DIR = ARTIFACTS_DIR / "performance"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
