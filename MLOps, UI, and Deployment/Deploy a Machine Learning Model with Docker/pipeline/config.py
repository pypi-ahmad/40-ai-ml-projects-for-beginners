"""Central configuration for training, artifact paths, and reproducibility."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RANDOM_SEED = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "MedHouseVal"
FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout used across the project."""

    root: Path
    models: Path
    outputs: Path
    benchmarks: Path
    figures: Path
    logs: Path
    notebooks_outputs: Path


def get_project_paths() -> ProjectPaths:
    """Return canonical project paths and create required directories."""
    root = Path(__file__).resolve().parent.parent
    paths = ProjectPaths(
        root=root,
        models=root / "models",
        outputs=root / "outputs",
        benchmarks=root / "outputs" / "benchmarks",
        figures=root / "outputs" / "figures",
        logs=root / "outputs" / "logs",
        notebooks_outputs=root / "notebooks" / "outputs",
    )
    for path in (
        paths.models,
        paths.outputs,
        paths.benchmarks,
        paths.figures,
        paths.logs,
        paths.notebooks_outputs,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return paths

