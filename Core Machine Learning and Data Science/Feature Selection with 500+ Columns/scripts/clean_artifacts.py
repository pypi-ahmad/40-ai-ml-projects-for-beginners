"""Remove generated runtime artifacts for clean re-runs.

Usage:
    python scripts/clean_artifacts.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIR_TARGETS = [
    PROJECT_ROOT / "outputs" / "figures",
    PROJECT_ROOT / "outputs" / "metrics",
    PROJECT_ROOT / "outputs" / "models",
    PROJECT_ROOT / "outputs" / "executed_notebooks",
    PROJECT_ROOT / "data" / "real",
    PROJECT_ROOT / "data" / "synthetic",
    PROJECT_ROOT / "catboost_info",
    PROJECT_ROOT / "notebooks" / "catboost_info",
]

FILE_TARGETS = [
    PROJECT_ROOT / "outputs" / "pipeline_config.json",
    PROJECT_ROOT / "outputs" / "production_model.pkl",
    PROJECT_ROOT / "outputs" / "selected_features.csv",
    PROJECT_ROOT / "logs.log",
    PROJECT_ROOT / "notebooks" / "logs.log",
]


def clean_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file():
        path.unlink()
        print(f"Removed file: {path}")
        return

    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    print(f"Cleared: {path}")


def main() -> int:
    for target in DIR_TARGETS:
        target.mkdir(parents=True, exist_ok=True)
        clean_path(target)

    for target in FILE_TARGETS:
        clean_path(target)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
