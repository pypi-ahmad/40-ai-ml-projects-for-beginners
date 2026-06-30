"""Execute all tutorial notebooks end-to-end with nbconvert.

Usage:
    python scripts/run_all_notebooks.py
"""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
EXECUTED_DIR = PROJECT_ROOT / "outputs" / "executed_notebooks"

NOTEBOOKS = [
    "01_synthetic_intro.ipynb",
    "02_real_dataset_exploration.ipynb",
    "03_feature_selection_funnel.ipynb",
    "04_benchmarking.ipynb",
    "05_advanced_visualizations.ipynb",
    "06_pipeline_shap_inference.ipynb",
    "07_error_analysis.ipynb",
]


def run_notebook(nb_name: str) -> None:
    input_path = NOTEBOOK_DIR / nb_name
    output_name = nb_name.replace(".ipynb", ".executed.ipynb")

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(input_path),
        "--output",
        output_name,
        "--output-dir",
        str(EXECUTED_DIR),
        "--ExecutePreprocessor.timeout=7200",
    ]

    print(f"[RUN] {nb_name}")
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}:{existing_pythonpath}"
        if existing_pythonpath
        else str(PROJECT_ROOT)
    )
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)


def main() -> int:
    EXECUTED_DIR.mkdir(parents=True, exist_ok=True)
    failures = []

    for nb in NOTEBOOKS:
        try:
            run_notebook(nb)
        except subprocess.CalledProcessError as exc:
            failures.append((nb, exc.returncode))
            print(f"[FAIL] {nb} (exit={exc.returncode})")

    if failures:
        print("\nNotebook execution failed:")
        for nb, code in failures:
            print(f"  - {nb}: exit code {code}")
        return 1

    print("\nAll notebooks executed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
