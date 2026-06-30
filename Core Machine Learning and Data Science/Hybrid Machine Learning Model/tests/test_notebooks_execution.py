from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


RUN_FLAG = os.getenv("RUN_NOTEBOOK_EXECUTION", "0") == "1"


@pytest.mark.skipif(not RUN_FLAG, reason="Set RUN_NOTEBOOK_EXECUTION=1 to run full notebook execution")
def test_all_notebooks_execute(project_root):
    notebooks = [
        "01_eda.ipynb",
        "02_feature_engineering.ipynb",
        "03_baseline_models.ipynb",
        "04_deep_learning.ipynb",
        "05_hybrid_models.ipynb",
        "06_weight_optimization.ipynb",
        "07_backtesting.ipynb",
        "08_shap_analysis.ipynb",
        "09_evaluation_report.ipynb",
    ]
    nb_dir = project_root / "notebooks"

    for nb in notebooks:
        cmd = [
            str(project_root / ".venv" / "bin" / "python"),
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(nb_dir / nb),
            "--output",
            str(Path("/tmp") / f"executed_{nb}"),
        ]
        subprocess.run(cmd, cwd=str(project_root), check=True)
