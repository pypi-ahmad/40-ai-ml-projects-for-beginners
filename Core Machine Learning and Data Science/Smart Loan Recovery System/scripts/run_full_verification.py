"""Run complete local verification for Smart Loan Recovery System."""

from __future__ import annotations

import shutil
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.loan_recovery import DATA_PATH, OUTPUTS_DIR, SmartLoanRecoveryPipeline
NOTEBOOKS = [
    "01_loan_recovery_foundations_eda.ipynb",
    "02_data_quality_feature_engineering.ipynb",
    "03_borrower_segmentation.ipynb",
    "04_risk_prediction_baselines_lazypredict.ipynb",
    "05_pycaret_vs_manual_workflow.ipynb",
    "06_flaml_optimization_evaluation.ipynb",
    "07_explainable_ai_strategy_engine.ipynb",
    "08_dashboard_and_streamlit_deployment.ipynb",
]
LEGACY_ROOT_ARTIFACTS = [
    "clusters_pca.png",
    "confusion_matrix.png",
    "elbow_plot.png",
    "model_comparison.png",
    "roc_curves.png",
    "logs.log",
]


def run_cmd(cmd: list[str]) -> None:
    """Run a command and raise on failure."""
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def clean_generated_artifacts() -> None:
    """Delete generated outputs to test clean reproducibility."""
    if OUTPUTS_DIR.exists():
        shutil.rmtree(OUTPUTS_DIR)
    for name in LEGACY_ROOT_ARTIFACTS:
        path = ROOT / name
        if path.exists():
            path.unlink()
    catboost_info = ROOT / "catboost_info"
    if catboost_info.exists():
        shutil.rmtree(catboost_info)


def execute_notebooks() -> None:
    """Execute tutorial notebooks sequentially."""
    for nb in NOTEBOOKS:
        nb_path = ROOT / "notebooks" / nb
        out_path = ROOT / "notebooks" / nb.replace(".ipynb", ".executed.ipynb")
        run_cmd(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                str(nb_path),
                "--output",
                str(out_path.name),
                "--output-dir",
                str(out_path.parent),
            ]
        )


def main() -> None:
    """Execute full verification workflow."""
    clean_generated_artifacts()

    pipeline = SmartLoanRecoveryPipeline(data_path=DATA_PATH, random_state=42, strict_mode=True)
    pipeline.run()

    execute_notebooks()

    run_cmd([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"])

    print("Full verification completed successfully.")


if __name__ == "__main__":
    main()
