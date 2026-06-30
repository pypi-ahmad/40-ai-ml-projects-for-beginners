"""Execute all project notebooks from top to bottom.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/run_notebooks.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

NOTEBOOK_DIR = Path("notebooks")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_JUPYTER_DIR = PROJECT_ROOT / "artifacts" / ".jupyter"
LOCAL_IPYTHON_DIR = PROJECT_ROOT / "artifacts" / ".ipython"


def main() -> None:
    LOCAL_JUPYTER_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_IPYTHON_DIR.mkdir(parents=True, exist_ok=True)

    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not notebooks:
        raise SystemExit("No notebooks found in notebooks/.")

    for notebook in notebooks:
        print(f"Executing {notebook} ...")
        cmd = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(notebook),
            "--output",
            notebook.name,
            "--output-dir",
            str(NOTEBOOK_DIR),
            "--ExecutePreprocessor.timeout=1800",
            "--ExecutePreprocessor.kernel_name=python3",
        ]
        env = dict(os.environ)
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(PROJECT_ROOT)
            if not current_pythonpath
            else f"{PROJECT_ROOT}:{current_pythonpath}"
        )
        env.setdefault("IPYTHONDIR", str(LOCAL_IPYTHON_DIR))
        env.setdefault("JUPYTER_RUNTIME_DIR", str(LOCAL_JUPYTER_DIR))
        env.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / ".mplconfig"))
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Notebook execution failed. If running in a restricted sandbox, "
                "kernel socket creation may be blocked. Run locally to complete full execution."
            ) from exc

    print("All notebooks executed successfully.")


if __name__ == "__main__":
    main()
