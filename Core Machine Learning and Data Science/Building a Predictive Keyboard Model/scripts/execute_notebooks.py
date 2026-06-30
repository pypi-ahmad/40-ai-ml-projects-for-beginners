"""Execute all tutorial notebooks top-to-bottom.

Usage:
    uv run python scripts/execute_notebooks.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    notebook_dir = project_root / "notebooks"
    executed_dir = project_root / "outputs" / "executed_notebooks"
    executed_dir.mkdir(parents=True, exist_ok=True)

    notebooks = sorted(notebook_dir.glob("*.ipynb"))
    if not notebooks:
        raise FileNotFoundError("No notebooks found in notebooks/.")

    for notebook in notebooks:
        out_name = notebook.stem + "-executed.ipynb"
        cmd = [
            "uv",
            "run",
            "jupyter-nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(notebook),
            "--output",
            out_name,
            "--output-dir",
            str(executed_dir),
            "--ExecutePreprocessor.timeout=3600",
        ]
        print("$", " ".join(cmd))
        try:
            subprocess.run(cmd, cwd=project_root, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Notebook execution failed. If you see 'Operation not permitted' "
                "from jupyter kernel startup, run this script outside restricted "
                "sandbox mode."
            ) from exc

    print(f"Executed notebooks saved to: {executed_dir}")


if __name__ == "__main__":
    main()
