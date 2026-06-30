"""Execute project notebooks end-to-end and save executed copies."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _discover_notebooks(notebook_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in notebook_dir.glob("*.ipynb")
        if not path.name.endswith(".executed.ipynb")
    )


def execute_notebook(path: Path, timeout: int) -> Path:
    output_stem = f"{path.stem}.executed"
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--output",
        output_stem,
        "--output-dir",
        str(path.parent),
        "--ExecutePreprocessor.timeout",
        str(timeout),
        "--ExecutePreprocessor.kernel_name",
        "python3",
        str(path),
    ]
    subprocess.run(cmd, check=True)
    return path.parent / f"{output_stem}.ipynb"


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute tutorial notebooks end-to-end.")
    parser.add_argument(
        "--notebook-dir",
        default="notebooks",
        help="Directory containing notebook sources.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Per-notebook execution timeout in seconds.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    notebook_dir = (root / args.notebook_dir).resolve()
    if not notebook_dir.exists():
        raise FileNotFoundError(f"Notebook directory not found: {notebook_dir}")

    os.environ.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))
    os.environ.setdefault("UV_CACHE_DIR", str(root / ".uv-cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("outputs/benchmarks").mkdir(parents=True, exist_ok=True)

    notebooks = _discover_notebooks(notebook_dir)
    if not notebooks:
        raise RuntimeError(f"No notebook sources found under {notebook_dir}")

    print(f"Executing {len(notebooks)} notebook(s) from {notebook_dir}")
    for notebook in notebooks:
        print(f"Executing: {notebook.name}")
        executed_path = execute_notebook(notebook, timeout=args.timeout)
        print(f"Saved: {executed_path}")


if __name__ == "__main__":
    main()

