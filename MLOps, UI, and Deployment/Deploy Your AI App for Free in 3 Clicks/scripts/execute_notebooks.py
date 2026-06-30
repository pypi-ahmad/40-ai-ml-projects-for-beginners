"""Execute all tutorial notebooks end-to-end.

Usage:
    uv run python scripts/execute_notebooks.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"
EXECUTED_DIR = ROOT / "outputs" / "executed_notebooks"


def execute_notebook(path: Path, timeout: int, inplace: bool) -> tuple[bool, float, str]:
    """Execute one notebook and return status tuple."""
    start = time.perf_counter()
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )

    try:
        client.execute()
        elapsed = time.perf_counter() - start

        if inplace:
            out_path = path
        else:
            relative = path.relative_to(NOTEBOOKS_DIR)
            out_path = EXECUTED_DIR / relative.parent / f"{path.stem}.executed.ipynb"
            out_path.parent.mkdir(parents=True, exist_ok=True)

        nbformat.write(nb, out_path)
        return True, elapsed, str(out_path)
    except CellExecutionError as err:
        elapsed = time.perf_counter() - start
        return False, elapsed, str(err)
    except Exception as err:
        elapsed = time.perf_counter() - start
        return False, elapsed, str(err)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute project notebooks")
    parser.add_argument("--timeout", type=int, default=900, help="Per-cell timeout in seconds")
    parser.add_argument("--inplace", action="store_true", help="Write execution outputs back to source notebooks")
    parser.add_argument(
        "--include-api",
        action="store_true",
        help="Also execute notebooks/api track (requires API dependencies).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    notebooks = sorted(path for path in NOTEBOOKS_DIR.rglob("*.ipynb") if path.is_file())
    if not args.include_api:
        notebooks = [
            path
            for path in notebooks
            if path.relative_to(NOTEBOOKS_DIR).parts[0] != "api"
        ]
    if not notebooks:
        print("No notebooks found.")
        return 1

    print(f"Executing {len(notebooks)} notebook(s)...")
    failed = False

    for notebook in notebooks:
        print(f"\n[RUN] {notebook.name}")
        ok, elapsed, message = execute_notebook(notebook, timeout=args.timeout, inplace=args.inplace)
        if ok:
            print(f"[OK] {notebook.name} in {elapsed:.1f}s -> {message}")
        else:
            failed = True
            print(f"[FAIL] {notebook.name} in {elapsed:.1f}s")
            print(message)

    if failed:
        print("\nNotebook execution failed.")
        return 1

    print("\nAll notebooks executed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
