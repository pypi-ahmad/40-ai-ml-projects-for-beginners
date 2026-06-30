"""Execute all tutorial notebooks end-to-end and save executed copies."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for notebook execution."""

    parser = argparse.ArgumentParser(description="Execute project notebooks.")
    parser.add_argument(
        "--notebook-dir",
        default="notebooks",
        help="Source notebook directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/executed_notebooks",
        help="Directory to store executed notebook copies.",
    )
    parser.add_argument("--timeout", type=int, default=1800, help="Per-cell timeout in seconds.")
    return parser.parse_args()


def execute_notebook(path: Path, output_dir: Path, timeout: int, execution_dir: Path) -> Path:
    """Execute one notebook and persist executed copy."""

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(execution_dir)}},
    )

    logger.info("Executing notebook: %s", path)
    try:
        client.execute()
    except PermissionError as exc:
        raise RuntimeError(
            "Notebook execution requires local kernel socket access, but the environment "
            f"blocked it ({exc}). Run this script on a normal local shell."
        ) from exc
    except OSError as exc:
        if exc.errno == 1:
            raise RuntimeError(
                "Notebook kernel startup failed due restricted socket permissions. "
                "Run notebook execution outside sandboxed environment."
            ) from exc
        raise

    output_path = output_dir / path.name
    nbformat.write(notebook, output_path)
    return output_path


def main() -> None:
    """Execute all notebooks in lexical order."""

    args = parse_args()
    source_dir = Path(args.notebook_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_dir = source_dir.resolve().parent

    notebook_paths = sorted(source_dir.glob("*.ipynb"))
    if not notebook_paths:
        raise FileNotFoundError(f"No notebooks found in {source_dir}")

    executed_outputs = []
    try:
        for notebook_path in notebook_paths:
            executed_outputs.append(
                execute_notebook(notebook_path, output_dir, args.timeout, execution_dir)
            )
    except RuntimeError as exc:
        logger.error("Notebook execution failed: %s", exc)
        sys.exit(1)

    logger.info("Executed %d notebooks.", len(executed_outputs))
    for path in executed_outputs:
        logger.info("- %s", path)


if __name__ == "__main__":
    main()
