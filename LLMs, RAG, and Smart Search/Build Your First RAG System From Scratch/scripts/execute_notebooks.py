"""Execute all tutorial notebooks end-to-end."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("IPYTHONDIR", str((Path.cwd() / ".ipython").resolve()))
Path(os.environ["IPYTHONDIR"]).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute project notebooks")
    parser.add_argument("--path", default="notebooks", help="Notebook directory")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-notebook timeout seconds")
    return parser.parse_args()


def execute_notebook(path: Path, timeout: int) -> None:
    logger.info("Executing %s", path.name)
    with path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": str(path.parent)}})
    except Exception as exc:
        raise RuntimeError(
            f"Notebook execution failed for {path.name}. "
            "This environment may block kernel socket creation; run in an unrestricted local shell."
        ) from exc

    executed_path = path.with_name(path.stem + ".executed.ipynb")
    with executed_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    logger.info("Executed notebook saved: %s", executed_path.name)


def main() -> None:
    args = parse_args()
    nb_dir = Path(args.path)
    notebooks = sorted(nb_dir.glob("*.ipynb"))

    if not notebooks:
        raise FileNotFoundError(f"No notebooks found in {nb_dir}")

    for notebook_path in notebooks:
        if notebook_path.name.endswith(".executed.ipynb"):
            continue
        execute_notebook(notebook_path, timeout=args.timeout)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
