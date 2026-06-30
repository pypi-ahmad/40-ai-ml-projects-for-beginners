"""Execute tutorial notebook end-to-end."""

from __future__ import annotations

import subprocess
from pathlib import Path

NOTEBOOK = Path("notebooks/01_document_qa_zero_to_hero.ipynb")
OUTPUT = Path("notebooks/01_document_qa_zero_to_hero.executed.ipynb")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(NOTEBOOK),
        "--output",
        str(OUTPUT.name),
        "--output-dir",
        str(OUTPUT.parent),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
