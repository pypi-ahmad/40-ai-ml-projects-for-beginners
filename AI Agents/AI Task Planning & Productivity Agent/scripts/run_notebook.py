"""Execute notebooks sequentially for deterministic validation."""

from __future__ import annotations

import subprocess
from pathlib import Path


NOTEBOOKS = [
    "notebooks/01_zero_to_hero_productivity_agent.ipynb",
]


def main() -> None:
    for notebook in NOTEBOOKS:
        path = Path(notebook)
        if not path.exists():
            raise FileNotFoundError(f"Notebook missing: {notebook}")
        subprocess.run(
            [
                "uv",
                "run",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                str(path),
                "--output",
                str(path.name),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
