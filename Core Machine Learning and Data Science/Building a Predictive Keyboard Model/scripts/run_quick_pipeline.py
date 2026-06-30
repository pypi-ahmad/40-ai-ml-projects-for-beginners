"""One-command quick pipeline runner for local verification."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    run(
        [
            "uv",
            "run",
            "python",
            "scripts/prepare_data.py",
            "--include-wikitext",
            "--wikitext-train-tokens",
            "60000",
            "--wikitext-val-tokens",
            "10000",
            "--wikitext-test-tokens",
            "10000",
        ],
        cwd=project_root,
    )
    run(
        [
            "uv",
            "run",
            "python",
            "scripts/train_and_benchmark.py",
            "--profile",
            "quick",
            "--include-wikitext",
            "--prefer-gpu",
            "--wikitext-train-tokens",
            "60000",
            "--wikitext-val-tokens",
            "10000",
            "--wikitext-test-tokens",
            "10000",
        ],
        cwd=project_root,
    )


if __name__ == "__main__":
    main()
