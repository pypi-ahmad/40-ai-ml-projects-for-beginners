"""Run optional FastAPI/API-track quality gates.

This is a convenience wrapper around ``validate_project.py --include-api``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optional API quality gates.")
    parser.add_argument(
        "--skip-notebooks",
        action="store_true",
        help="Skip notebook execution checks.",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Pass --no-sync to nested uv run commands (use after uv sync).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    cmd = ["uv", "run", "python", "scripts/validate_project.py", "--include-api"]
    if args.skip_notebooks:
        cmd.append("--skip-notebooks")
    if args.no_sync:
        cmd.append("--no-sync")

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")

    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
    if result.returncode != 0:
        print("[FAIL] Optional API quality gate failed.")
        return 1

    print("[OK] Optional API quality gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
