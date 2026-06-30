"""Run deployment quality gates.

Default mode validates the Streamlit-first deployment track.
Use ``--include-api`` to run the optional FastAPI/API-track checks.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _uv_run(*args: str, no_sync: bool) -> list[str]:
    cmd = ["uv", "run"]
    if no_sync:
        cmd.append("--no-sync")
    cmd.extend(args)
    return cmd


def _streamlit_required_checks(*, no_sync: bool) -> list[list[str]]:
    return [
        _uv_run("pytest", "-q", no_sync=no_sync),
        _uv_run("python", "scripts/generate_diagrams.py", no_sync=no_sync),
        _uv_run("python", "scripts/benchmark_runtime.py", no_sync=no_sync),
        _uv_run("python", "scripts/build_notebooks.py", no_sync=no_sync),
        _uv_run(
            "python",
            "-c",
            "from streamlit_app.app import main; print('streamlit-import-ok')",
            no_sync=no_sync,
        ),
        _uv_run("python", "scripts/capture_evidence.py", no_sync=no_sync),
    ]


def _api_optional_checks(*, no_sync: bool) -> list[list[str]]:
    return [
        _uv_run("python", "scripts/generate_ames_snapshot.py", no_sync=no_sync),
        _uv_run("python", "scripts/train_api_models.py", no_sync=no_sync),
        _uv_run("python", "scripts/generate_api_diagrams.py", no_sync=no_sync),
        _uv_run("python", "scripts/build_api_notebooks.py", no_sync=no_sync),
        _uv_run("pytest", "-q", "-o", "addopts=", "tests/api", no_sync=no_sync),
        _uv_run("python", "scripts/benchmark_api_runtime.py", no_sync=no_sync),
    ]


def _kernel_execution_supported(*, no_sync: bool) -> bool:
    """Return True when local socket creation is available for Jupyter kernels."""
    probe = subprocess.run(
        _uv_run("python", "-c", "import socket; s=socket.socket(); s.close(); print('ok')", no_sync=no_sync),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deployment quality gates.")
    parser.add_argument(
        "--include-api",
        action="store_true",
        help="Run optional FastAPI/API-track checks in addition to Streamlit checks.",
    )
    parser.add_argument(
        "--skip-notebooks",
        action="store_true",
        help="Skip notebook execution checks.",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Pass --no-sync to uv run commands (use after uv sync).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")

    checks = _streamlit_required_checks(no_sync=args.no_sync)
    if args.include_api:
        checks.extend(_api_optional_checks(no_sync=args.no_sync))

    if args.skip_notebooks:
        print("[SKIP] Notebook execution skipped by --skip-notebooks flag.")
    elif _kernel_execution_supported(no_sync=args.no_sync):
        notebook_cmd = _uv_run("python", "scripts/execute_notebooks.py", no_sync=args.no_sync)
        if args.include_api:
            notebook_cmd.append("--include-api")
        checks.append(notebook_cmd)
    else:
        print("[SKIP] Notebook execution skipped: local socket creation is unavailable in this runtime.")

    failed = False
    for cmd in checks:
        print(f"\n[RUN] {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
        if result.returncode != 0:
            failed = True
            print(f"[FAIL] {' '.join(cmd)}")
        else:
            print(f"[OK] {' '.join(cmd)}")

    if failed:
        print("\nValidation failed. Fix failures before deployment.")
        return 1

    print("\nAll validation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
