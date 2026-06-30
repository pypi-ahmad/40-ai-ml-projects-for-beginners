"""Deterministic project verification runner.

Runs lint, tests, pipeline, and notebook execution, then writes a summary JSON.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "reports" / "verification_summary.json"


def run_step(name: str, cmd: list[str]) -> dict[str, object]:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    return {
        "name": name,
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 3),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "passed": proc.returncode == 0,
    }


def main() -> int:
    steps = [
        ("ruff", [sys.executable, "-m", "ruff", "check", "src", "scripts", "streamlit_app", "tests"]),
        ("pytest", [sys.executable, "-m", "pytest", "-q"]),
        (
            "pipeline",
            [
                sys.executable,
                "scripts/run_pipeline.py",
                "--algorithms",
                "kmeans",
                "minibatch_kmeans",
                "dbscan",
                "hdbscan",
                "agglomerative",
                "gmm",
                "--run-automl",
            ],
        ),
        ("notebooks", [sys.executable, "scripts/run_notebooks.py"]),
    ]

    results = [run_step(name, command) for name, command in steps]
    overall_passed = all(step["passed"] for step in results)

    payload = {
        "project_root": str(PROJECT_ROOT),
        "generated_at_epoch": int(time.time()),
        "overall_passed": overall_passed,
        "steps": results,
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Verification summary written to {SUMMARY_PATH}")
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
