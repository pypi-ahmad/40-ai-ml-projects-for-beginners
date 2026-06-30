"""Run an end-to-end local verification workflow for the project."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import asyncio
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_step(name: str, cmd: list[str]) -> None:
    print(f"\n[verify] {name}")
    print(f"[verify] cmd: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def smoke_test_api() -> None:
    from httpx import ASGITransport, AsyncClient

    from api.main import app, service, settings as api_settings

    print("\n[verify] API smoke checks")
    if service.engine is None:
        service.load(api_settings.model_path)

    async def _run() -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://verify") as client:
            health = await client.get("/health")
            health.raise_for_status()
            payload = health.json()
            if payload["status"] not in {"healthy", "degraded"}:
                raise RuntimeError(f"Unexpected health payload: {payload}")

            sample = {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
            predict = await client.post("/predict", json=sample)
            predict.raise_for_status()
            if "prediction" not in predict.json():
                raise RuntimeError("Missing prediction field in /predict response")

            metrics = await client.get("/metrics")
            metrics.raise_for_status()
            if "prediction_stats" not in metrics.json():
                raise RuntimeError("Missing prediction_stats in /metrics response")

    asyncio.run(_run())

    print("[verify] API smoke checks passed")


def smoke_test_cli() -> None:
    print("\n[verify] CLI smoke checks")
    cmd = [
        sys.executable,
        "-m",
        "ml_package.cli.predict",
        "--model-path",
        "models/iris_model.pkl",
        "predict",
        "--sepal-length",
        "5.1",
        "--sepal-width",
        "3.5",
        "--petal-length",
        "1.4",
        "--petal-width",
        "0.2",
    ]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=True)
    start = result.stdout.find("{")
    end = result.stdout.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError(f"CLI output did not contain JSON payload. output={result.stdout}")
    json_blob = result.stdout[start:end + 1]
    payload = json.loads(json_blob)
    if "prediction" not in payload:
        raise RuntimeError("CLI prediction payload missing prediction field")
    print("[verify] CLI smoke checks passed")


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    os.environ.setdefault("UV_CACHE_DIR", str(ROOT / ".uv-cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(ROOT / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
    Path(ROOT / "outputs" / "benchmarks").mkdir(parents=True, exist_ok=True)

    run_step("Train and package models", [sys.executable, "scripts/train_model.py"])
    run_step("Generate figures", [sys.executable, "scripts/generate_figures.py"])
    run_step("Regenerate notebook sources", [sys.executable, "scripts/generate_notebooks.py"])
    run_step(
        "Execute notebooks",
        [sys.executable, "scripts/execute_notebooks.py", "--notebook-dir", "notebooks"],
    )
    run_step("Run test suite", [sys.executable, "-m", "pytest", "-q"])
    smoke_test_cli()
    smoke_test_api()

    print("\n[verify] Full local verification completed successfully.")


if __name__ == "__main__":
    main()
