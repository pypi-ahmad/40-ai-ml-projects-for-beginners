"""Generate API-focused educational notebooks for FastAPI + ML serving."""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "api"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


def write_notebook(name: str, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    out_path = OUT_DIR / name
    nbf.write(nb, out_path)
    print(f"Wrote {out_path.relative_to(ROOT)}")


def build_01_api_fundamentals() -> None:
    cells = [
        md(
            """
            # API Track 01 - REST and FastAPI Foundations

            ## Definition
            A REST API exposes resources through HTTP methods with clear request/response contracts.

            ## Motivation
            Serving ML through APIs decouples training from consumers and enables backend/frontend integration.

            ## Architecture
            ![FastAPI Architecture](../../outputs/figures/fastapi-architecture.png)
            """
        ),
        code(
            """
            from ml_api.app import create_app

            app = create_app()
            app.title, app.version
            """
        ),
        md(
            """
            ## Best Practices
            - Use strict schema validation.
            - Keep a stable error envelope.
            - Instrument latency and request counts.

            ## Common Mistakes
            - Mixing training code into request handlers.
            - Returning inconsistent response shapes.
            - Ignoring invalid input paths.
            """
        ),
    ]
    write_notebook("01_rest_and_fastapi_foundations.ipynb", cells)


def build_02_model_serving() -> None:
    cells = [
        md(
            """
            # API Track 02 - Model Training and Serving Contracts

            Covers dataset loading, preprocessing split discipline, classical model benchmarking,
            and artifact serialization.
            """
        ),
        code(
            """
            import json
            from pathlib import Path

            benchmark_path = Path('../../outputs/api_benchmarks/model_benchmark.json')
            if benchmark_path.exists():
                data = json.loads(benchmark_path.read_text())
                data[:3]
            else:
                print('Run: uv run python scripts/train_api_models.py')
            """
        ),
        md(
            """
            ## Validation Principles
            - Fit preprocessing on train split only.
            - Compare every model on identical splits.
            - Persist metadata with feature schema hash.
            """
        ),
    ]
    write_notebook("02_model_training_and_serving.ipynb", cells)


def build_03_testing() -> None:
    cells = [
        md(
            """
            # API Track 03 - Testing and Contract Verification

            This chapter demonstrates endpoint tests, invalid payload checks, and docs availability checks.
            """
        ),
        code(
            """
            import subprocess

            subprocess.run(['uv', 'run', 'pytest', '-q', 'tests/api'], check=False)
            """
        ),
        md(
            """
            ## Common Mistakes
            - Testing only success paths.
            - Skipping batch validation behavior.
            - Leaving docs unverified.
            """
        ),
    ]
    write_notebook("03_testing_validation_and_contracts.ipynb", cells)


def build_04_observability() -> None:
    cells = [
        md(
            """
            # API Track 04 - Observability, Explainability, and Performance

            Explains `/metrics` and `/explain` contracts and runtime benchmarking discipline.

            ![Request Flow](../../outputs/figures/fastapi-request-flow.png)
            """
        ),
        code(
            """
            import json
            from pathlib import Path

            perf_path = Path('../../outputs/metrics/fastapi_runtime_benchmark.json')
            if perf_path.exists():
                snapshot = json.loads(perf_path.read_text())
                {
                    'startup_ms': snapshot['startup_ms'],
                    'single_predict_ms': snapshot['single_predict_ms'],
                    'batch_predict_ms': snapshot['batch_predict_ms'],
                }
            else:
                print('Run: uv run python scripts/benchmark_api_runtime.py')
            """
        ),
        md(
            """
            ## Lessons
            - Batch endpoints should be vectorized, not looped over single calls.
            - Explanations should include method metadata.
            - Metrics should be useful to operators, not just developers.
            """
        ),
    ]
    write_notebook("04_observability_explainability_and_performance.ipynb", cells)


def main() -> int:
    build_01_api_fundamentals()
    build_02_model_serving()
    build_03_testing()
    build_04_observability()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
