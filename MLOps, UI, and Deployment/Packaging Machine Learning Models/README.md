# Packaging Machine Learning Models with Python

Project #12 from **40 AI/ML Projects for Beginners**

A production-oriented, beginner-friendly framework that teaches how a trained ML model becomes a reusable, versioned, testable software artifact.

## Executive Summary

Most beginner projects stop at:

`train -> save .pkl -> expose one endpoint`

This project goes further and teaches professional ML packaging workflows:

- Model packaging architecture
- Serialization strategy (Pickle, Joblib, ONNX, TorchScript)
- Security-aware artifact loading
- Version registry and rollback
- Reusable Python prediction package
- FastAPI serving with strict schemas
- CLI for batch and automation use-cases
- Testing strategy (unit, API, integration, security)
- Monitoring-ready metrics/logging
- Notebook-based zero-to-hero tutorial track

Dataset: **Iris** (focus is packaging and lifecycle engineering, not leaderboard chasing).

---

## Model Packaging Fundamentals

### What is model packaging?
Model packaging is converting trained models into governed software artifacts with:

- stable interfaces
- metadata and lineage
- integrity controls
- deployment-ready runtime behavior

### Lifecycle

`Raw Data -> Training -> Artifact -> Packaging -> Serving -> Production`

### Why packaging matters

- **Reproducibility**: deterministic training + saved artifacts + lockfile
- **Portability**: same model usable in API, CLI, scripts, notebooks
- **Versioning**: `v1`/`v2` promotion and rollback paths
- **Operations**: health checks, metrics, logs, error boundaries
- **Security**: safe deserialization policy + integrity verification

---

## Architecture

High-level flow:

`Client -> FastAPI -> Wrapper Layer -> Model Artifact -> Prediction`

### Core components

- `ml_package/model_loader.py`
  - unified loading/saving for `.pkl`, `.joblib`, `.onnx`, `.pt`
  - checksum manifest verification
  - trusted-digest allow-list enforcement
- `ml_package/prediction_engine.py`
  - consistent single/batch prediction contract
  - latency and model metadata
- `ml_package/validation.py`
  - Pydantic input contracts (single + batch)
- `ml_package/versioning.py`
  - model registry with lineage metadata and rollback
- `ml_package/explainability.py`
  - SHAP local/global explainability wrapper
- `api/main.py`
  - FastAPI service layer and endpoint orchestration
- `ml_package/cli/predict.py`
  - prediction CLI (`predict`, `batch`, `info`, `serve`)
- `scripts/train_model.py`
  - reproducible train/benchmark/serialize/register workflow
- `scripts/verify_project.py`
  - end-to-end verification pipeline

---

## Repository Layout

```text
api/                  # FastAPI app and request/response schemas
ml_package/           # Reusable prediction package
scripts/              # Training, figures, notebooks, full verification
tests/                # Unit + API + integration/security tests
models/               # Serialized artifacts + manifests + registry
outputs/              # Benchmarks + figures
notebooks/            # Tutorial notebooks + executed copies
pyproject.toml        # Packaging metadata and dependencies
uv.lock               # Reproducible lockfile
```

---

## Tooling Stack and Tradeoffs

### Mandatory stack
- Scikit-Learn
- FastAPI
- Pydantic
- Joblib
- Pickle

### Additional stack
- LazyPredict
- FLAML
- PyCaret
- XGBoost
- LightGBM
- CatBoost
- SHAP
- ONNX (`skl2onnx`, `onnxruntime`)
- TorchScript (`torch`)

### Why LazyPredict / FLAML / PyCaret all exist here

- **LazyPredict**
  - Strength: fast broad baseline discovery
  - Weakness: less control over deeper tuning
- **FLAML**
  - Strength: efficient AutoML under time budget
  - Weakness: less transparent than manual modeling for beginners
- **PyCaret**
  - Strength: highly productive low-code experiment API
  - Weakness: Python version compatibility constraints in many environments

### Python 3.12.10 + PyCaret note
This project keeps PyCaret integration and records runtime status explicitly.
On Python 3.12.10, PyCaret may report unsupported runtime and is marked as `skipped` in benchmark outputs instead of breaking the pipeline.

---

## Dataset, Benchmarking, and Evaluation

### Required model families benchmarked

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- SVM
- KNN
- XGBoost
- LightGBM
- CatBoost

Outputs:

- `outputs/benchmarks/model_benchmark.csv`
- `outputs/benchmarks/model_benchmark.json`
- `outputs/benchmarks/automl_benchmark.json`

### Current version summary (`outputs/benchmarks/version_comparison.json`)

- `v1` (LogisticRegression):
  - Accuracy: `0.9667`
  - F1 Macro: `0.9666`
- `v2` (KNN):
  - Accuracy: `1.0000`
  - F1 Macro: `1.0000`

### Metrics covered

- Accuracy
- Precision (macro)
- Recall (macro)
- F1 (macro)
- ROC AUC (OvR)

Visual diagnostics:

- confusion matrix
- ROC curves
- precision-recall curves

---

## Serialization Concepts and Security

Formats compared:

- Pickle (`.pkl`)
- Joblib (`.joblib`)
- ONNX (`.onnx`)
- TorchScript (`.pt`)

Measured:

- save time
- load time
- inference time
- file size

Output:

- `outputs/benchmarks/serialization_benchmark.json`

### Security model (hardened)

- `verify_integrity=True` and manifest verification required in API/CLI loading paths
- default policy is fail-closed for unsafe pickle/joblib loads
- unsafe formats are allowed only when artifact digest is trusted
- trusted digests can be supplied via env and registry metadata

Key risk reminder:

- Never deserialize untrusted pickle/joblib artifacts.

---

## Versioning and Lineage

Registry file: `models/registry.json`

Implemented flow:

- `v1`: baseline model
- `v2`: promoted candidate
- active version tracking + rollback support

Metadata tracked per version:

- model path
- metrics
- artifact SHA256
- dataset fingerprint
- parent version
- tags
- creation timestamp

---

## FastAPI Integration

Run API:

```bash
MPLCONFIGDIR=.mplconfig UV_CACHE_DIR=.uv-cache uv run --no-sync uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict-batch`
- `GET /metrics`
- `GET /metrics/prometheus`
- `POST /explain`

OpenAPI docs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## Validation Layer (Pydantic)

Validation guarantees include:

- strict payload schemas (`extra='forbid'`)
- required feature fields
- type checking
- realistic range constraints
- finite-value checks
- batch size limits

Invalid payloads return structured `422` responses.

---

## Reusable Python Package and CLI

### Editable install (verified locally)

```bash
UV_CACHE_DIR=.uv-cache uv pip install -e . --no-deps --no-build-isolation
```

### CLI examples

Single prediction:

```bash
MPLCONFIGDIR=.mplconfig UV_CACHE_DIR=.uv-cache uv run --no-sync ml-predict --model-path models/iris_model.pkl predict --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2
```

Batch prediction:

```bash
MPLCONFIGDIR=.mplconfig UV_CACHE_DIR=.uv-cache uv run --no-sync ml-predict --model-path models/iris_model.pkl batch path/to/samples.json
```

Model info:

```bash
MPLCONFIGDIR=.mplconfig UV_CACHE_DIR=.uv-cache uv run --no-sync ml-predict --model-path models/iris_model.pkl info
```

---

## Monitoring and Logging

- prediction event logging
- error logging
- load-time logging
- latency statistics
- JSON metrics endpoint (`/metrics`)
- Prometheus text endpoint (`/metrics/prometheus`)

---

## Explainability

`POST /explain` supports:

- `mode=local`: per-sample SHAP attribution
- `mode=global`: global feature importance ranking

Wrapper handles SHAP availability and returns explicit error payload when unavailable.

---

## Notebooks (Mini-Book Track)

1. `notebooks/01_model_packaging_foundations.ipynb`
2. `notebooks/02_iris_eda_and_understanding.ipynb`
3. `notebooks/03_model_benchmarking_and_selection.ipynb`
4. `notebooks/04_evaluation_metrics_and_diagnostics.ipynb`
5. `notebooks/05_serialization_deep_dive.ipynb`
6. `notebooks/06_wrapper_api_validation.ipynb`
7. `notebooks/07_versioning_testing_and_cli.ipynb`
8. `notebooks/08_monitoring_explainability_and_deployment.ipynb`

Executed copies are generated as `*.executed.ipynb`.

Each notebook follows:

- definition
- theory
- motivation
- real-world framing
- visual explanation
- code explanation
- best practices
- common mistakes

---

## Setup and Reproducibility

### Environment setup

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

### Individual stages

```bash
make train
make figures
make notebooks
make notebooks-exec
make test
```

### Full end-to-end verification

```bash
make verify
```

This runs:

- training + serialization + versioning
- figure generation
- notebook generation + execution
- full pytest suite
- CLI smoke checks
- API smoke checks

---

## Testing Strategy

Run tests:

```bash
MPLCONFIGDIR=.mplconfig UV_CACHE_DIR=.uv-cache uv run --no-sync pytest -q
```

Coverage layers:

- unit tests: loader, validation, prediction engine, logging, versioning
- security tests: manifest, digest allow-list, unsafe deserialization blocking
- API tests: schema and endpoint behavior
- integration tests: live API checks (`requests`, auto-skipped when socket bind is restricted)
- CLI tests: single/batch parsing, malformed inputs, command error behavior

---

## Figures and Artifacts

Architecture/lifecycle figures:

- `outputs/figures/packaging_architecture.png`
- `outputs/figures/packaging_lifecycle_workflow.png`
- `outputs/figures/api_prediction_flow.png`

Benchmark figures:

- `outputs/figures/model_benchmark_scores.png`
- `outputs/figures/model_inference_latency.png`
- `outputs/figures/serialization_benchmark.png`
- `outputs/figures/automl_framework_comparison.png`

---

## Lessons Learned

- model packaging is software engineering, not just model serialization
- registry + checksums + trust policy are core to safe model operations
- strict validation at system boundaries prevents silent runtime failures
- API + CLI + package reuse is a strong production and portfolio pattern
- reproducibility requires executable workflows, not only narrative docs

---

## Production Recommendations

- add signed artifacts (beyond checksum) for higher trust environments
- add CI gates for full `make verify`
- export metrics to Prometheus/Grafana
- add model drift monitoring and alerting
- add release automation for version promotion/rollback workflows

---

## Final Verification Report

See: `FINAL_PROJECT_VERIFICATION_REPORT.md`

It documents:

- audit findings
- fixes implemented
- reproducibility evidence
- security/versioning/API/testing review
- final scoring and remaining limitations
