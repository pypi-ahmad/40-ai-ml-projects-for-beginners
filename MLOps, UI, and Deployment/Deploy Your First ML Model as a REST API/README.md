# Deploy Your First ML Model as a REST API (Project #9)

Portfolio-grade end-to-end ML API system built with FastAPI.

This project teaches how to move from model notebook work to production-style model serving with validation, testing, observability, and explainability.

## 1) Why APIs Exist (Beginner Zero-to-Hero)

### What is an API?
An API (Application Programming Interface) is contract that lets one system request work from another system.

### Restaurant analogy
- Client: customer
- Server: waiter
- Request: order
- Endpoint: menu item route
- Response: food + bill (status + JSON)

### Request-response lifecycle
Client -> REST API -> Validation Layer -> ML Model -> Prediction -> JSON Response

### Why this matters for ML
- Model training and model serving are different concerns.
- API makes model reusable for web apps, backend jobs, and partner systems.
- API contracts make behavior testable and monitorable.

## 2) REST Fundamentals

### HTTP methods in this project
- `GET` read service/model state (`/health`, `/model-info`, `/metrics`)
- `POST` send inference/explain payload (`/predict`, `/predict-batch`, `/explain`)

### Request components
- Body: JSON payload with named California Housing fields
- Headers: optional `X-API-Key`, optional `X-Request-ID`
- Query params: optional in future extensions

### Status codes
- `200` success
- `401` auth failure
- `413` payload too large
- `422` validation failure
- `429` rate limit exceeded
- `503` model not loaded
- `500` unexpected internal failure

## 3) FastAPI + Pydantic Deep Dive

### Why FastAPI
- Native type hints -> request/response validation
- Automatic OpenAPI docs (`/docs`, `/redoc`)
- Strong developer experience for ML services

### Why Pydantic
- Runtime validation
- Type safety
- Structured serialization
- Generated API schema/docs

### Framework tradeoff summary
- Flask: minimal, flexible, more manual typing/docs work
- FastAPI: typed, fast development, auto docs
- Django: batteries-included, heavier for focused inference API
- Node APIs: ecosystem broad, type-safety depends on chosen stack

## 4) Project Architecture

```text
app/
  config.py                  # environment + runtime settings
  main.py                    # app factory, middleware stack
  models/
    schemas.py               # Pydantic contracts
    predictor.py             # model loading/predict/explain
  routers/
    health.py                # /health, /admin/reload
    info.py                  # /model-info (+ /info alias)
    predict.py               # /predict, /predict-batch
    explain.py               # /explain
    metrics.py               # /metrics
  services/
    validation.py            # finite value + batch guards
    metrics_store.py         # SQLite request metrics
    tracking.py              # optional W&B init
  utils/
    middleware.py            # auth/rate limit/request-id/logging/metrics
    exceptions.py            # global error envelope
    logging.py               # loguru config

src/
  data.py                    # California Housing loader + splits
  training.py                # model candidates + selection
  evaluation.py              # MAE/MSE/RMSE/R²/MAPE
  benchmarking.py            # LazyPredict/FLAML/PyCaret adapters
  serialization.py           # joblib metadata + optional ONNX
  visualization.py           # model/perf charts

scripts/
  train.py                   # end-to-end training + benchmarking + artifacts
  benchmark_api.py           # latency/throughput (cold/warm/single/batch)
  verify_environment.py      # local runtime dependency/artifact checks
  run_notebooks.py           # execute all notebooks end-to-end
  generate_notebooks.py      # regenerate tutorial notebooks

notebooks/
  01_api_fundamentals_rest.ipynb
  02_fastapi_pydantic.ipynb
  03_dataset_eda_modeling.ipynb
  04_benchmarking_and_selection.ipynb
  05_serialization_serving_xai.ipynb
  06_testing_monitoring_performance.ipynb

docs/
  architecture.md            # mermaid architecture + request flow diagrams
  performance.md             # measured latency/throughput/memory notes
```

## 5) Dataset + Modeling (California Housing)

- Target: `MedHouseVal`
- Features: `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`
- Deterministic split: 70% train / 15% val / 15% test
- Seed: `42`

### Models benchmarked
- LinearRegression
- Ridge
- Lasso
- ElasticNet
- RandomForest
- ExtraTrees
- XGBoost (if installed)
- LightGBM (if installed)
- CatBoost (if installed)

### AutoML/benchmark tooling
- LazyPredict: quick broad scan
- FLAML: budget-aware AutoML
- PyCaret: high-level experimentation pipeline

Training profiles control runtime and AutoML scope:
- `--profile quick`: fast deterministic run; AutoML tools are skipped with explicit notes.
- `--profile full`: runs LazyPredict/FLAML/PyCaret adapters (if installed).
- `--automl-max-train-rows`: reproducible downsampling for heavy AutoML tools.
- `--flaml-seconds`: FLAML time budget.

If optional tools are missing, benchmark outputs explicitly record `status=skipped` with import error notes.

### Metric definitions
- MAE: mean absolute error
- MSE: mean squared error
- RMSE: square root of MSE
- R²: explained variance score
- MAPE: mean absolute percentage error

Artifacts saved to:
- `models/model.joblib`
- `models/metadata.json`
- `artifacts/benchmarks/*.csv`
- `artifacts/figures/*.png`
- `artifacts/reports/*`

## 6) Serialization: Pickle vs Joblib vs ONNX

- Pickle: flexible Python serialization, unsafe with untrusted files
- Joblib: optimized for NumPy/sklearn objects, standard local serving choice
- ONNX: runtime-portable format, extra conversion constraints

This project uses Joblib for serving and attempts optional ONNX export during training.

## 7) API Endpoints

### `GET /health`
Service liveness + model + metrics DB readiness.

### `GET /model-info`
Model metadata and feature schema version.

### `POST /predict`
Single record inference.

### `POST /predict-batch`
Batch inference with configurable size cap.

### `POST /explain`
SHAP-based local explanation for one record.

### `GET /metrics`
SQLite-backed API telemetry + model evaluation metrics.

### Request schema (`/predict`)
```json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.9841,
  "AveBedrms": 1.0238,
  "Population": 322.0,
  "AveOccup": 2.5556,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

## 8) Security Basics

- Strict Pydantic schema validation
- NaN/inf rejection
- Optional API key (`X-API-Key`) for inference/explain endpoints
- In-memory per-IP rate limiting
- Security headers middleware
- Request size limiting (`MAX_REQUEST_BODY_BYTES`) with content-length and body-length checks
- Request ID propagation for traceability

### Standard error envelope
```json
{
  "code": "VALIDATION_ERROR",
  "detail": "Request validation failed.",
  "request_id": "uuid",
  "field_errors": [{"field": "Latitude", "message": "Field required"}]
}
```

## 9) Testing Strategy

### Manual testing
- Browser + Swagger UI (`/docs`)
- ReDoc (`/redoc`)

### Programmatic testing
- `httpx` / `requests` examples in notebooks and scripts

### Automated testing
- Pytest integration suite in `tests/test_api.py`
- Validates success and failure paths for all key endpoints

## 10) Performance Benchmarking

`benchmark_api.py` measures:
- Cold start latency
- Warm latency (avg/p50/p95)
- Throughput (single and batch)
- Batch size impact

Outputs:
- `artifacts/performance/api_performance_summary.json`
- `artifacts/performance/single_predict_latencies.csv`
- `artifacts/performance/single_latency_histogram.png`

## 11) Configuration

Use `.env` (copy from `.env.example`).

Important settings:
- `MODEL_PATH`, `METADATA_PATH`
- `API_KEY` (optional)
- `MAX_BATCH_SIZE`
- `RATE_LIMIT_PER_MINUTE`
- `METRICS_DB_PATH`

## 12) Quickstart

```bash
# 1) Create local venv (project-local)
uv venv .venv
source .venv/bin/activate

# 2) Install runtime + dev + training extras
uv sync
uv sync --extra dev --extra train

# 3) Train and generate artifacts
python scripts/train.py --profile quick

# Optional full benchmark profile (requires train extras installed)
python scripts/train.py --profile full --flaml-seconds 120 --automl-max-train-rows 4000

# 4) Start API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 5) Open docs
# http://127.0.0.1:8000/docs
```

## 13) Execute Entire Learning Pipeline

```bash
# Rebuild notebooks from template generator
python scripts/generate_notebooks.py

# Execute notebooks top-to-bottom
python scripts/run_notebooks.py

# Verify local environment + optional dependencies
python scripts/verify_environment.py

# Run API tests
pytest -v

# Run API perf benchmark (API must be running)
python scripts/benchmark_api.py --base-url http://127.0.0.1:8000

# Socket-restricted environments: benchmark in-process via ASGI transport
python scripts/benchmark_api.py --in-process --request-log-sample-rate 0.02
```

## 14) CI and Reproducibility

- GitHub Actions workflow: `.github/workflows/ci.yml`
- CI executes:
  - compile checks
  - pytest API/unit suite
  - quick training profile
  - in-process performance benchmark
- CI uploads generated `artifacts/` and `models/` for inspection.

## 15) Lessons Learned

- API-first thinking forces explicit contracts and better engineering discipline.
- Validation + structured errors are as important as model quality.
- Benchmarking multiple models/tools reveals tradeoffs hidden by single-model tutorials.
- Production readiness requires observability, not only prediction endpoint availability.

## 16) Future Improvements (Next Project Bridge)

- Full Docker/Kubernetes deployment pipeline
- CI/CD with automated notebook execution + contract tests
- External metrics backend (Prometheus/Grafana)
- Model registry + signed artifact promotion flow
- Canary and shadow deployment support
