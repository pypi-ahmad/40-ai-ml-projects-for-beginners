# Deploy Your First ML Model as a REST API — Design Spec

## Goal

Build a production-quality ML API system teaching end-to-end model serving with FastAPI. Reader finishes understanding API fundamentals, REST architecture, FastAPI, Pydantic, model serving, serialization, validation, API testing, and production engineering.

## Architecture

```
Client (Browser/curl/httpx)
  ↓  HTTP Request (JSON)
REST API (FastAPI + Uvicorn)
  ↓  Pydantic Validation
Validation Layer
  ↓  Validated Input
ML Predictor (Joblib-loaded model)
  ↓  Raw Prediction
Post-processing (inverse transform)
  ↓
Response (JSON)
  ↑  Back to Client
```

## Component Design

### 1. Application Package (`app/`)

- **`main.py`** — FastAPI app factory, lifespan, middleware registration, router includes
- **`config.py`** — Pydantic `BaseSettings` for env vars (`MODEL_PATH`, `LOG_LEVEL`, `APP_NAME`, `VERSION`)
- **`models/predictor.py`** — Singleton predictor class: `load()`, `predict()`, `predict_batch()`, `explain()`
- **`models/schemas.py`** — All Pydantic request/response models
- **`routers/`** — One file per endpoint group (health, info, predict, explain, metrics)
- **`services/tracking.py`** — In-memory request counter, latency histogram
- **`services/validation.py`** — Custom validators (range checks, missing field handling)
- **`utils/logging.py`** — Structured JSON logging with `loguru`
- **`utils/exceptions.py`** — Custom exception classes + FastAPI exception handlers

### 2. Notebook Suite (`notebooks/`)

| Notebook | Content | Outputs |
|----------|---------|---------|
| `01-api-fundamentals` | What is API? REST, HTTP methods, status codes, JSON, request-response lifecycle, diagrams | `outputs/api-lifecycle-diagram.png` |
| `02-modeling-and-benchmarking` | California Housing EDA, 9-model benchmark via LazyPredict/PyCaret/FLAML, SHAP analysis, serialization | `outputs/model-comparison.png`, `models/best_model.joblib`, `outputs/shap_summary.png` |
| `03-fastapi-and-pydantic` | FastAPI creation, routing, path/query params, Pydantic deep-dive, validation, Swagger | Interactive examples |
| `04-api-development` | All 6 endpoints, error handling, middleware, request tracking | Running API server |
| `05-testing-and-benchmarking` | pytest, httpx, latency/throughput benchmarks, cold/warm start, batch vs single | `outputs/latency-benchmark.png` |
| `06-production-readiness` | Logging, config management, model versioning, container prep, security | Production-ready setup |

### 3. API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health + uptime |
| GET | `/model-info` | Model metadata (name, version, features, metrics, trained-at) |
| POST | `/predict` | Single house price prediction |
| POST | `/predict-batch` | Batch predictions (up to 1000) |
| GET | `/metrics` | Request count, latency p50/p95/p99, error rate |
| POST | `/explain` | SHAP-based feature contribution explanation |

### 4. Pydantic Schemas

```python
# Request
class PredictRequest(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

class BatchPredictRequest(BaseModel):
    instances: list[PredictRequest]

class ExplainRequest(PredictRequest):
    pass

# Response
class PredictResponse(BaseModel):
    prediction: float
    prediction_id: str
    model_version: str
    latency_ms: float

class BatchPredictResponse(BaseModel):
    predictions: list[float]
    count: int
    model_version: str
    latency_ms: float

class ExplainResponse(BaseModel):
    prediction: float
    shap_values: dict[str, float]
    base_value: float
    top_features: list[dict]
```

## Data Flow

1. Client sends JSON via HTTP POST
2. FastAPI routes to handler
3. Pydantic validates + coerces types
4. Custom validators check ranges
5. Predictor loads model (lazy singleton)
6. Input scaled (StandardScaler)
7. Model predicts
8. Output inverse-transformed
9. Response packaged + returned
10. Tracking service records metrics

## Model Pipeline

**Models benchmarked:** Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost

**Evaluation metrics:** MAE, MSE, RMSE, R², MAPE

**Tools:** LazyPredict (quick comparison), PyCaret (automated pipeline), FLAML (hyperparameter optimization)

**Best model:** Selected by R² on validation set, serialized with joblib

**Explainability:** SHAP TreeExplainer for feature contributions

## Error Handling

- Missing fields → 422 with field-level detail
- Invalid types → 422 with type error
- Range violations → 422 with custom message
- Empty batch → 422
- Model not loaded → 503
- Internal error → 500 (logged, generic message returned)

## Testing Strategy

- Unit tests: Pydantic schemas, validators, predictor
- Integration tests: httpx against TestClient
- Performance: time.perf_counter benchmarks
- All tests in `pytest` with `conftest.py` fixtures

## Performance Benchmarks

| Scenario | Metric |
|----------|--------|
| Cold start latency | First request time |
| Warm single prediction | p50/p95/p99 latency |
| Batch (10, 100, 1000) | Latency vs throughput |
| Concurrent requests | Degradation curve |

## Deliverables

- `app/` — production-grade FastAPI package
- `notebooks/01` through `06` — complete tutorial suite
- `tests/` — pytest test suite
- `models/` — serialized best model + scaler
- `outputs/` — figures, benchmark CSVs, diagrams
- `pyproject.toml` — dependency management via uv
- `README.md` — mini-book documentation
- `.env.example` — configuration template

## Constraints

- Python 3.12.10, uv, local execution only
- No Docker (prep only)
- All notebooks must run end-to-end
- Every section explains what/why before code
