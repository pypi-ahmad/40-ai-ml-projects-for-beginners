# ML REST API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build production-grade ML REST API teaching API fundamentals, model serving, FastAPI, Pydantic, testing, and production readiness. 6 notebooks + Python package + tests.

**Architecture:** FastAPI app factory with modular routers, Pydantic v2 schemas, lazy-loaded model singleton, in-memory request tracking, structured logging via loguru.

**Tech Stack:** Python 3.12.10, uv, FastAPI, Uvicorn, Pydantic v2, scikit-learn, LightGBM/XGBoost/CatBoost, SHAP, loguru, pytest, httpx

## Global Constraints

- Python 3.12.10 exactly (`.python-version`)
- uv only for package/venv management (no pip, no conda)
- Virtual environment inside project dir (`.venv/`)
- No Docker — prep only
- All notebooks executable end-to-end without manual intervention
- California Housing dataset from `sklearn.datasets`
- Joblib for model serialization
- All code must be importable as `pip install -e .`
- Type hints throughout Python code
- Structured logging with `loguru`, no `print`

---

### Task 1: Scaffold Project

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.env.example`
- Create: `app/__init__.py`
- Create: `app/main.py`
- Create: `app/config.py`
- Create: `app/models/__init__.py`
- Create: `app/models/schemas.py`
- Create: `app/models/predictor.py`
- Create: `app/routers/__init__.py`
- Create: `app/routers/health.py`
- Create: `app/routers/info.py`
- Create: `app/routers/predict.py`
- Create: `app/routers/explain.py`
- Create: `app/routers/metrics.py`
- Create: `app/services/__init__.py`
- Create: `app/services/tracking.py`
- Create: `app/services/validation.py`
- Create: `app/utils/__init__.py`
- Create: `app/utils/logging.py`
- Create: `app/utils/exceptions.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_api.py`
- Create: `models/.gitkeep`
- Create: `notebooks/.gitkeep`
- Create: `outputs/.gitkeep`

**Interfaces:**
- Consumes: Nothing (foundational)
- Produces: Complete directory structure, pyproject.toml with all deps, skeleton modules

- [ ] **Step 1: Create directories**

```bash
PROJ="/home/ahmad/AI/Github/40 AI-ML Projects for Beginners/MLOps, UI, and Deployment/Deploy Your First ML Model as a REST API"
mkdir -p "$PROJ"/{app/{models,routers,services,utils},tests,models,notebooks,outputs}
touch "$PROJ"/models/.gitkeep "$PROJ"/notebooks/.gitkeep "$PROJ"/outputs/.gitkeep
```

- [ ] **Step 2: Write pyproject.toml**

```toml
[project]
name = "ml-api-deployment"
version = "0.1.0"
description = "Deploy Your First ML Model as a REST API"
requires-python = "==3.12.10"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "scikit-learn>=1.5.0",
    "joblib>=1.4.0",
    "numpy>=1.26.0",
    "pandas>=2.2.0",
    "loguru>=0.7.0",
    "shap>=0.45.0",
    "python-multipart>=0.0.9",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "httpx>=0.27.0",
    "lazypredict>=0.2.0",
    "flaml>=3.0.0",
    "pycaret>=3.3.0",
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
    "catboost>=1.2.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "jupyter>=1.0.0",
    "nbformat>=5.9.0",
    "nbconvert>=7.16.0",
]

[build-system]
requires = ["setuptools>=69.0.0"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 3: Write .python-version**

```
3.12.10
```

- [ ] **Step 4: Write .env.example**

```
MODEL_PATH=models/best_model.joblib
SCALER_PATH=models/scaler.joblib
LOG_LEVEL=INFO
APP_NAME=California Housing API
VERSION=0.1.0
DEBUG=false
```

- [ ] **Step 5: Write app/__init__.py**

```python
from importlib.metadata import version

__version__ = version("ml-api-deployment")
```

- [ ] **Step 6: Write app/config.py**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_path: str = "models/best_model.joblib"
    scaler_path: str = "models/scaler.joblib"
    log_level: str = "INFO"
    app_name: str = "California Housing API"
    version: str = "0.1.0"
    debug: bool = False

    model_config = {"env_file": ".env", "env_prefix": ""}


settings = Settings()
```

- [ ] **Step 7: Write app/utils/logging.py**

```python
import sys
from loguru import logger
from app.config import settings


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {name}:{function}:{line} | {message}",
        colorize=True,
    )
    logger.add(
        "logs/api.log",
        level="DEBUG",
        rotation="10 MB",
        retention="1 month",
        format="{time} | {level} | {name} | {message}",
    )
```

- [ ] **Step 8: Write app/utils/exceptions.py**

```python
from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger


class ModelNotLoadedError(Exception):
    pass


class PredictionError(Exception):
    pass


async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    return JSONResponse(
        status_code=503,
        content={"detail": "Model not loaded. Please try again shortly."},
    )


async def prediction_error_handler(request: Request, exc: PredictionError):
    logger.error(f"Prediction failed: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Prediction failed. Check server logs."},
    )
```

- [ ] **Step 9: Write app/models/schemas.py**

```python
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    MedInc: float = Field(..., ge=0, description="Median income in block group")
    HouseAge: float = Field(..., ge=0, description="Median house age")
    AveRooms: float = Field(..., ge=0, description="Average rooms per household")
    AveBedrms: float = Field(..., ge=0, description="Average bedrooms per household")
    Population: float = Field(..., ge=0, description="Block group population")
    AveOccup: float = Field(..., ge=0, description="Average occupancy")
    Latitude: float = Field(..., ge=32, le=42, description="Latitude")
    Longitude: float = Field(..., ge=-125, le=-114, description="Longitude")

    @field_validator("AveRooms")
    @classmethod
    def rooms_greater_than_bedrooms(cls, v, info):
        if "AveBedrms" in info.data and v <= info.data["AveBedrms"]:
            raise ValueError("AveRooms must exceed AveBedrms")
        return v


class BatchPredictRequest(BaseModel):
    instances: list[PredictRequest] = Field(..., min_length=1, max_length=1000)


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


class ExplainRequest(PredictRequest):
    pass


class ExplainResponse(BaseModel):
    prediction: float
    shap_values: dict[str, float]
    base_value: float
    top_features: list[dict]


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    features: list[str]
    metrics: dict[str, float]
    trained_at: str | None = None
    model_type: str


class MetricsResponse(BaseModel):
    total_requests: int
    total_errors: int
    latency_ms: dict[str, float]
    error_rate: float
```

- [ ] **Step 10: Write app/models/predictor.py**

```python
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import shap
from loguru import logger

from app.config import settings
from app.utils.exceptions import ModelNotLoadedError, PredictionError
from app.models.schemas import PredictRequest


class Predictor:
    _instance: "Predictor | None" = None

    def __new__(cls) -> "Predictor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._scaler = None
            cls._instance._loaded = False
            cls._instance._load_time = 0.0
            cls._instance._explainer = None
        return cls._instance

    def load(self) -> None:
        model_path = Path(settings.model_path)
        scaler_path = Path(settings.scaler_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        self._model = joblib.load(model_path)
        self._scaler = joblib.load(scaler_path)
        self._loaded = True
        self._load_time = time.time()
        logger.info(f"Model loaded from {model_path}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model(self):
        if not self._loaded:
            self.load()
        return self._model

    @property
    def scaler(self):
        if not self._loaded:
            self.load()
        return self._scaler

    def predict(self, features: PredictRequest) -> float:
        if not self._loaded:
            raise ModelNotLoadedError("Model not loaded")
        try:
            raw = np.array([[
                features.MedInc, features.HouseAge, features.AveRooms,
                features.AveBedrms, features.Population, features.AveOccup,
                features.Latitude, features.Longitude,
            ]])
            scaled = self._scaler.transform(raw)
            pred = self._model.predict(scaled)[0]
            return float(pred)
        except Exception as e:
            raise PredictionError(str(e))

    def predict_batch(self, instances: list[PredictRequest]) -> list[float]:
        if not self._loaded:
            raise ModelNotLoadedError("Model not loaded")
        try:
            rows = [[
                f.MedInc, f.HouseAge, f.AveRooms, f.AveBedrms,
                f.Population, f.AveOccup, f.Latitude, f.Longitude,
            ] for f in instances]
            raw = np.array(rows)
            scaled = self._scaler.transform(raw)
            preds = self._model.predict(scaled)
            return [float(p) for p in preds]
        except Exception as e:
            raise PredictionError(str(e))

    def explain(self, features: PredictRequest) -> dict:
        if not self._loaded:
            raise ModelNotLoadedError("Model not loaded")
        try:
            raw = np.array([[
                features.MedInc, features.HouseAge, features.AveRooms,
                features.AveBedrms, features.Population, features.AveOccup,
                features.Latitude, features.Longitude,
            ]])
            scaled = self._scaler.transform(raw)
            pred = float(self._model.predict(scaled)[0])
            if self._explainer is None:
                self._explainer = shap.TreeExplainer(self._model)
            shap_values = self._explainer.shap_values(scaled)
            feature_names = [
                "MedInc", "HouseAge", "AveRooms", "AveBedrms",
                "Population", "AveOccup", "Latitude", "Longitude",
            ]
            sv_dict = dict(zip(feature_names, map(float, shap_values[0])))
            base = float(self._explainer.expected_value)
            top = sorted(sv_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            return {
                "prediction": pred,
                "shap_values": sv_dict,
                "base_value": base,
                "top_features": [
                    {"feature": k, "impact": round(v, 4)}
                    for k, v in top
                ],
            }
        except Exception as e:
            raise PredictionError(str(e))

    def get_model_info(self) -> dict:
        if not self._loaded:
            raise ModelNotLoadedError("Model not loaded")
        return {
            "name": type(self._model).__name__,
            "version": settings.version,
            "features": [
                "MedInc", "HouseAge", "AveRooms", "AveBedrms",
                "Population", "AveOccup", "Latitude", "Longitude",
            ],
            "metrics": getattr(self._model, "best_score", {}) or {},
            "model_type": type(self._model).__module__,
        }
```

- [ ] **Step 11: Write app/services/tracking.py**

```python
import time
from collections import defaultdict
from threading import Lock


class RequestTracker:
    def __init__(self):
        self._lock = Lock()
        self._total_requests = 0
        self._total_errors = 0
        self._latencies: list[float] = []

    def record_request(self, latency_ms: float) -> None:
        with self._lock:
            self._total_requests += 1
            self._latencies.append(latency_ms)

    def record_error(self) -> None:
        with self._lock:
            self._total_errors += 1

    def get_metrics(self) -> dict:
        with self._lock:
            total = self._total_requests
            errors = self._total_errors
            lats = sorted(self._latencies)
        n = len(lats)
        p50 = lats[int(n * 0.5)] if n else 0.0
        p95 = lats[int(n * 0.95)] if n else 0.0
        p99 = lats[int(n * 0.99)] if n else 0.0
        return {
            "total_requests": total,
            "total_errors": errors,
            "latency_ms": {"p50": p50, "p95": p95, "p99": p99},
            "error_rate": errors / total if total else 0.0,
        }


tracker = RequestTracker()
```

- [ ] **Step 12: Write app/services/validation.py**

```python
from app.models.schemas import PredictRequest


def validate_features(features: PredictRequest) -> list[str]:
    warnings: list[str] = []
    if features.AveOccup > 100:
        warnings.append("AveOccup unusually high (>100)")
    if features.Population > 50_000:
        warnings.append("Population unusually high (>50,000)")
    if features.HouseAge > 100:
        warnings.append("HouseAge unusually high (>100)")
    return warnings
```

- [ ] **Step 13: Write app/routers/health.py**

```python
import time
from fastapi import APIRouter

from app.models.predictor import Predictor
from app.models.schemas import HealthResponse

router = APIRouter(tags=["Health"])
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
def health_check():
    predictor = Predictor()
    return HealthResponse(
        status="healthy",
        uptime_seconds=time.time() - _start_time,
        model_loaded=predictor.is_loaded,
    )
```

- [ ] **Step 14: Write app/routers/info.py**

```python
from fastapi import APIRouter, HTTPException

from app.models.predictor import Predictor
from app.models.schemas import ModelInfoResponse
from app.utils.exceptions import ModelNotLoadedError

router = APIRouter(tags=["Model Info"])


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    try:
        predictor = Predictor()
        info = predictor.get_model_info()
        return ModelInfoResponse(**info)
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=503, detail=str(e))
```

- [ ] **Step 15: Write app/routers/predict.py**

```python
import time
import uuid
from fastapi import APIRouter, HTTPException

from app.models.predictor import Predictor
from app.models.schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
)
from app.services.tracking import tracker
from app.utils.exceptions import ModelNotLoadedError, PredictionError

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictResponse)
def predict(features: PredictRequest):
    start = time.perf_counter()
    try:
        predictor = Predictor()
        pred = predictor.predict(features)
        latency = (time.perf_counter() - start) * 1000
        tracker.record_request(latency)
        return PredictResponse(
            prediction=pred,
            prediction_id=str(uuid.uuid4()),
            model_version="0.1.0",
            latency_ms=round(latency, 2),
        )
    except (ModelNotLoadedError, PredictionError) as e:
        tracker.record_error()
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/predict-batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    start = time.perf_counter()
    try:
        predictor = Predictor()
        preds = predictor.predict_batch(request.instances)
        latency = (time.perf_counter() - start) * 1000
        tracker.record_request(latency)
        return BatchPredictResponse(
            predictions=preds,
            count=len(preds),
            model_version="0.1.0",
            latency_ms=round(latency, 2),
        )
    except (ModelNotLoadedError, PredictionError) as e:
        tracker.record_error()
        raise HTTPException(status_code=503, detail=str(e))
```

- [ ] **Step 16: Write app/routers/explain.py**

```python
import time
from fastapi import APIRouter, HTTPException

from app.models.predictor import Predictor
from app.models.schemas import ExplainRequest, ExplainResponse
from app.utils.exceptions import ModelNotLoadedError, PredictionError

router = APIRouter(tags=["Explainability"])


@router.post("/explain", response_model=ExplainResponse)
def explain(features: ExplainRequest):
    try:
        predictor = Predictor()
        result = predictor.explain(features)
        return ExplainResponse(**result)
    except (ModelNotLoadedError, PredictionError) as e:
        raise HTTPException(status_code=503, detail=str(e))
```

- [ ] **Step 17: Write app/routers/metrics.py**

```python
from fastapi import APIRouter

from app.models.schemas import MetricsResponse
from app.services.tracking import tracker

router = APIRouter(tags=["Metrics"])


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    m = tracker.get_metrics()
    return MetricsResponse(**m)
```

- [ ] **Step 18: Write app/main.py**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import settings
from app.models.predictor import Predictor
from app.routers import health, info, predict, explain, metrics
from app.utils.exceptions import (
    ModelNotLoadedError, PredictionError,
    model_not_loaded_handler, prediction_error_handler,
)
from app.utils.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    predictor = Predictor()
    try:
        predictor.load()
        logger.info("Model loaded on startup")
    except FileNotFoundError:
        logger.warning("No model found at startup. Load model before predicting.")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan,
)

app.add_exception_handler(ModelNotLoadedError, model_not_loaded_handler)
app.add_exception_handler(PredictionError, prediction_error_handler)

app.include_router(health.router)
app.include_router(info.router)
app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(metrics.router)


@app.get("/")
def root():
    return {
        "app": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "openapi": "/openapi.json",
    }
```

- [ ] **Step 19: Write tests/conftest.py**

```python
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_features():
    return {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984,
        "AveBedrms": 1.024,
        "Population": 322.0,
        "AveOccup": 2.556,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }
```

- [ ] **Step 20: Write tests/test_api.py**

```python
from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "app" in data
    assert "version" in data


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_model_info_endpoint_no_model(client: TestClient):
    response = client.get("/model-info")
    assert response.status_code == 503


def test_predict_invalid_data(client: TestClient):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_negative_values(client: TestClient):
    data = {
        "MedInc": 8.3252,
        "HouseAge": -5,
        "AveRooms": 6.984,
        "AveBedrms": 1.024,
        "Population": 322.0,
        "AveOccup": 2.556,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_metrics_endpoint(client: TestClient):
    response = client.get("/metrics")
    assert response.status_code == 200


def test_explain_invalid(client: TestClient):
    response = client.post("/explain", json={})
    assert response.status_code == 422


def test_batch_predict_empty(client: TestClient):
    response = client.post("/predict-batch", json={"instances": []})
    assert response.status_code == 422
```

- [ ] **Step 21: Create setup.py (editable install support)**

```python
from setuptools import setup

setup()
```

- [ ] **Step 22: Create MANIFEST.in**

```
include .env.example
recursive-include models *.joblib
```

- [ ] **Step 23: Create logs/.gitkeep**

```bash
mkdir -p "/home/ahmad/AI/Github/40 AI-ML Projects for Beginners/MLOps, UI, and Deployment/Deploy Your First ML Model as a REST API/logs"
touch "/home/ahmad/AI/Github/40 AI-ML Projects for Beginners/MLOps, UI, and Deployment/Deploy Your First ML Model as a REST API/logs/.gitkeep"
```

- [ ] **Step 24: Setup uv environment and install**

```bash
PROJ="/home/ahmad/AI/Github/40 AI-ML Projects for Beginners/MLOps, UI, and Deployment/Deploy Your First ML Model as a REST API"
cd "$PROJ"
uv python pin 3.12.10
uv venv --python 3.12.10
source .venv/bin/activate
uv pip install -e ".[dev]"
```

- [ ] **Step 25: Run tests to verify package**

```bash
PROJ="/home/ahmad/AI/Github/40 AI-ML Projects for Beginners/MLOps, UI, and Deployment/Deploy Your First ML Model as a REST API"
cd "$PROJ"
source .venv/bin/activate
python -c "from app.main import app; print('App imported OK')"
pytest tests/ -v --tb=short 2>&1 | head -40
```

Expected: All 7 tests pass or at least import succeeds.

---

### Task 2: Notebook 01 — API Fundamentals

**Files:**
- Create: `notebooks/01-api-fundamentals.ipynb`
- Create: `outputs/api-lifecycle-diagram.png` (text-based placeholder)

**Interfaces:**
- Consumes: Nothing (standalone educational notebook)
- Produces: Interactive API fundamentals tutorial

This notebook covers:
1. What is an API? (physical world analogy — restaurant waiter)
2. REST principles (resources, verbs, statelessness)
3. HTTP methods (GET, POST, PUT, DELETE, PATCH)
4. Status codes (1xx, 2xx, 3xx, 4xx, 5xx) with real examples
5. JSON format (serialization, types, nesting)
6. Request-response lifecycle (diagram)
7. cURL examples against real public APIs
8. Python `requests` library demo
9. What makes a good API? (consistency, versioning, documentation)

Each section: explanation → diagram/text → code example with output.

- [ ] **Step 1: Write full notebook content**

```bash
# Create notebook using the provided 01-api-fundamentals.ipynb content
PROJ="/home/ahmad/AI/Github/40 AI-ML Projects for Beginners/MLOps, UI, and Deployment/Deploy Your First ML Model as a REST API"
# Content will be written inline with full educational material
```

The notebook will be written with markdown cells containing diagrams in mermaid-compatible format (for Jupyter extensions or visual display), code cells demonstrating HTTP with `requests` and `http.client`, and rich explanations.

- [ ] **Step 2: Verify notebook loads**

```bash
cd "$PROJ"
source .venv/bin/activate
python -c "import nbformat; nbformat.load('notebooks/01-api-fundamentals.ipynb'); print('Valid notebook')"
```

---

### Task 3: Notebook 02 — Modeling and Benchmarking

**Files:**
- Create: `notebooks/02-modeling-and-benchmarking.ipynb` (if not yet created)
- Create: \`outputs/model-comparison.png\`
- Create: \`outputs/shap_summary.png\`
- Create: \`models/best_model.joblib\`
- Create: \`models/scaler.joblib\`

**Interfaces:**
- Consumes: California Housing from sklearn
- Produces: Serialized best model + scaler (\`models/best_model.joblib\`, \`models/scaler.joblib\`), benchmark figures

Covers:
1. California Housing dataset EDA (distributions, correlations, missing values)
2. Feature engineering (interactions, polynomial features test)
3. Train/test split (80/20, stratified on binned target)
4. LazyPredict full benchmark (all 40+ regressors)
5. Top model refinement (Random Forest, XGBoost, LightGBM, CatBoost)
6. PyCaret automated pipeline (compare_models, tune, finalize)
7. FLAML hyperparameter optimization
8. Best model selection (min RMSE wins)
9. SHAP analysis (summary plot, dependence, force)
10. Model serialization with joblib
11. StandardScaler fitting and serialization

- [ ] **Step 1: Write notebook**

- [ ] **Step 2: Execute notebook to produce model artifacts**

```bash
cd "$PROJ"
source .venv/bin/activate
jupyter nbconvert --to notebook --execute notebooks/02-modeling-and-benchmarking.ipynb --output 02-modeling-and-benchmarking-executed.ipynb --ExecutePreprocessor.timeout=600
```

- [ ] **Step 3: Verify artifacts exist**

```bash
ls -la "$PROJ"/models/best_model.joblib "$PROJ"/models/scaler.joblib
```

---

### Task 4: Notebook 03 — FastAPI and Pydantic Deep Dive

**Files:**
- Create: \`notebooks/03-fastapi-and-pydantic.ipynb\`

**Interfaces:**
- Consumes: FastAPI, Pydantic v2
- Produces: Interactive FastAPI/Pydantic tutorial

1. What is FastAPI? (ASGI, Starlette, Pydantic foundation)
2. FastAPI vs Flask vs Django REST
3. ASGI vs WSGI explained
4. FastAPI application (create app, run with uvicorn)
5. Path parameters, query parameters
6. Pydantic v2 deep dive (BaseModel, Field, validators, model_config)
7. Request body parsing (automatic from JSON)
8. Response models (response_model, response_model_exclude_none)
9. Swagger UI and ReDoc (auto-generated docs)
10. Error handling (HTTPException, custom handlers, validation errors)

- [ ] **Step 1: Write notebook**

---

### Task 5: Notebook 04 — Full API Development

**Files:**
- Create: \`notebooks/04-api-development.ipynb\`

**Interfaces:**
- Consumes: \`app/\` package, \`models/best_model.joblib\`, \`models/scaler.joblib\`
- Produces: Running API demonstration with real requests

1. App factory pattern and lifespan events
2. All 6 endpoints built and tested inline
3. Model singleton pattern explanation
4. Middleware (CORS, request ID, timing)
5. Request validation in action (show 422 errors)
6. Live API calls from notebook (httpx against running server)
7. Error responses demonstration
8. Swagger walkthrough

- [ ] **Step 1: Write notebook**

---

### Task 6: Notebook 05 — Testing and Benchmarking

**Files:**
- Create: \`notebooks/05-testing-and-benchmarking.ipynb\`
- Create: \`outputs/latency-benchmark.png\`

**Interfaces:**
- Consumes: \`app/\` package, running API from notebook
- Produces: Performance benchmark figures and test results

1. HTTP testing with \`httpx\` (async and sync)
2. FastAPI \`TestClient\` for integration tests
3. Unit vs integration vs e2e tests
4. Cold start latency measurement
5. Warm latency benchmark (p50/p95/p99 over 1000 requests)
6. Batch size vs latency/throughput (size 1, 10, 100, 1000)
7. Concurrent request benchmark
8. Error rate under load
9. Comparison: single vs batch endpoint

- [ ] **Step 1: Write notebook**

---

### Task 7: Notebook 06 — Production Readiness

**Files:**
- Create: \`notebooks/06-production-readiness.ipynb\`
- Create: \`Dockerfile\` (prep only — not tested)

**Interfaces:**
- Consumes: \`app/\` package
- Produces: Production-readiness documentation, Dockerfile, security checklist

1. Structured logging (loguru in action)
2. Configuration management (pydantic-settings, env files, .env.example)
3. Error handling strategy (structured errors, stack traces hidden)
4. Model versioning strategy (semver for models)
5. Basic auth / API key concept
6. Rate limiting (slowapi concept)
7. CORS configuration
8. Containerization prep (Dockerfile, .dockerignore)
9. Production checklist (authentication, monitoring, CI/CD, scaling)
10. Load testing concept (locust)

- [ ] **Step 1: Write notebook**

- [ ] **Step 2: Write Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install -e "."
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Write .dockerignore**

```
.venv/
.env
*.joblib
notebooks/
outputs/
logs/
.git/
```

---

### Task 8: README and Final Polish

**Files:**
- Create: \`README.md\`

**Interfaces:**
- Consumes: All previous tasks
- Produces: Complete project README

Covers:
1. Project overview and learning objectives
2. Prerequisites and setup
3. Project structure explanation
4. Running the API
5. API documentation (endpoints table)
6. Notebook guide (what each notebook covers)
7. Model performance summary
8. Production deployment notes
9. Next steps and related projects

- [ ] **Step 1: Write README.md**

---

### Task 9: Verify Everything End to End

**Files:**
- Modify: None (verification only)

- [ ] **Step 1: Run full test suite**

```bash
cd "$PROJ"
source .venv/bin/activate
pytest tests/ -v
```

- [ ] **Step 2: Verify API starts**

```bash
cd "$PROJ"
source .venv/bin/activate
timeout 5 python -c "import uvicorn; uvicorn.run('app.main:app', host='127.0.0.1', port=8000)" 2>&1 || true
```

- [ ] **Step 3: Verify notebooks are valid**

```bash
cd "$PROJ"
for nb in notebooks/*.ipynb; do
  python -c "import nbformat; nbformat.load('$nb'); print('OK: $nb')"
done
```

- [ ] **Step 4: Verify imports**

```bash
cd "$PROJ"
source .venv/bin/activate
python -c "
from app.main import app
from app.config import settings
from app.models.schemas import PredictRequest, PredictResponse
from app.models.predictor import Predictor
from app.services.tracking import tracker
print('All imports OK')
"
```
