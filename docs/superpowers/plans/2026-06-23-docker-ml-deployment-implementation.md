# Deploy ML Model with Docker — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained MLOps project with 6 tutorial notebooks, a production-grade FastAPI ML API, Docker containerization, and a portfolio-quality README.

**Architecture:** Self-contained project directory with independent `pyproject.toml`. Six notebooks spiral from Docker fundamentals through MLOps, modeling, FastAPI deployment, Docker containerization, and production monitoring. A `app/` package serves the trained model via REST API, containerized with both naive and optimized Dockerfiles.

**Tech Stack:** Python 3.12, uv, FastAPI, Uvicorn, scikit-learn, XGBoost, LightGBM, CatBoost, LazyPredict, PyCaret, FLAML, SHAP, Docker, Docker Compose, pytest, httpx

## Global Constraints

- Python 3.12.10 only
- Use `uv` for package/venv management
- Self-contained project (no monorepo coupling)
- All notebooks must execute top-to-bottom without errors
- Graceful AutoML fallback: if PyCaret/FLAML import fails, fall back to LazyPredict + manual scikit-learn benchmarks
- SHAP background sample limited to 100 rows
- No hardcoded secrets or credentials
- Portfolio-quality mini-book README

---

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `app/__init__.py`
- Create: `app/main.py`
- Create: `app/schemas.py`
- Create: `app/model.py`
- Create: `app/utils.py`
- Create: `tests/__init__.py`
- Create: `tests/test_api.py`
- Create: `docker/Dockerfile`
- Create: `docker/Dockerfile.optimized`
- Create: `docker-compose.yml`
- Create: `data/.gitkeep`
- Create: `models/.gitkeep`
- Create: `outputs/figures/.gitkeep`
- Create: `outputs/benchmarks/.gitkeep`

**Interfaces:**
- Consumes: (none — first task)
- Produces: Project skeleton with all placeholder files, pyproject deps, and directory structure

- [ ] **Step 1: Create directory structure and pyproject.toml**

```bash
PROJ="/home/ahmad/AI/Github/40 AI-ML Projects for Beginners/MLOps, UI, and Deployment/Deploy a Machine Learning Model with Docker"
mkdir -p "$PROJ/notebooks" "$PROJ/app" "$PROJ/models" "$PROJ/docker" "$PROJ/tests" "$PROJ/outputs/figures" "$PROJ/outputs/benchmarks" "$PROJ/data"
```

Now create `$PROJ/pyproject.toml`:

```toml
[project]
name = "docker-ml-deployment"
version = "1.0.0"
description = "End-to-end MLOps project teaching Docker containerization, FastAPI model serving, and production deployment"
requires-python = "==3.12.10"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "pydantic>=2.10.0",
    "scikit-learn>=1.5.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "joblib>=1.4.0",
    "httpx>=0.28.0",
    "pytest>=8.0.0",
    "pytest-httpx>=0.35.0",
    "shap>=0.45.0",
    "loguru>=0.7.0",
    "matplotlib>=3.9.0",
    "seaborn>=0.13.0",
    "jupyter>=1.1.0",
    "lazypredict>=0.2.0",
]

[project.optional-dependencies]
full = [
    "xgboost>=2.1.0",
    "lightgbm>=4.5.0",
    "catboost>=1.2.0",
    "pycaret>=3.3.0",
    "flaml>=3.2.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create app/main.py (scaffold)**

```python
from fastapi import FastAPI
from app.schemas import HealthResponse

app = FastAPI(title="California Housing ML API", version="1.0.0")

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy")
```

- [ ] **Step 3: Create app/schemas.py (scaffold)**

```python
from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str

class ModelInfoResponse(BaseModel):
    model_name: str
    features: list[str]
    version: str

class PredictionInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

class BatchPredictionInput(BaseModel):
    instances: list[PredictionInput]

class PredictionOutput(BaseModel):
    predicted_value: float

class BatchPredictionOutput(BaseModel):
    predictions: list[float]

class ErrorResponse(BaseModel):
    detail: str

class ExplainRequest(BaseModel):
    input: PredictionInput

class ExplainResponse(BaseModel):
    shap_values: dict[str, float]
    base_value: float
    prediction: float
```

- [ ] **Step 4: Create app/model.py (scaffold)**

```python
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"
_model = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict(features: list[float]) -> float:
    import numpy as np
    model = load_model()
    return float(model.predict([np.array(features)])[0])

def predict_batch(features_batch: list[list[float]]) -> list[float]:
    import numpy as np
    model = load_model()
    return [float(p) for p in model.predict(np.array(features_batch))]
```

- [ ] **Step 5: Create app/utils.py (scaffold)**

```python
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}")
logger.add("outputs/api.log", rotation="10 MB")
```

- [ ] **Step 6: Create test files**

`tests/test_api.py`:
```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

- [ ] **Step 7: Create Docker scaffold files**

`docker/Dockerfile`:
```dockerfile
FROM python:3.12.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`docker/Dockerfile.optimized`:
```dockerfile
FROM python:3.12.10-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12.10-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY app/ app/
COPY models/ models/
EXPOSE 8000
USER nobody
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`docker-compose.yml`:
```yaml
services:
  ml-api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - LOG_LEVEL=INFO
```

- [ ] **Step 8: Create requirements.txt**

```
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pydantic>=2.10.0
scikit-learn>=1.5.0
pandas>=2.2.0
numpy>=1.26.0
joblib>=1.4.0
httpx>=0.28.0
pytest>=8.0.0
shap>=0.45.0
loguru>=0.7.0
matplotlib>=3.9.0
seaborn>=0.13.0
jupyter>=1.1.0
lazypredict>=0.2.0
```

- [ ] **Step 9: Initialize venv and verify**

```bash
cd "$PROJ"
uv venv --python 3.12.10
uv pip install -e ".[full]"
uv run pytest tests/ -v
```

Expected: 1 test passes.

- [ ] **Step 10: Commit**

```bash
cd "$PROJ"
git add -A
git commit -m "feat: project scaffold with pyproject, app package, tests, Dockerfiles"
```

---

### Task 2: Notebook 01 — Docker Fundamentals

**Files:**
- Create: `notebooks/01-docker-fundamentals.ipynb`

**Interfaces:**
- Consumes: (none — conceptual introduction)
- Produces: (none — educational content)

- [ ] **Step 1: Write notebook as JSON**

The notebook covers:
- What is Docker? Why containers exist? Comparison with VMs
- Docker architecture: images, containers, registries, layers, volumes, networks
- Docker CLI tutorial: `pull`, `run`, `ps`, `logs`, `exec`, `stop`, `rm`, `inspect`, `prune`
- Hands-on: Run a simple container, explore layers
- Best practices and common mistakes

Write at `$PROJ/notebooks/01-docker-fundamentals.ipynb` [actual JSON content — this will be a large file with markdown cells, code cells, and outputs]

I'll use the notebook format with:
- Cell 1: Title + intro (markdown)
- Cell 2: What is Docker? (markdown + diagram explanation)
- Cell 3: VMs vs Containers comparison table (markdown)
- Cell 4: Docker architecture overview (markdown)
- Cell 5: Check Docker installation (code — `!docker --version`)
- Cell 6: Docker CLI tutorial with examples (code cells)
- Cell 7: References and next steps (markdown)

- [ ] **Step 2: Execute notebook top-to-bottom**

```bash
cd "$PROJ"
uv run jupyter nbconvert --to notebook --execute notebooks/01-docker-fundamentals.ipynb --output notebooks/01-docker-fundamentals.ipynb --inplace
```

Expected: No errors.

---

### Task 3: Notebook 02 — MLOps Fundamentals

**Files:**
- Create: `notebooks/02-mlops-fundamentals.ipynb`

**Interfaces:**
- Consumes: (none — conceptual)
- Produces: (none — educational)

- [ ] **Step 1: Write notebook**

Covers:
- What is MLOps? Model lifecycle stages
- "Development Machine vs Production Server" differences
- "Works on My Machine" syndrome and dependency drift
- Where Docker fits in the MLOps pipeline
- Diagram: Dev → Package → Deploy → Monitor → Maintain

- [ ] **Step 2: Execute notebook**

```bash
cd "$PROJ"
uv run jupyter nbconvert --to notebook --execute notebooks/02-mlops-fundamentals.ipynb --output notebooks/02-mlops-fundamentals.ipynb --inplace
```

Expected: No errors.

---

### Task 4: Notebook 03 — Modeling and Benchmarking

**Files:**
- Create: `notebooks/03-modeling-and-benchmarking.ipynb`
- Modify: `app/model.py` (update if needed after training)

**Interfaces:**
- Consumes: (none — trains from scratch)
- Produces: `models/best_model.joblib` (serialized model)
- Produces: `outputs/figures/` (model comparison charts)
- Produces: `outputs/benchmarks/model_ranking.csv` (benchmark table)

- [ ] **Step 1: Write the notebook**

Covers:
- Load California Housing dataset from sklearn
- EDA: distributions, correlations, target analysis
- Feature engineering
- Train 9+ models: LinearRegression, Ridge, Lasso, RandomForest, ExtraTrees, XGBoost, LightGBM, CatBoost, GradientBoosting
- AutoML: LazyPredict (primary), PyCaret/FLAML (try/except fallback)
- Metrics: MAE, MSE, RMSE, R², MAPE
- Ranking table with all model scores
- Model comparison bar chart (saved to outputs/figures/)
- Serialization: save best model as models/best_model.joblib
- Save model metadata (feature names, model type, version)

```python
# Key cells content:

# Cell: Load and explore dataset
from sklearn.datasets import fetch_california_housing
import pandas as pd
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["MedHouseVal"] = data.target

# Cell: Train/test split
from sklearn.model_selection import train_test_split
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cell: LazyPredict benchmark
from lazypredict.Supervised import LazyRegressor
lazy = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = lazy.fit(X_train, X_test, y_train, y_test)
models.to_csv("../outputs/benchmarks/lazypredict_ranking.csv")

# Cell: Manual models for cross-validation
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

models_dict = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = []
for name, model in models_dict.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R²": r2_score(y_test, preds),
    })

# Cell: XGBoost (optional)
try:
    import xgboost as xgb
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    preds = xgb_model.predict(X_test)
    results.append({
        "Model": "XGBoost",
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R²": r2_score(y_test, preds),
    })
except ImportError:
    pass

# Same pattern for LightGBM and CatBoost

# Cell: Pick best and save
import joblib
results_df = pd.DataFrame(results).sort_values("R²", ascending=False)
best_model_name = results_df.iloc[0]["Model"]
best_model = models_dict[best_model_name]
joblib.dump(best_model, "../models/best_model.joblib")

# Save metadata
import json
metadata = {
    "model_name": best_model_name,
    "features": list(X.columns),
    "r2_score": float(results_df.iloc[0]["R²"]),
    "n_train_samples": len(X_train),
    "n_test_samples": len(X_test),
}
with open("../models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

- [ ] **Step 2: Execute notebook**

```bash
cd "$PROJ"
uv run jupyter nbconvert --to notebook --execute notebooks/03-modeling-and-benchmarking.ipynb --output notebooks/03-modeling-and-benchmarking.ipynb --inplace
```

Expected: No errors. `models/best_model.joblib` exists. `outputs/benchmarks/` has CSV files.

---

### Task 5: Notebook 04 — FastAPI Deployment

**Files:**
- Create: `notebooks/04-fastapi-deployment.ipynb`
- Modify: `app/main.py` (full implementation)
- Modify: `app/schemas.py` (finalize)
- Modify: `app/model.py` (finalize)
- Modify: `app/utils.py` (finalize)

**Interfaces:**
- Consumes: `models/best_model.joblib` (from Task 4)
- Consumes: `models/model_metadata.json` (from Task 4)
- Produces: Full FastAPI application with all endpoints

- [ ] **Step 1: Update app code to final versions**

`app/main.py`:
```python
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from app.schemas import (
    HealthResponse, ModelInfoResponse,
    PredictionInput, PredictionOutput,
    BatchPredictionInput, BatchPredictionOutput,
    ErrorResponse, ExplainRequest, ExplainResponse,
)
from app.model import load_model, predict, predict_batch
from app.utils import logger
import numpy as np

app = FastAPI(
    title="California Housing ML API",
    description="Production-grade ML API for predicting California housing values. "
                "Trained on the California Housing dataset with ensemble methods.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

METADATA_PATH = Path(__file__).resolve().parent.parent / "models" / "model_metadata.json"

def get_metadata() -> dict:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return {"model_name": "unknown", "features": [], "version": "1.0.0"}

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy")

@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    meta = get_metadata()
    return ModelInfoResponse(
        model_name=meta.get("model_name", "unknown"),
        features=meta.get("features", []),
        version=meta.get("version", "1.0.0"),
    )

@app.post("/predict", response_model=PredictionOutput)
async def predict_endpoint(input_data: PredictionInput):
    try:
        features = [
            input_data.MedInc, input_data.HouseAge, input_data.AveRooms,
            input_data.AveBedrms, input_data.Population, input_data.AveOccup,
            input_data.Latitude, input_data.Longitude,
        ]
        result = predict(features)
        return PredictionOutput(predicted_value=result)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch", response_model=BatchPredictionOutput)
async def predict_batch_endpoint(input_data: BatchPredictionInput):
    try:
        features_batch = [
            [
                inst.MedInc, inst.HouseAge, inst.AveRooms, inst.AveBedrms,
                inst.Population, inst.AveOccup, inst.Latitude, inst.Longitude,
            ]
            for inst in input_data.instances
        ]
        results = predict_batch(features_batch)
        return BatchPredictionOutput(predictions=results)
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

`app/model.py`:
```python
import joblib
import json
import numpy as np
from pathlib import Path
from app.utils import logger

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"
METADATA_PATH = Path(__file__).resolve().parent.parent / "models" / "model_metadata.json"

_model = None

def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run notebook 03 first.")
        _model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    return _model

def predict(features: list[float]) -> float:
    model = load_model()
    arr = np.array(features).reshape(1, -1)
    return float(model.predict(arr)[0])

def predict_batch(features_batch: list[list[float]]) -> list[float]:
    model = load_model()
    arr = np.array(features_batch)
    return [float(p) for p in model.predict(arr)]
```

`app/utils.py`:
```python
from loguru import logger
import sys

logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
    level="INFO",
)
logger.add(
    "outputs/api.log",
    rotation="10 MB",
    retention="1 week",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
)
```

- [ ] **Step 2: Write notebook**

Covers:
- FastAPI introduction, theory, comparison with Flask
- Pydantic validation, request/response models
- Run with Uvicorn in notebook (subprocess)
- Test with httpx
- Swagger/OpenAPI documentation
- Error handling, structured logging

```python
# Cell: Start server in background
import subprocess, time, httpx
proc = subprocess.Popen(
    ["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8001"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)
time.sleep(2)

# Cell: Test health endpoint
response = httpx.get("http://127.0.0.1:8001/health")
assert response.json()["status"] == "healthy"

# Cell: Test predict
input_data = {
    "MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.9841,
    "AveBedrms": 1.0238, "Population": 322.0, "AveOccup": 2.5556,
    "Latitude": 37.88, "Longitude": -122.23,
}
response = httpx.post("http://127.0.0.1:8001/predict", json=input_data)
assert response.status_code == 200
print(f"Prediction: ${response.json()['predicted_value']:.2f}k")

# Cell: Cleanup
proc.terminate()
proc.wait()
```

- [ ] **Step 3: Execute notebook**

```bash
cd "$PROJ"
uv run jupyter nbconvert --to notebook --execute notebooks/04-fastapi-deployment.ipynb --output notebooks/04-fastapi-deployment.ipynb --inplace
```

Expected: No errors.

- [ ] **Step 4: Verify full API tests**

```bash
cd "$PROJ"
uv run pytest tests/ -v
```

Expected: Tests pass.

---

### Task 6: Notebook 05 — Docker Containerization

**Files:**
- Create: `notebooks/05-docker-containerization.ipynb`
- Modify: `docker/Dockerfile` (finalize)
- Modify: `docker/Dockerfile.optimized` (finalize)
- Modify: `docker-compose.yml` (finalize)

**Interfaces:**
- Consumes: `app/` package, `models/`, `docker/` (from previous tasks)
- Produces: Working Docker images, docker-compose setup

- [ ] **Step 1: Write the notebook**

Covers:
- Dockerfile deep dive: FROM, WORKDIR, COPY, RUN, EXPOSE, CMD, ENTRYPOINT
- Layer caching principles
- Naive vs Optimized Dockerfile
- Build and compare image sizes
- Run container, test API
- Docker Compose: multi-service setup
- Container networking, environment variables

```python
# Cell: Build naive image
import subprocess
result = subprocess.run(
    ["docker", "build", "-t", "ml-api-naive", "-f", "docker/Dockerfile", "."],
    capture_output=True, text=True
)
print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

# Cell: Build optimized image
result = subprocess.run(
    ["docker", "build", "-t", "ml-api-optimized", "-f", "docker/Dockerfile.optimized", "."],
    capture_output=True, text=True
)
print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

# Cell: Compare image sizes
result = subprocess.run(
    ["docker", "images", "--format", "{{.Repository}}:{{.Tag}} {{.Size}}"],
    capture_output=True, text=True
)
for line in result.stdout.strip().split("\n"):
    if "ml-api" in line:
        print(line)

# Cell: Run container and test
import time, httpx
container = subprocess.Popen(
    ["docker", "run", "-d", "-p", "8002:8000", "--name", "ml-api-test", "ml-api-optimized"],
    capture_output=True, text=True
)
container_id = container.stdout.strip()
time.sleep(3)
try:
    response = httpx.get("http://localhost:8002/health", timeout=5)
    print(f"Health: {response.json()}")
finally:
    subprocess.run(["docker", "stop", container_id], capture_output=True)
    subprocess.run(["docker", "rm", container_id], capture_output=True)

# Cell: Docker Compose
import yaml
with open("../docker-compose.yml") as f:
    config = yaml.safe_load(f)
print(yaml.dump(config, default_flow_style=False))
```

- [ ] **Step 2: Execute notebook**

```bash
cd "$PROJ"
uv run jupyter nbconvert --to notebook --execute notebooks/05-docker-containerization.ipynb --output notebooks/05-docker-containerization.ipynb --inplace
```

Expected: No errors.

---

### Task 7: Notebook 06 — Production and Monitoring

**Files:**
- Create: `notebooks/06-production-and-monitoring.ipynb`
- Modify: `app/main.py` (add explain endpoint)
- Modify: `app/model.py` (add explain function)
- Modify: `app/schemas.py` (add explain schemas if needed)
- Modify: `tests/test_api.py` (add explain tests)

**Interfaces:**
- Consumes: `models/best_model.joblib` (from Task 4)
- Consumes: Full API code (from Task 5)
- Produces: SHAP explainability endpoint, performance benchmarks, monitoring setup

- [ ] **Step 1: Update app code with SHAP explain endpoint**

`app/model.py` — add:
```python
import shap

_explainer = None
_BACKGROUND_SIZE = 100

def _init_explainer():
    global _explainer
    if _explainer is not None:
        return
    import pandas as pd
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    X_bg = pd.DataFrame(data.data[:100], columns=data.feature_names)
    model = load_model()
    if hasattr(model, "predict") and hasattr(model, "feature_importances_"):
        _explainer = shap.TreeExplainer(model, X_bg, feature_perturbation="interventional")
    else:
        _explainer = shap.KernelExplainer(model.predict, X_bg)

def explain(features: list[float]) -> dict:
    import numpy as np
    _init_explainer()
    model = load_model()
    arr = np.array(features).reshape(1, -1)
    shap_values = _explainer.shap_values(arr)
    import pandas as pd
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    feature_names = data.feature_names
    return {
        "shap_values": dict(zip(feature_names, shap_values[0].tolist())),
        "base_value": float(_explainer.expected_value) if isinstance(_explainer.expected_value, (int, float)) else float(_explainer.expected_value[0]),
        "prediction": float(model.predict(arr)[0]),
    }
```

`app/main.py` — add endpoint:
```python
@app.post("/explain", response_model=ExplainResponse)
async def explain_endpoint(input_data: PredictionInput):
    try:
        features = [
            input_data.MedInc, input_data.HouseAge, input_data.AveRooms,
            input_data.AveBedrms, input_data.Population, input_data.AveOccup,
            input_data.Latitude, input_data.Longitude,
        ]
        result = explain(features)
        return ExplainResponse(**result)
    except Exception as e:
        logger.error(f"Explain failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

`tests/test_api.py` — add:
```python
def test_explain_endpoint():
    response = client.post("/explain", json={
        "MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.9841,
        "AveBedrms": 1.0238, "Population": 322.0, "AveOccup": 2.5556,
        "Latitude": 37.88, "Longitude": -122.23,
    })
    assert response.status_code == 200
    data = response.json()
    assert "shap_values" in data
    assert "base_value" in data
    assert "prediction" in data

def test_predict_batch():
    response = client.post("/predict-batch", json={
        "instances": [
            {"MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.9841, "AveBedrms": 1.0238, "Population": 322.0, "AveOccup": 2.5556, "Latitude": 37.88, "Longitude": -122.23},
            {"MedInc": 2.0, "HouseAge": 30.0, "AveRooms": 4.0, "AveBedrms": 1.0, "Population": 500.0, "AveOccup": 3.0, "Latitude": 34.0, "Longitude": -118.0},
        ]
    })
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2
```

- [ ] **Step 2: Write notebook**

Covers:
- Container testing: smoke tests, integration tests
- Performance benchmarking: startup time, latency, throughput, memory
- Host vs Docker execution comparison (table)
- Security: non-root users, secrets management, vulnerability scanning
- SHAP explainability: feature importance, prediction explanations
- Production checklist
- Summary and next steps

```python
# Cell: Container performance benchmark
import subprocess, time, httpx, json

# Measure host startup
start = time.time()
host_proc = subprocess.Popen(
    ["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8003"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
time.sleep(2)
host_startup = time.time() - start

# Measure host latency
latencies = []
for _ in range(20):
    t0 = time.time()
    httpx.post("http://127.0.0.1:8003/predict", json={
        "MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.9841,
        "AveBedrms": 1.0238, "Population": 322.0, "AveOccup": 2.5556,
        "Latitude": 37.88, "Longitude": -122.23,
    })
    latencies.append((time.time() - t0) * 1000)
host_proc.terminate()

print(f"Host startup: {host_startup:.2f}s")
print(f"Host latency: mean={np.mean(latencies):.1f}ms, p95={np.percentile(latencies, 95):.1f}ms")

# Cell: Container benchmark
container_id = subprocess.run(
    ["docker", "run", "-d", "-p", "8004:8000", "--name", "ml-api-bench", "ml-api-optimized"],
    capture_output=True, text=True
).stdout.strip()
time.sleep(3)

container_latencies = []
for _ in range(20):
    t0 = time.time()
    httpx.post("http://127.0.0.1:8004/predict", json={
        "MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.9841,
        "AveBedrms": 1.0238, "Population": 322.0, "AveOccup": 2.5556,
        "Latitude": 37.88, "Longitude": -122.23,
    })
    container_latencies.append((time.time() - t0) * 1000)

subprocess.run(["docker", "stop", container_id], capture_output=True)
subprocess.run(["docker", "rm", container_id], capture_output=True)

print(f"Container latency: mean={np.mean(container_latencies):.1f}ms, p95={np.percentile(container_latencies, 95):.1f}ms")
```

- [ ] **Step 3: Run tests to verify**

```bash
cd "$PROJ"
uv run pytest tests/ -v
```

Expected: All tests pass (including new explain and batch tests).

- [ ] **Step 4: Execute notebook**

```bash
cd "$PROJ"
uv run jupyter nbconvert --to notebook --execute notebooks/06-production-and-monitoring.ipynb --output notebooks/06-production-and-monitoring.ipynb --inplace
```

Expected: No errors.

---

### Task 8: README — Portfolio-Quality Mini-Book

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: All previous tasks (entire project complete)
- Produces: README.md

- [ ] **Step 1: Write README**

Topics:
- Project overview and learning objectives
- Prerequisites (Docker, Python 3.12.10, uv)
- Project structure diagram
- Quick start guide (clone → venv → install → run notebooks → train → run API → docker)
- Detailed notebook guide (what each one covers)
- API reference (all endpoints with curl examples)
- Docker usage (build, run, compose)
- Performance benchmarks (host vs container)
- SHAP explainability example
- Production checklist
- Tech stack table
- License

- [ ] **Step 2: Verify README renders correctly**

```bash
cd "$PROJ"
# Check for broken internal links
grep -oP '\[.*?\]\(.*?\)' README.md | grep -v 'http' | grep -v 'mailto'
```

Expected: No broken links.

---

### Task 9: Final Integration Verification

**Files:**
- (none — verification only)

- [ ] **Step 1: Run all tests**

```bash
cd "$PROJ"
uv run pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 2: Re-execute all notebooks**

```bash
cd "$PROJ"
for nb in notebooks/0*.ipynb; do
    echo "=== Executing $nb ==="
    uv run jupyter nbconvert --to notebook --execute "$nb" --output "$nb" --inplace
    echo "=== Done ==="
done
```

Expected: All 6 notebooks execute without errors.

- [ ] **Step 3: Docker build and smoke test**

```bash
cd "$PROJ"
docker build -t ml-api-final -f docker/Dockerfile .
docker run -d -p 8005:8000 --name ml-api-final-test ml-api-final
sleep 3
curl -s http://localhost:8005/health
docker stop ml-api-final-test
docker rm ml-api-final-test
```

Expected: Health check returns `{"status":"healthy"}`.

- [ ] **Step 4: Final commit**

```bash
cd "$PROJ"
git add -A
git commit -m "feat: complete docker-ml-deployment project with 6 notebooks, API, Docker, and README"
```
