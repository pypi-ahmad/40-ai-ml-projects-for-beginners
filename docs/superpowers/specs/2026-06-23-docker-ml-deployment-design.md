# Deploy a Machine Learning Model with Docker — Design Document

## Overview

End-to-end MLOps project teaching Docker containerization, FastAPI model serving, and production deployment principles using the California Housing dataset. Self-contained portfolio-quality project with 6 tutorial notebooks and production-grade Dockerized API.

## Project Structure

```
Deploy a Machine Learning Model with Docker/
├── notebooks/
│   ├── 01-docker-fundamentals.ipynb
│   ├── 02-mlops-fundamentals.ipynb
│   ├── 03-modeling-and-benchmarking.ipynb
│   ├── 04-fastapi-deployment.ipynb
│   ├── 05-docker-containerization.ipynb
│   └── 06-production-and-monitoring.ipynb
├── app/
│   ├── main.py           # FastAPI application
│   ├── schemas.py        # Pydantic models
│   ├── model.py          # Model loading/prediction
│   └── utils.py          # Helpers, logging, monitoring
├── models/               # Serialized .joblib files
├── docker/
│   ├── Dockerfile
│   └── Dockerfile.optimized
├── docker-compose.yml
├── tests/
│   └── test_api.py
├── outputs/
│   ├── figures/
│   └── benchmarks/
├── data/                 # California Housing dataset
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Notebook Design (6 notebooks, ~2h total read time)

### 01-docker-fundamentals.ipynb
- What is Docker? Why containers exist?
- VMs vs Containers (diagram)
- Docker architecture: images, containers, registries, layers, volumes, networks
- Docker CLI tutorial: `pull`, `run`, `ps`, `logs`, `exec`, `stop`, `rm`, `inspect`, `prune`
- Hands-on: Run a simple container, explore layers
- Best practices and common mistakes

### 02-mlops-fundamentals.ipynb
- What is MLOps? Model lifecycle
- Development Machine vs Production Server
- "Works on My Machine" syndrome
- Dependency management, environment drift
- Where Docker fits in MLOps pipeline
- Diagram: Dev → Package → Deploy → Monitor → Maintain

### 03-modeling-and-benchmarking.ipynb
- California Housing dataset: EDA, feature engineering
- Train: Linear Regression, Ridge, Lasso, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, Gradient Boosting
- AutoML: LazyPredict, PyCaret, FLAML (with graceful fallback)
- Metrics: MAE, MSE, RMSE, R², MAPE
- Ranking tables, model comparison charts
- Model serialization: Pickle vs Joblib vs ONNX tradeoffs
- Save best model as `.joblib`

### 04-fastapi-deployment.ipynb
- FastAPI introduction, theory, comparison with Flask
- Pydantic validation, request/response models
- Endpoints: GET `/health`, GET `/model-info`, POST `/predict`, POST `/predict-batch`, GET `/metrics`, POST `/explain`
- Run with Uvicorn, test with httpx
- Swagger/OpenAPI documentation
- Error handling, structured logging

### 05-docker-containerization.ipynb
- Dockerfile deep dive: FROM, WORKDIR, COPY, RUN, EXPOSE, CMD, ENTRYPOINT
- Layer caching, image optimization
- Multi-stage builds
- Naive vs Optimized Dockerfile comparison
- Build, run, test container
- Docker Compose introduction
- Container networking, environment variables

### 06-production-and-monitoring.ipynb
- Container testing: smoke tests, integration tests
- Performance benchmarking: startup time, latency, throughput, memory
- Host vs Docker execution comparison
- Security: non-root users, secrets, vulnerability scanning
- SHAP explainability: feature importance, prediction explanations
- Production checklist
- Summary and next steps

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Python | 3.12.10 | Required |
| Package mgmt | uv | Required |
| Web framework | FastAPI | Required |
| ASGI server | Uvicorn | Required |
| Validation | Pydantic | Required |
| ML models | scikit-learn, XGBoost, LightGBM, CatBoost | Benchmark breadth |
| AutoML | LazyPredict, PyCaret, FLAML | Educational comparison |
| Serialization | joblib | Best for sklearn |
| Explainability | SHAP | Industry standard |
| Container | Docker | Required |
| Orchestration | Docker Compose | Required |
| Testing | pytest + httpx | FastAPI testing |

## Key Design Decisions

1. **Self-contained project** — independent `pyproject.toml`, no monorepo coupling, portable for portfolio
2. **Split notebooks** — each covers one concept layer, can be studied independently
3. **Graceful AutoML fallback** — if PyCaret/FLAML conflict, fall back to LazyPredict + manual benchmarks
4. **Production-grade API** — validation, logging, error handling, health checks, metrics
5. **Docker optimization comparison** — naive vs multi-stage, measurable size/latency differences
6. **SHAP in API** — `/explain` endpoint for real prediction explanations

## Verification Criteria

- All 6 notebooks execute top-to-bottom without errors
- `uv run pytest tests/` passes
- `docker build -t ml-api .` succeeds
- `docker run -p 8000:8000 ml-api` serves healthy API
- All endpoints return correct responses
- Benchmark tables contain real metrics
- Diagrams/figures are generated and saved
- README is comprehensive (mini-book style)
