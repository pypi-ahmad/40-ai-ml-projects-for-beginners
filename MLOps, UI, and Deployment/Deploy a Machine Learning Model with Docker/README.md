# Project #10: Deploy a Machine Learning Model with Docker

Production-grade MLOps tutorial project for packaging, containerizing, deploying, and operating an ML model consistently across environments.

## Why This Project Exists

A trained model is not useful until it runs reliably in production.

Typical failure path:

1. Model works on notebook machine.
2. Deployment machine has different Python/dependency versions.
3. Serialized model breaks or API behavior drifts.
4. Team sees: **"works on my machine"**.

This project teaches how to prevent that using Docker + FastAPI + MLOps discipline.

## What You Learn (Zero to Hero)

- Docker fundamentals: images, containers, layers, volumes, networks, registries.
- MLOps lifecycle: development, packaging, deployment, monitoring, maintenance.
- Model benchmarking on California Housing with manual + AutoML workflows.
- Production API design with FastAPI, strict validation, metrics, and rate limiting.
- Containerized deployment with Docker Compose networking.
- Monitoring stack with Prometheus + Grafana.
- Security and production-readiness checklists.

## Tooling Rationale (Strengths, Weaknesses, Tradeoffs)

| Tool | Why Used | Strength | Weakness | Tradeoff |
|---|---|---|---|---|
| FastAPI | Model serving layer | Very fast API development + validation | Async/data validation adds learning curve | Better correctness/maintainability for slight complexity |
| Uvicorn | ASGI server | Lightweight, production-ready | Needs deployment config discipline | Simpler operations than heavier stacks |
| Pydantic | Input/output contracts | Strict schema validation | Validation rules can feel verbose | Safer inference APIs |
| scikit-learn | Baselines + classical models | Stable APIs, broad adoption | Some models weaker on nonlinear data | Excellent teaching and baseline value |
| Joblib | Model persistence | Fast for numpy/sklearn artifacts | Python ecosystem coupling | Great for sklearn-style serving |
| LazyPredict | Quick baseline sweep | Fast broad comparison | Limited tuning control | High signal early, then deeper methods |
| FLAML | Efficient AutoML | Budget-aware optimization | Extra dependency + search complexity | Better performance with controlled runtime |
| PyCaret | Low-code model workflows | Rapid experimentation | Heavy dependency graph | Great for education and fast iterations |
| Docker | Runtime consistency | Eliminates environment drift | Requires Docker literacy | Reliable cross-machine execution |
| Docker Compose | Service orchestration | Easy multi-service local stack | Not full orchestrator like Kubernetes | Best stepping stone for beginners |

PyCaret compatibility note: PyCaret currently raises a runtime guard on Python `3.12.10`, so this project keeps PyCaret as an optional benchmark path with graceful fallback logging while still running fully end-to-end on the required Python version.

## Architecture

```text
Client (curl/httpx/Postman)
        |
        v
Docker Network (mlops-net)
        |
        +--> ml-api (FastAPI + model + SHAP + metrics)
        |        |
        |        +--> /predict /predict-batch /explain /metrics
        |
        +--> prometheus (scrapes ml-api /metrics)
        |
        +--> grafana (dashboards over Prometheus)
```

## Repository Layout

```text
.
├── app/                         # FastAPI app, schema contracts, model loading
├── pipeline/                    # Reusable training/benchmarking modules
├── scripts/                     # End-to-end automation scripts
├── docker/                      # Naive + optimized Dockerfiles
├── monitoring/                  # Prometheus + Grafana provisioning
├── notebooks/                   # 8-part tutorial mini-book
├── models/                      # Serialized model + metadata + ONNX demo artifact
├── outputs/                     # Benchmarks, figures, logs
├── tests/                       # API contract and integration-style tests
├── docker-compose.yml
└── pyproject.toml
```

## Environment Setup (uv + Python 3.12.10)

```bash
uv venv .venv
source .venv/bin/activate
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

## End-to-End Execution

### 1) Train + benchmark + export artifacts

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train_pipeline.py --profile deep
```

Profiles:

- `fast`: quick smoke checks
- `balanced`: practical default
- `deep`: portfolio-grade benchmark depth (long runtime)

### 2) Run tests

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

### 3) Run API locally

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4) Run Dockerized stack

```bash
docker compose up -d --build
curl http://127.0.0.1:8001/health
docker compose down
```

### 5) One-command automation

```bash
bash scripts/full_project_run.sh deep
```

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness status |
| GET | `/model-info` | Serving model, metrics, training metadata |
| POST | `/predict` | Single inference |
| POST | `/predict-batch` | Canonical batch inference endpoint |
| POST | `/batch-predict` | Backward-compatible alias |
| POST | `/explain` | SHAP feature attribution |
| GET | `/metrics` | Prometheus metrics |

Example:

```bash
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"MedInc":8.3,"HouseAge":42,"AveRooms":6.9,"AveBedrms":1.0,"Population":230,"AveOccup":3.2,"Latitude":37.9,"Longitude":-122.2}'
```

## Docker Learning Path

### Key commands you practice

```bash
docker build -f docker/Dockerfile -t ml-api-naive .
docker build -f docker/Dockerfile.optimized -t ml-api-optimized .
docker run -p 8001:8000 ml-api-optimized
docker logs <container_id>
docker exec -it <container_id> sh
docker inspect <container_id>
docker stop <container_id>
docker rm <container_id>
```

### Naive vs Optimized Dockerfile

- `docker/Dockerfile`: teaching baseline, installs broad dependencies.
- `docker/Dockerfile.optimized`: multi-stage, runtime-only deps, non-root user, smaller attack surface.

## Monitoring and Logging

- Prometheus scrapes `ml-api:8000/metrics` every 15s.
- Grafana dashboard `ML API Overview` includes:
  - request throughput
  - p95 latency
  - error rate
  - in-flight requests
- API logs are structured JSON for ingestion into log backends.

Access:

- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000` (`admin` / `admin`)

## Security Fundamentals Implemented

- Non-root container user in optimized image.
- Strict request validation via Pydantic.
- Rate limiting on inference/explain endpoints.
- Minimal runtime dependency set in optimized container.
- Secret handling through environment variables (no hardcoded secrets).

## Performance Benchmarking

Host and Docker benchmark utility:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/performance_benchmark.py \
  --label host --base-url http://127.0.0.1:8000

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/performance_benchmark.py \
  --label docker --base-url http://127.0.0.1:8001
```

Outputs:

- `outputs/benchmarks/performance_host.json`
- `outputs/benchmarks/performance_docker.json`
- `outputs/benchmarks/host_vs_docker_summary.csv`
- `outputs/figures/host-vs-docker-performance.png`

Latest measured sample (120 requests, concurrency 20):

| Metric | Host | Docker |
|---|---:|---:|
| Startup time | 0.050s | 0.049s |
| P95 latency | 397.3ms | 383.9ms |
| Throughput | 76.24 req/s | 76.64 req/s |
| Failures | 0 | 0 |

Docker image size comparison:

| Image | Size |
|---|---:|
| Naive (`docker/Dockerfile`) | 3.55 GB |
| Optimized (`docker/Dockerfile.optimized`) | 946 MB |

## Tutorial Mini-Book (8 Notebooks)

1. `01-deployment-problem-and-environment-drift.ipynb`
2. `02-docker-fundamentals.ipynb`
3. `03-mlops-fundamentals.ipynb`
4. `04-dataset-eda-and-feature-engineering.ipynb`
5. `05-model-benchmarking-and-evaluation.ipynb`
6. `06-serialization-onnx-and-explainability.ipynb`
7. `07-fastapi-serving.ipynb`
8. `08-docker-compose-observability-and-production-readiness.ipynb`

Each major section follows: definition, theory, motivation, real-world example, visual explanation, code explanation, best practices, common mistakes.

## Production Readiness Checklist

- [ ] Reproducible environment (`uv`, pinned Python)
- [ ] Deterministic training setup (seed, fixed split)
- [ ] Baseline + advanced benchmark evidence
- [ ] Serialization and metadata versioning
- [ ] API contract tests passing
- [ ] Docker image build + runtime validation
- [ ] Compose stack health checks passing
- [ ] Metrics + dashboard visibility
- [ ] Security baseline checks complete

## Future Improvements

- Add CI artifact publishing and registry push.
- Add drift monitoring and alerting strategy.
- Add canary/blue-green deployment playbook.
- Add authn/authz layer for protected inference APIs.
