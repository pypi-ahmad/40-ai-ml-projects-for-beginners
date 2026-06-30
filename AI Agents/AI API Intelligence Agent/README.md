# Project #26: Production-Grade AI API Intelligence Agent

Production-ready multi-agent AI API Intelligence system built with LangGraph, Ollama, FastAPI, Streamlit, and Typer CLI.

This README is based on a **real end-to-end execution on June 30, 2026** with actual outputs and generated artifacts.

## 1) What This Project Does

Given a query, agent:
1. Plans API strategy.
2. Routes to relevant connectors.
3. Handles authentication (API key, bearer, optional no-auth fallback for public APIs).
4. Fetches API data asynchronously with retries/backoff.
5. Validates and reasons over results.
6. Generates reports (Markdown/HTML/PDF/JSON/CSV).
7. Persists run history and response artifacts.
8. Exposes functionality through FastAPI, Streamlit, and CLI.

## 2) Verified Runtime Environment (Real Run)

- OS: Ubuntu (Linux)
- Python (system): `3.14.4`
- Project Python (`uv` venv): `3.12.10`
- `uv`: `0.11.19`
- Package manager: `uv` only

## 3) Architecture (Production Layout)

```text
src/api_intel_agent/
  agents/         # LangGraph state + nodes + runner
  connectors/     # API connector framework and providers
  auth/           # API auth + JWT auth manager
  memory/         # SQLite + Chroma memory
  cache/          # SQLite/Redis cache backends
  tools/          # Tool registry + builtins + OpenAPI toolgen + adapters
  analytics/      # Metrics and chart generation
  reporting/      # Multi-format report generation
  api/            # FastAPI app + auth dependencies + DB
  cli/            # Typer CLI
  monitoring/     # CPU/RAM/GPU/cache/latency telemetry
  scheduler/      # Background scheduling hooks
  config/         # YAML + Hydra/Pydantic settings

streamlit_app/    # Dashboard pages
notebooks/        # Zero-to-hero notebook + executed notebook
docs/             # Architecture docs
k8s/              # Deployment/service/config/hpa manifests
```

LangGraph flow:
`Request Planner -> API Router -> Authentication -> Data Fetch -> Validation -> Reasoning -> Report Generator -> Memory -> Reflection`

## 4) Setup (Zero to Hero)

### Step A: Create and sync environment

```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
UV_CACHE_DIR=.uv-cache uv sync --all-groups --extra notebook
```

### Step B: Configure secrets

```bash
cp .env.example .env
# edit .env and set at minimum:
# AGENT_JWT_SECRET=...
# optional provider tokens (GITHUB_TOKEN, NEWS_API_KEY, ...)
```

### Step C: Build package

```bash
UV_CACHE_DIR=.uv-cache uv build
```

Build outputs from real run:
- `dist/ai_api_intelligence_agent-0.1.0-py3-none-any.whl`
- `dist/ai_api_intelligence_agent-0.1.0.tar.gz`

### Step D: Run tests

```bash
UV_CACHE_DIR=.uv-cache uv run pytest -q
```

Real run result: `10 passed`.

## 5) Real End-to-End Execution (No Mock, No Dry Run)

## A) Demo script (live APIs)

```bash
AGENT_DISABLE_CHROMA=1 \
AGENT__LLM__TIMEOUT_SECONDS=10 \
AGENT__AGENT__RETRY_MAX_ATTEMPTS=1 \
UV_CACHE_DIR=.uv-cache \
timeout 180s uv run python scripts/run_demo.py
```

Real result:
- `run_id`: `71cd915b-dc6d-45a6-885d-b79aefe415e7`
- `status`: `success`
- `insights`:
  - `jsonplaceholder returned 100 records`
  - `openlibrary returned 100 records`
- `errors`: `[]`
- reports generated in all formats.

## B) FastAPI live server + real endpoint flow

### Start server

```bash
AGENT_DISABLE_SCHEDULER=1 \
AGENT_DISABLE_CHROMA=1 \
AGENT__LLM__TIMEOUT_SECONDS=10 \
AGENT__AGENT__RETRY_MAX_ATTEMPTS=1 \
UV_CACHE_DIR=.uv-cache \
uv run python main.py
```

### Real E2E calls executed

- `GET /health` -> `ok`
- `POST /auth/register` -> `ok`
- `POST /auth/token` -> `bearer`
- `POST /analyze` -> success
- `GET /history` -> records returned
- `GET /report/{run_id}` -> report payload returned
- `GET /docs` -> HTTP `200`
- `GET /metrics` -> telemetry payload returned

FastAPI analyze real output:
- `run_id`: `d4170cd2-83dc-4d0d-8c62-51818bad95b3`
- `status`: `success`
- `errors_count`: `0`
- `sources_count`: `2`
- report artifacts existed for `markdown/html/pdf/json/csv`.

## C) Streamlit live server execution

### Start

```bash
AGENT_DISABLE_CHROMA=1 UV_CACHE_DIR=.uv-cache uv run python app.py
```

Real verification:
- local HTTP check returned `200`
- HTML response contained Streamlit app content.

## D) CLI real execution

Executed commands:

```bash
uv run agent query "jsonplaceholder openlibrary python trends"
uv run agent github
uv run agent weather --city London --lat 51.5 --lon -0.12
uv run agent analyze artifacts/reports/<latest>.json
uv run agent search memory "python"
```

Real outcomes:
- `agent query` generated run output with recommendations.
- `agent github` returned non-empty table (e.g., `TheAlgorithms/Python`).
- `agent weather` returned live weather JSON payload.
- `agent analyze` returned structured JSON analysis.
- `agent search memory` executed successfully.

## E) Notebook sequential execution

```bash
AGENT_DISABLE_CHROMA=1 \
AGENT__LLM__TIMEOUT_SECONDS=10 \
AGENT__AGENT__RETRY_MAX_ATTEMPTS=1 \
UV_CACHE_DIR=.uv-cache \
uv run python scripts/execute_notebook_sequential.py \
  notebooks/zero_to_hero_api_intelligence_agent.ipynb \
  --output notebooks/zero_to_hero_api_intelligence_agent.executed.ipynb
```

Real output file generated:
- `notebooks/zero_to_hero_api_intelligence_agent.executed.ipynb`

## 6) Verified Generated Artifacts

From real run:

- Reports: `artifacts/reports/report_<run_id>.{md,html,pdf,json,csv}`
- Charts: `artifacts/charts/api_latency_<timestamp>.html`
- Memory DB: `artifacts/memory/agent_memory.db`
  - queries/responses/api summaries persisted
- Cache DB: `artifacts/cache/cache.db`
- Build artifacts: `dist/*.whl`, `dist/*.tar.gz`
- Executed notebook: `notebooks/zero_to_hero_api_intelligence_agent.executed.ipynb`

## 7) FastAPI Endpoint Reference

- `POST /auth/register`
- `POST /auth/token`
- `POST /analyze`
- `POST /query`
- `GET /github`
- `GET /news`
- `GET /weather`
- `GET /repos`
- `GET /memory`
- `GET /search`
- `GET /history`
- `GET /report/{run_id}`
- `GET /report?run_id=...`
- `GET /health`
- `GET /metrics`
- `POST /tools/openapi`
- `GET /docs`

## 8) Configuration

Main config: `configs/settings.yaml`

Main groups:
- `llm`
- `agent`
- `auth`
- `cache`
- `memory`
- `apis`
- `monitoring`
- `scheduler`
- `ui`
- `reports`

Environment override format:
- `AGENT__SECTION__KEY=value`

Examples:

```bash
export AGENT__LLM__TIMEOUT_SECONDS=10
export AGENT__AGENT__RETRY_MAX_ATTEMPTS=1
export AGENT_DISABLE_CHROMA=1
```

## 9) Production Notes

- Public APIs now support no-auth fallback where appropriate (e.g., GitHub) to avoid hard failure when token not present.
- Missing required credentials return explicit `skipped_missing_credentials` status.
- Chroma memory degrades gracefully in restricted/offline environments.
- Report generation is deterministic across formats.

## 10) Deploy

- Docker: `Dockerfile`, `docker-compose.yml`
- Kubernetes manifests:
  - `k8s/deployment.yaml`
  - `k8s/service.yaml`
  - `k8s/configmap.yaml`
  - `k8s/secret-template.yaml`
  - `k8s/hpa.yaml`

