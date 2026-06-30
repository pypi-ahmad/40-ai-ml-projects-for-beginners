# FastAPI ML API Guide (Optional Advanced Track)

## Overview
This repository contains two tracks:
- Streamlit deployment track (required/default path)
- FastAPI production-style ML serving track (`ml_api/`) as an optional advanced module

The FastAPI track serves curated Ames Housing predictions with strict request validation, batch inference, explainability, and observability.

## Quick Start

```bash
uv sync --frozen --extra api
uv run python scripts/generate_ames_snapshot.py
uv run python scripts/train_api_models.py
uv run uvicorn api_app:app --reload
```

Open:
- Swagger: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Endpoints

- `GET /health` : process status, uptime, model readiness.
- `GET /model-info` : model type/version, feature count, dataset hash, benchmark summary.
- `POST /predict` : single-record prediction using strict full-schema payload.
- `POST /predict-batch` : vectorized batch inference with size guard.
- `GET /metrics` : request counts and latency distributions.
- `POST /explain` : SHAP (when installed) or perturbation-based explanation.

## Validation Rules
- Input schema is explicit and `extra="forbid"`.
- Numeric ranges enforce domain sanity.
- Year consistency validators prevent impossible records.
- Batch payloads are constrained by schema + runtime limits.

## Error Contract
All non-2xx responses use:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {},
    "request_id": "..."
  }
}
```

## Reproducibility
- Deterministic seed in training pipeline.
- Immutable committed dataset snapshot (`data/raw/ames_housing_curated.csv`).
- Metadata includes dataset hash and benchmark metrics.

## Optional Exhaustive Stack
Install heavy benchmark dependencies:

```bash
uv sync --frozen --extra api --extra automl
```

Then rerun training to generate full optional benchmark outputs.
