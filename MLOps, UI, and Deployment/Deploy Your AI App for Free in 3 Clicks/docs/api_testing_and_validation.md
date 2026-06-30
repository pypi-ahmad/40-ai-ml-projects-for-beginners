# FastAPI Testing and Validation (Optional Track)

## Test Suite

```bash
uv run pytest -q -o addopts= tests/api
```

Coverage areas:
- Endpoint contracts
- Invalid payload and type failures
- Batch limit behavior
- Explainability endpoint
- Docs availability (`/docs`, `/redoc`)
- Training + serialization integrity

## End-to-End Validation

```bash
uv run python scripts/validate_api_project.py --skip-notebooks
```

This executes:
1. Dataset snapshot generation
2. Model training + serialization
3. API tests
4. Runtime benchmark capture

## Performance Artifacts
- `outputs/metrics/fastapi_runtime_benchmark.json`
- `outputs/api_benchmarks/model_benchmark.csv`
- `outputs/api_benchmarks/model_benchmark.json`
- `outputs/api_benchmarks/automl_benchmark.json`
