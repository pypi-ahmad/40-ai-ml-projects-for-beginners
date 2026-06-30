# Deployment Troubleshooting Playbook

## Fast Triage Order
1. Build/deploy logs (dependency/import errors first).
2. App startup health and resource pressure.
3. Secrets presence (`HF_API_TOKEN`, optional model settings).
4. Inference provider health (HF API -> Ollama -> fallback).
5. User input edge cases and validation paths.

## Common Failure Modes

### 1) Missing or stale dependencies
- **Signal:** `ModuleNotFoundError` or build failure.
- **Diagnosis:** Compare `pyproject.toml`, `uv.lock`, and `requirements.txt`.
- **Fix:** Refresh lock/export, commit, redeploy.

### 2) App starts locally but fails in cloud
- **Signal:** Works locally, fails on Streamlit Cloud startup.
- **Diagnosis:** Check pinned versions, entrypoint path (`app.py`), and missing secrets.
- **Fix:** Reproduce with clean env (`uv sync --frozen`) and redeploy from tested commit.

### 3) Slow startup / cold-start pain
- **Signal:** First load is slow, occasional timeouts.
- **Diagnosis:** Measure startup and first-inference latency in `runtime_benchmark.json`.
- **Fix:** Reduce eager work at import time, rely on cache-backed inference path.

### 4) Memory pressure / throttling
- **Signal:** App instability, resets, or degraded responsiveness.
- **Diagnosis:** Inspect logs + runtime metrics snapshots.
- **Fix:** Reduce payload sizes, avoid heavy model loading on cloud free tier, tune caching.

### 5) Secret/config issues
- **Signal:** HF path unavailable; app always falls back.
- **Diagnosis:** Verify Streamlit cloud secrets and local `.streamlit/secrets.toml`.
- **Fix:** Set/rotate token, redeploy, confirm provider path in app status.

### 6) Classification input misuse
- **Signal:** Poor labels or user confusion.
- **Diagnosis:** Too many categories, oversized labels, invalid combinations.
- **Fix:** Enforce category count/length rules and show guided validation errors.

### 7) Optional API track failures
- **Signal:** `tests/api` or API training scripts fail.
- **Diagnosis:** Check whether `uv sync --frozen --extra api` was run.
- **Fix:** Treat as optional in default pipeline; run dedicated API gate when needed.

## Incident Notes Template
- Incident date/time (UTC):
- User impact:
- Root cause:
- Detection source:
- Fix applied:
- Preventive action:
