# Deploy Your AI App for Free in 3 Clicks

Streamlit-first deployment and MLOps portfolio project with an optional advanced FastAPI model-serving track.

## Executive Summary
This repository teaches end-to-end AI app delivery with a production-minded workflow:
- build a modular Streamlit app,
- integrate resilient inference providers,
- validate with reproducible quality gates,
- deploy to Streamlit Community Cloud,
- monitor, troubleshoot, and iterate.

The **default path is Streamlit deployment**.  
An **optional API track** is included for advanced backend/model-serving practice.

---

## Quick Start

### 1) Environment and dependencies (Streamlit default)
```bash
uv venv .venv
source .venv/bin/activate
uv sync --frozen
```

### 2) Run tests and validation (required path)
```bash
uv run pytest -q
uv run python scripts/validate_project.py
```

After the first successful `uv sync`, you can speed repeated local checks with:
```bash
uv run python scripts/validate_project.py --no-sync --skip-notebooks
```

### 3) Launch the app locally
```bash
uv run streamlit run app.py
```

### 4) Optional API track setup
```bash
uv sync --frozen --extra api
uv run python scripts/validate_api_project.py --skip-notebooks
```

---

## Deployment Fundamentals
A deployable AI app needs more than model code:
- stable entrypoint (`app.py`),
- pinned runtime dependencies (`requirements.txt`, `uv.lock`),
- externalized secrets (`.streamlit/secrets.toml`),
- repeatable validation pipeline (`scripts/validate_project.py`).

Core Streamlit architecture:
- `streamlit_app/app.py`: routing/session bootstrap
- `streamlit_app/pages/`: sentiment, summarization, classification, translation
- `streamlit_app/components/`: sidebar and reusable UI
- `streamlit_app/utils/`: config, model inference, caching, helpers

---

## MLOps Fundamentals
The project emphasizes reproducibility and operational discipline:
- deterministic fallback inference chain (HF API -> Ollama -> rule-based fallback),
- centralized config resolution (`st.secrets` -> env -> defaults),
- runtime benchmarking (`outputs/metrics/runtime_benchmark.json`),
- evidence capture (`outputs/deployment/local_validation_evidence.json`),
- notebook generation from scripts for consistent educational artifacts.

Optional API track adds:
- deterministic dataset snapshot and model serialization,
- API contract tests,
- benchmark artifacts for single vs batch prediction.

---

## Git Basics
Typical day-to-day workflow:
```bash
git checkout -b feature/<name>
git add .
git commit -m "feat: <what changed>"
git push origin feature/<name>
```

Recommended habits:
- small focused commits,
- test before push,
- avoid direct risky work on `main`.

---

## GitHub Workflow
1. Push validated branch to GitHub.
2. Open PR and review CI results.
3. Merge only after Streamlit required checks pass.
4. Trigger deployment from the tested branch.

CI design:
- `streamlit-required`: blocking quality gate.
- `api-optional`: non-blocking advanced backend gate.

---

## Streamlit Cloud Deployment
1. Create/push GitHub repo.
2. In Streamlit Community Cloud, create app from repo.
3. Set entrypoint to `app.py`.
4. Add secrets in dashboard (at minimum `HF_API_TOKEN`).
5. Verify pages and inference methods after boot.

Secrets template: `.streamlit/secrets.toml.example`

Reference docs:
- `docs/deployment_checklist.md`
- `docs/troubleshooting_playbook.md`
- `docs/monitoring_guide.md`

---

## Troubleshooting
Start with:
1. dependency/import errors,
2. startup/resource pressure,
3. secrets/config,
4. inference provider availability,
5. malformed input behavior.

Detailed runbook: `docs/troubleshooting_playbook.md`

---

## Monitoring
Primary signals:
- app availability and startup health,
- inference latency (cold vs warm),
- fallback usage trend,
- validation/runtime errors.

Artifacts:
- `outputs/metrics/runtime_benchmark.json`
- `outputs/deployment/local_validation_evidence.json`

Optional API signals:
- `outputs/metrics/fastapi_runtime_benchmark.json`

---

## Security
Implemented baseline practices:
- secrets outside repo,
- `.streamlit/secrets.toml` ignored by Git,
- input validation limits for text/categories,
- graceful handling of provider failures without exposing credentials.

Still recommended for production:
- secret rotation policy,
- dependency CVE scanning,
- centralized audit logging and alerting.

---

## Results
Generated project evidence includes:
- executed notebooks in `outputs/executed_notebooks/`,
- architecture diagrams in `outputs/figures/`,
- runtime benchmarks in `outputs/metrics/`,
- deployment evidence JSON in `outputs/deployment/`.

Optional API artifacts:
- model artifacts in `outputs/api_model/`,
- model benchmarks in `outputs/api_benchmarks/`.

---

## Reproducibility
Fresh-machine flow:
```bash
git clone <repo>
cd <repo>
uv venv .venv
source .venv/bin/activate
uv sync --frozen
uv run python scripts/validate_project.py
```

Optional API flow:
```bash
uv sync --frozen --extra api
uv run python scripts/validate_api_project.py --skip-notebooks
```

---

## Optional Advanced API Track
FastAPI module remains available for backend portfolio extension:
- app entrypoint: `api_app.py`
- core package: `ml_api/`
- API docs at runtime: `/docs`, `/redoc`
- dedicated docs under `docs/api_*`

Use only when you need backend model-serving depth beyond Streamlit deployment goals.

---

## Lessons Learned
- Deployment reliability comes from disciplined packaging and validation, not framework choice.
- Fallback-aware inference design keeps user-facing apps resilient.
- Streamlit projects benefit from explicit reproducibility artifacts (locks, evidence JSON, generated notebooks).
- Separating required vs optional tracks prevents over-coupled CI and onboarding friction.

---

## Future Improvements
1. Add production-grade structured logging sink and alert routing.
2. Add continuous dependency vulnerability scanning.
3. Add synthetic monitoring checks against deployed public URL.
4. Expand API track with auth/rate limiting when backend deployment becomes primary.
5. Add containerized Streamlit deployment profile for non-Streamlit-cloud targets.

---

## Final Verification Report
See `FINAL_PROJECT_VERIFICATION_REPORT.md` for full audit summary, implemented fixes, limitations, and final scoring.
