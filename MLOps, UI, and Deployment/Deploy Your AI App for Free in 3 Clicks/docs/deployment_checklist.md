# Deployment Checklist

## 1. Development
- [ ] Scope, target users, and success criteria are defined.
- [ ] Fallback inference strategy is documented.
- [ ] No secrets are hardcoded in app code, notebooks, or screenshots.

## 2. Streamlit Required Validation
- [ ] `uv run pytest -q` passes (Streamlit/default suite).
- [ ] App import smoke test passes: `uv run python -c "from streamlit_app.app import main; print('ok')"`.
- [ ] App launches locally: `uv run streamlit run app.py`.
- [ ] All critical flows validated: sentiment, summarization, classification, translation.
- [ ] Invalid input paths are verified (empty input, too-long input, invalid categories).

## 3. Optional API Track Validation
- [ ] API dependencies synced: `uv sync --frozen --extra api`.
- [ ] API gate passes: `uv run python scripts/validate_api_project.py --skip-notebooks`.
- [ ] Endpoint contracts verified (`/health`, `/model-info`, `/predict`, `/predict-batch`, `/metrics`, `/explain`).

## 4. Packaging and Reproducibility
- [ ] `pyproject.toml` reflects current dependencies and optional extras.
- [ ] `uv.lock` is refreshed.
- [ ] `requirements.txt` remains Streamlit-cloud minimal and pinned.
- [ ] `.streamlit/secrets.toml.example` is current.
- [ ] Notebook generators were rerun and notebooks regenerated.

## 5. Cloud Deployment (Streamlit Community Cloud)
- [ ] GitHub repository and branch are up to date.
- [ ] Streamlit app is linked to the correct repo/branch and entrypoint `app.py`.
- [ ] Cloud secrets are configured in Streamlit dashboard.
- [ ] Build logs show healthy startup and no import failures.
- [ ] Public URL is reachable and pages render correctly.

## 6. Monitoring and Operations
- [ ] Runtime benchmark snapshot is updated under `outputs/metrics/`.
- [ ] Troubleshooting playbook covers current known failure modes.
- [ ] Incident and deployment templates are ready for reuse.

## 7. Maintenance
- [ ] Post-release review notes are captured.
- [ ] Next improvements backlog is prioritized.
- [ ] Dependency update cadence is scheduled.
