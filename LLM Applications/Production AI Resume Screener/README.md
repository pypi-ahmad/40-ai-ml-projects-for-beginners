# Production-Grade AI Resume Screening & Talent Intelligence Platform

Enterprise-style local ATS powered by Ollama + RAG + OCR + API/UI/CLI/MCP.

This repository was executed end-to-end with real runs (not mock/smoke), and outputs in `outputs/e2e/` and `outputs/reports/` were generated from live execution.

## 1) What This Project Does

This platform ingests resumes at scale, parses structured candidate profiles, compares them against job descriptions, computes explainable scores, supports semantic recruiter search, generates interview question packs, and exports hiring reports.

Core capabilities:

- Multi-format ingestion: PDF, DOCX, TXT, Markdown, images, scanned PDFs
- OCR routing: digital PDF vs scanned PDF detection
- Hybrid parsing: rule-based extraction + LLM JSON fixup
- Blind hiring mode: PII redaction before scoring/search
- Semantic + weighted scoring with evidence
- Duplicate detection: exact hash + semantic similarity support
- RAG recruiter assistant over ChromaDB
- Interview question generation
- Report generation: Markdown, HTML, JSON, PDF
- Multi-interface operation: FastAPI, Streamlit, Typer CLI, MCP server

## 2) Real Environment Used

Verified on this machine:

- OS: Linux (Ubuntu environment)
- Python: `3.14.4`
- uv: `0.11.19`
- Runtime inference stack: local Ollama endpoint `http://localhost:11434`

> Project target remains Python 3.12+, local CPU/GPU inference support.

## 3) Tech Stack

- LLM + orchestration: `Ollama`, `LangChain`
- API: `FastAPI`, `Uvicorn`
- UI: `Streamlit`
- CLI: `Typer`, `Rich`
- Data: `SQLite`, `SQLAlchemy`
- Vector DB: `ChromaDB`
- Embeddings: `sentence-transformers` primary, Ollama fallback
- OCR/doc: `PyMuPDF`, `pytesseract`, `opencv-python`, optional `PaddleOCR`
- Config: `Hydra/OmegaConf` YAML
- Analytics: `pandas`, `plotly`
- Reports: `jinja2`, `reportlab`
- Monitoring/logging: structured logging + system metrics capture

## 4) Architecture (Code Map)

- `src/resume_ai/service.py`: orchestration facade for all interfaces
- `src/resume_ai/ingestion/`: loaders, dedupe, ingestion pipeline
- `src/resume_ai/parsing/`: resume/JD parsing + normalization + redaction
- `src/resume_ai/ocr/`: OCR extraction and preprocessing
- `src/resume_ai/matching/`: weighted explainable scoring engine
- `src/resume_ai/rag/`: semantic recruiter search assistant
- `src/resume_ai/reports/`: Markdown/HTML/JSON/PDF exports
- `src/resume_ai/api/`: FastAPI endpoints
- `src/resume_ai/ui/`: Streamlit dashboard
- `src/resume_ai/cli/`: Typer CLI
- `src/resume_ai/mcp/`: MCP tool server
- `src/resume_ai/db/`: SQLAlchemy models/repository/session
- `config/`: runtime/model/embedding/OCR/scoring/prompt YAML

Data flow:

1. Resume upload -> format detection -> OCR (if needed) -> structured parse
2. Blind redaction (default ON) -> embedding -> SQLite + Chroma persistence
3. JD parse + normalization -> scoring with evidence and confidence
4. Recruiter semantic queries via RAG over Chroma collections
5. Interview/report generation from candidate-job context

## 5) Installation

```bash
uv venv .venv
source .venv/bin/activate
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev
```

## 6) Build + Test (Real Commands Used)

```bash
UV_CACHE_DIR=/tmp/uv-cache uv build
UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall src
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Observed result:

- Build artifacts created:
  - `dist/production_ai_resume_screener-0.1.0.tar.gz`
  - `dist/production_ai_resume_screener-0.1.0-py3-none-any.whl`
- Tests: `7 passed`

## 7) Run the Platform

### 7.1 FastAPI

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run uvicorn resume_ai.api.main:app --host 127.0.0.1 --port 19017
```

> Default `8000` may be occupied by other local services; use free port as above.

### 7.2 Streamlit Dashboard

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run streamlit run src/resume_ai/ui/app.py --server.headless true --server.port 19018 --server.address 127.0.0.1
```

### 7.3 CLI

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run resume-ai ingest data/samples/sample_resume.txt --blind-mode
UV_CACHE_DIR=/tmp/uv-cache uv run resume-ai score 1 7
UV_CACHE_DIR=/tmp/uv-cache uv run resume-ai search "LangGraph production" --top-k 5
UV_CACHE_DIR=/tmp/uv-cache uv run resume-ai interview 1 7
UV_CACHE_DIR=/tmp/uv-cache uv run resume-ai report 1 7 --output-dir outputs/reports
UV_CACHE_DIR=/tmp/uv-cache uv run resume-ai compare 7 1
```

### 7.4 MCP Server

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run resume-ai-mcp
```

MCP tools exposed:

- `resume_search`
- `candidate_lookup`
- `generate_interview`
- `score_candidate`
- `generate_report`

## 8) Live End-to-End Execution Proof

Real live API flow was executed and persisted in `outputs/e2e/`:

- `health.json`
- `upload.json`
- `job.json`
- `score.json`
- `search.json`
- `interview.json`
- `reports.json`
- `reports_list.json`
- `analytics.json`
- `summary.json`
- CLI outputs (`cli_*.json`, `cli_dashboard.txt`)
- Streamlit HTML fetch (`streamlit_home.html`)

### Final E2E summary (`outputs/e2e/summary.json`)

- `candidate_id`: `1`
- `job_id`: `7`
- `total_score`: `60.36`
- `search_citations`: `1`
- reports generated:
  - `outputs/reports/candidate_1_job_7.md`
  - `outputs/reports/candidate_1_job_7.html`
  - `outputs/reports/candidate_1_job_7.json`
  - `outputs/reports/candidate_1_job_7.pdf`

### Database verification (post-run)

Current SQLite row counts:

- `candidates`: `1`
- `resumes`: `1`
- `job_descriptions`: `7`
- `candidate_job_scores`: `15`
- `interviews`: `3`
- `reports`: `16`
- `candidate_skills`: `14`
- `experience`: `13`
- `projects`: `1`

## 9) API Endpoints

- `GET /health`
- `POST /upload`
- `GET /resume/{resume_id}`
- `GET /candidate/{candidate_id}`
- `POST /job`
- `POST /score`
- `POST /compare`
- `POST /search`
- `POST /interview`
- `POST /reports`
- `GET /reports`
- `GET /analytics`
- `POST /notes`
- `POST /mcp/tool` (HTTP bridge)

Envelope contract:

```json
{
  "status": "ok",
  "data": {},
  "errors": [],
  "trace_id": "..."
}
```

## 10) Configuration

Primary files:

- `config/models/default.yaml`
- `config/embeddings/default.yaml`
- `config/ocr/default.yaml`
- `config/scoring/default.yaml`
- `config/runtime/default.yaml`
- `config/runtime/retries.yaml`

Runtime model/weight behavior is configurable via YAML edits and app restart.

## 11) Notes from Production Hardening

Implemented during final E2E stabilization:

- Fixed Chroma embedding dimension mismatch by enforcing configurable output dimension (`384`) in embedding service.
- Reduced Ollama fallback timeout to avoid long blocking on unavailable local model endpoints.
- Fixed duplicate-ingest behavior to sync candidate detail tables (`candidate_skills`, `experience`, `projects`) even on exact-hash duplicate resumes.
- Disabled Chroma anonymized telemetry in vector client settings.

## 12) Artifacts & Reports

- API/CLI/Streamlit evidence: `outputs/e2e/`
- Generated hiring reports: `outputs/reports/`
- Build outputs: `dist/`
- Architecture notes: `docs/architecture.md`
- Walkthrough notebook: `notebooks/01_resume_ai_platform_walkthrough.ipynb`

## 13) Screenshots (Real Captures)

Captured from live Streamlit/FastAPI/CLI execution:

- Dashboard: `outputs/screenshots/dashboard.png`
- Resume Upload: `outputs/screenshots/resume_upload.png`
- Resume Parsing: `outputs/screenshots/resume_parsing.png`
- Candidate Viewer: `outputs/screenshots/candidate_viewer.png`
- Ranking Table: `outputs/screenshots/ranking_table.png`
- Candidate Comparison: `outputs/screenshots/candidate_comparison.png`
- Interview Generator: `outputs/screenshots/interview_generator.png`
- Analytics: `outputs/screenshots/analytics.png`
- FastAPI Swagger: `outputs/screenshots/fastapi_swagger.png`
- CLI Execution: `outputs/screenshots/cli.png`

## 14) Known Warning (Non-blocking)

During tests, Chroma emits Python 3.16 deprecation warning (`asyncio.iscoroutinefunction`).

- No functional impact in current runtime.
- Tests and live E2E still complete successfully.
