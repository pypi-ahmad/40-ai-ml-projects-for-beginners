# Connect Your LLM to Google Sheets

Production-grade AI Spreadsheet Analytics Platform with Google Sheets + deterministic Python analytics + local LLM insights (Ollama).

## Project Overview
This project demonstrates hybrid analytics:
- Deterministic Python does all computation (filtering, grouping, aggregation, KPI math, trends).
- Local LLM does interpretation (insights, narratives, recommendations) from deterministic evidence only.

Primary stack:
- Google Sheets API (`gspread` + Google auth)
- Pandas + NumPy + Plotly
- Streamlit dashboard
- Ollama local models (`qwen3.5:4b`, `granite4.1:3b`)
- SQLite artifact/state store

## Why `temperature=0` matters for business analytics
Deterministic decoding is required for production analytics because it provides:
1. Reproducibility across repeated runs.
2. Auditability for stakeholder review and compliance.
3. Stable regression testing and benchmark comparison.
4. Lower variance in LLM-as-judge scoring.

This project enforces `OLLAMA_TEMPERATURE=0` in settings validation.

## Features
- Secure Google Sheets ingestion with service-account first auth.
- Multi-spreadsheet, multi-worksheet loading and metadata inspection.
- Local caching + incremental refresh diff.
- Data quality report (missing, duplicates, invalid date/number, outliers, mixed types, empty/constant columns).
- Configurable cleaning strategies.
- Automated EDA and KPI generation.
- Plotly visualizations (bar/line/scatter/pie/hist/heatmap/box/violin/correlation/time-series).
- Hybrid insights for multiple analyst personas.
- Conversational analytics (tool-first, evidence-first, memory-backed).
- Report pack export (Markdown/HTML/PDF/Excel/PowerPoint).
- Safe Google Sheets write-back (new worksheet by default).
- Benchmark suite (100 cases) + LLM-as-judge scaffolding.
- Streamlit AI Analyst Dashboard.

## Repository Layout
```text
app/streamlit_app.py
src/ai_spreadsheet_analytics/
docs/
notebooks/zero_to_hero.ipynb
data/benchmarks/questions.json
tests/
```

## Setup
### 1) Python + uv
```bash
uv venv .venv
source .venv/bin/activate
uv sync --extra dev
```

### 2) Environment
```bash
cp .env.example .env
```
Set credential/model paths in `.env`.

### 3) Ollama models
```bash
ollama pull qwen3.5:4b
ollama pull granite4.1:3b
```

### 4) Run tests
```bash
uv run pytest
```

### 5) Launch Streamlit app
```bash
uv run streamlit run app/streamlit_app.py
```

### 6) CLI quickstart
```bash
uv run asheets clean-csv data/samples/retail_sales.csv data/artifacts/retail_clean.csv
uv run asheets run-csv data/samples/retail_sales.csv --role executive
```

## Google Cloud Configuration Guide
Full beginner guide: [docs/google_cloud_setup.md](docs/google_cloud_setup.md)

Includes:
- Create project
- Enable Sheets/Drive APIs
- Service account + JSON keys
- Sharing spreadsheets
- Common permission errors
- Security best practices
- `.env` usage
- Git leak prevention

## Authentication Guide
- Runtime default: Service account (`GOOGLE_SERVICE_ACCOUNT_JSON`).
- Optional educational path: OAuth in connector module.
- Default scopes are read-only.
- Write-back requires broader scopes and explicit intent.

## Datasets
Registry and preparation guidance:
- [docs/datasets.md](docs/datasets.md)
- `scripts/prepare_datasets.py`
- `data/samples/*.csv`

Core domains:
- Retail/Sales
- HR
- Finance/Marketing mix

## Architecture
See [docs/architecture.md](docs/architecture.md) for mermaid diagrams:
- API/auth flow
- Ingestion/cleaning/EDA/LLM/report workflow
- Conversation pipeline
- Write-back safety model

## Notebook
Notebook path: [notebooks/zero_to_hero.ipynb](notebooks/zero_to_hero.ipynb)
Executed proof notebook: `notebooks/zero_to_hero.executed.ipynb`

Covers:
- Google APIs and auth (OAuth vs service account)
- `gspread` and pandas integration
- Hybrid deterministic + LLM workflow
- Prompt engineering roles
- REST vs LangChain comparison pattern
- KPI + report generation walkthrough

## Streamlit App
Main capabilities:
- Setup instructions in sidebar
- Spreadsheet selection
- Data preview + quality checks
- Cleaning options
- Analytics charts
- LLM insight generation
- Chat interface with memory
- Report export center
- AI Analyst metrics panel

## Benchmarking and Evaluation
- Benchmark cases: `data/benchmarks/questions.json` (100 questions).
- Benchmark runner: `src/ai_spreadsheet_analytics/benchmark.py`.
- LLM judge engine: `src/ai_spreadsheet_analytics/judge.py`.
- Benchmark execution script: `scripts/run_benchmark.py`.
- Model comparison script: `scripts/compare_models.py`.
- Latest generated artifacts:
  - `data/artifacts/benchmark_bench_099443cafc.json`
  - `data/artifacts/model_comparison.json`

Targets:
- Numeric correctness
- Hallucination rate
- Latency
- Consistency
- Business usefulness

Note: if Ollama daemon/models are unavailable, scripts emit fallback metrics and record connection errors.

## Testing
Test modules include:
- data validation
- cleaning logic
- analytics outputs
- prompt generation
- benchmark scoring
- auth helper behavior (mocked)

Run:
```bash
uv run pytest -q
```

## Security
- Credentials only via env vars.
- JSON keys ignored in git.
- Read-only scopes by default.
- Retries/backoff for API calls.
- Write-back never overwrites unless explicit `overwrite=True`.

## Performance Tracking
Performance helper (`performance.py`) supports:
- Load time
- Cleaning time
- Visualization generation time
- LLM latency
- Peak memory
- Token estimate from LLM responses

## Limitations
- Live Google and live Ollama execution need local credentials/models.
- PDF export currently text-oriented (no full chart embedding pipeline).
- LangChain adapter depends on optional package compatibility with installed version.
- Advanced forecasting/clustering are baseline implementations (linear trend + quantile clusters).

## Future Improvements
1. Add richer semantic parser for NL filters.
2. Add robust model-based clustering and forecasting modules.
3. Add async ingestion workers for large multi-sheet workloads.
4. Add Streamlit auth wizard and secret scanner integration.
5. Add CI pipeline with notebook execution and screenshot artifacts.

## References
- Google Sheets API docs
- Google Drive API docs
- gspread docs
- Ollama API docs
- Plotly docs
- Streamlit docs
