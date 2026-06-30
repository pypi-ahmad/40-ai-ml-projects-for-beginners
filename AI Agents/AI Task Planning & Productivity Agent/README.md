# Project #25: Production-Grade AI Task Planning & Productivity Agent

A production-ready local AI productivity assistant built with LangGraph, LangChain-Ollama, FastAPI, Streamlit, SQLite, ChromaDB, DuckDB, and MLflow.

This project was executed end-to-end in a real live run (no mocks/simulations) with verified outputs.

## 1. What You Get

- Multi-agent planning workflow (LangGraph): `Planner -> Scheduler -> Validator -> Reflection -> Memory -> Reporter`
- Messy input parsing and structured task extraction
- Multi-framework prioritization: Eisenhower, MoSCoW, ABCDE, RICE, ICE, WSJF, weighted
- Dependency-aware scheduling (NetworkX DAG)
- Replanning support for interruptions
- Persistent memory:
  - SQLite (operational data)
  - ChromaDB (semantic retrieval)
- Analytics with DuckDB + MLflow logging
- FastAPI production backend with JWT auth + RBAC basics
- Streamlit dashboard UI
- Typer CLI for automation
- Educational notebook that runs sequentially

## 2. Architecture

### 2.1 Core graph

```text
Planner -> Scheduler -> Validator -> Reflection -> Memory -> Reporter
```

### 2.2 Subsystems

- `ingestion/`: text, markdown, csv/json, local file normalization
- `extraction/`: heuristic + structured extraction to `Task`
- `prioritization/`: strategy registry and normalized scoring
- `dependencies/`: DAG build + cycle detection
- `scheduling/`: deadline-aware block planning
- `memory/`: SQLite + Chroma persistence and retrieval
- `analytics/`: metric snapshots and aggregate storage
- `api/`: FastAPI endpoints and auth
- `ui/`: Streamlit dashboard pages
- `tools/`: local-first tool calling + external connector contracts

Detailed architecture note: `docs/architecture.md`
Graph spec: `docs/agent_graph.mmd`

## 3. Project Layout

```text
configs/
src/task_planning_agent/
  agent/
  analytics/
  api/
  calendar/
  dependencies/
  extraction/
  ingestion/
  llm/
  memory/
  observability/
  prioritization/
  recommendations/
  reflection/
  reports/
  scheduling/
  tools/
  ui/
notebooks/
scripts/
tests/
artifacts/
```

## 4. Zero-to-Hero Setup (Exact)

## 4.1 Prerequisites

- Linux (tested on Ubuntu)
- Python 3.12
- `uv`
- Chrome/Chromium (for screenshot capture)

## 4.2 Install

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv sync --extra dev
cp .env.example .env
```

## 4.3 Build

```bash
uv run python -m compileall -q src
uv build
```

Build artifacts:
- `dist/task_planning_productivity_agent-0.1.0.tar.gz`
- `dist/task_planning_productivity_agent-0.1.0-py3-none-any.whl`

## 5. Run the System

## 5.1 FastAPI

```bash
uv run python app.py
```

- API docs: `http://127.0.0.1:8000/docs`
- Health: `GET /health`

## 5.2 Streamlit

```bash
uv run streamlit run streamlit_app.py
```

## 5.3 CLI

```bash
uv run task-agent plan --user-id ahmad --input-text "- Finish report by tomorrow 5pm 90min" --strategy wsjf
uv run task-agent replan --user-id ahmad --reason "Urgent blocker" --additional-input "- Fix blocker today 30min"
```

## 5.4 Notebook

```bash
uv run python scripts/run_notebook.py
```

Notebook:
- `notebooks/01_zero_to_hero_productivity_agent.ipynb`

## 6. API Surface

Implemented routes:

- `POST /auth/register`
- `POST /auth/login`
- `POST /plan`
- `POST /replan`
- `GET /tasks`
- `GET /history`
- `GET /search`
- `GET /calendar`
- `POST /calendar/export`
- `GET /preferences`
- `POST /preferences`
- `GET /report`
- `GET /health`
- `GET /strategies`
- `POST /tools/{tool_name}`

## 7. Real End-to-End Execution (Verified)

Latest full live run ID:
- `final_20260630T131643Z`

Run summary:
- `artifacts/reports/e2e_final_20260630T131643Z.md`

Master run log:
- `artifacts/logs/e2e_final_20260630T131643Z.log`

Key validated outcomes from the run:
- Notebook executed: PASS
- Artifact generation: PASS
- FastAPI live workflow: PASS
- CLI live workflow: PASS
- Streamlit live launch: PASS
- Output verification script: PASS (`VALIDATION_OK`)

Validated live outputs:
- Plan API response: `artifacts/reports/plan_response_final_20260630T131643Z.json`
  - `plan_id`: `e6b4a599-fa42-46a3-9827-6711e57991d9`
  - `schedule` length: `4`
- Replan API response: `artifacts/reports/replan_response_final_20260630T131643Z.json`
  - `plan_id`: `66d02518-174e-41ff-a645-7524cef7cb38`
  - `schedule` length: `8`
- ICS export: `artifacts/reports/calendar_export_final_20260630T131643Z.ics`
  - Verified VEVENT count: `8`
- Health response: `artifacts/reports/health_final_20260630T131643Z.json`
  - `status`: `ok`

Screenshots captured:
- FastAPI docs: `artifacts/screenshots/fastapi_docs_final_20260630T131643Z.png`
- Streamlit launch capture: `artifacts/screenshots/streamlit_final_20260630T131643Z.png`
- Planning visuals:
  - `artifacts/screenshots/timeline.png`
  - `artifacts/screenshots/gantt.png`
  - `artifacts/screenshots/kanban.png`
  - `artifacts/screenshots/dependency_graph.png`
  - `artifacts/screenshots/analytics_dashboard.png`
  - `artifacts/screenshots/memory_search.png`

## 8. Testing and Quality Gates

```bash
uv run ruff check src tests scripts
uv run pytest -q
```

Current status: passing.

## 9. Configuration

Main config: `configs/config.yaml`

Change without code edits:
- model families and fallback chains
- planner strategy
- scheduling objective weights
- memory paths
- calendar mode
- API/UI host and ports

## 10. Integrations

Implemented now:
- ICS/local calendar workflows
- Google calendar adapter stub (credential-gated)
- External connectors as production contracts + stubs:
  - GitHub Issues
  - Jira
  - Notion
  - Todoist
  - Google Tasks
  - Slack
  - Email summary
  - WhatsApp

## 11. Known Production Notes

- LLM fallback support is implemented and configurable; `llm.enabled` defaults to `false` for predictable local runs when Ollama is unavailable.
- Streamlit server launch and HTML response are verified in live run; headless screenshot capture shows loading skeleton state in this environment.

## 12. Next Hardening Steps

- Enable live Ollama model execution (`llm.enabled: true`) and verify model family fallback against running local models.
- Replace Google calendar stub with OAuth live adapter.
- Replace selected external connector stubs with credential-backed live implementations.
- Add browser-based E2E UI tests with element-level waits.
