# Project #25: Production-Grade AI Task Planning & Productivity Agent

Production-ready AI productivity assistant built with **LangGraph**, **LangChain-Ollama**, **FastAPI**, **Streamlit**, **SQLite**, **ChromaDB**, and **DuckDB**.

## What It Does

- Parses messy task input from text/markdown/CSV/JSON/files
- Extracts structured tasks (deadline, estimate, dependencies, context, people, tools, confidence)
- Prioritizes with multiple frameworks: Eisenhower, MoSCoW, ABCDE, RICE, ICE, WSJF, weighted
- Plans dependency-aware schedules with risk/confidence metadata
- Supports re-planning and interruption recovery
- Persists memory in SQLite + semantic retrieval in ChromaDB
- Runs reflection + recommendations after each schedule
- Tracks analytics in DuckDB and MLflow
- Exposes FastAPI, Streamlit dashboard, and Typer CLI

## Architecture

LangGraph workflow:

`Planner -> Scheduler -> Validator -> Reflection -> Memory -> Reporter`

Subsystems:
- Ingestion + Task Extraction
- Priority Analyzer
- Dependency Planner (NetworkX)
- Schedule Optimizer
- Calendar Agent (ICS/local + Google stub adapter)
- Memory Manager (SQLite + ChromaDB)
- Reflection Agent
- Recommendation Agent
- Report Generator
- Monitoring + Tool Registry

Detailed architecture: [`docs/architecture.md`](docs/architecture.md)

## Project Structure

```text
configs/                  # Hydra YAML config
src/task_planning_agent/
  agent/                  # LangGraph state, nodes, service
  api/                    # FastAPI app + auth + routers
  ui/                     # Streamlit dashboard
  extraction/             # Task extraction engine
  prioritization/         # Priority strategies + registry
  dependencies/           # Dependency graph planner
  scheduling/             # Scheduler
  calendar/               # ICS + calendar service
  memory/                 # SQLite + Chroma memory
  analytics/              # DuckDB + MLflow metrics
  tools/                  # MCP-style local tools + connector contracts
  reflection/             # Reflection agent
  recommendations/        # Recommendation engine
  reports/                # Report generation
tests/                    # Unit + integration-style tests
notebooks/                # Educational notebook
scripts/                  # notebook execution and utility scripts
artifacts/                # reports, logs, screenshots
```

## Setup (uv + Python 3.12)

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv sync --extra dev
cp .env.example .env
```

## Run

### FastAPI
```bash
uv run python app.py
```

API docs: `http://localhost:8000/docs`

### Streamlit Dashboard
```bash
uv run streamlit run streamlit_app.py
```

### CLI
```bash
uv run task-agent plan \
  --user-id ahmad \
  --input-text "- Finish report by tomorrow 5pm 90min" \
  --strategy wsjf
```

## FastAPI Endpoints

- `POST /auth/register`
- `POST /auth/login`
- `POST /plan`
- `POST /replan`
- `GET /tasks`
- `GET /calendar`
- `POST /calendar/export`
- `GET /history`
- `GET /search`
- `GET /preferences`
- `POST /preferences`
- `GET /report`
- `GET /health`
- `GET /docs`

## Streamlit Pages

- Dashboard
- Today's Plan
- Weekly Planner
- Task Inbox
- Calendar
- Memory
- Analytics
- Reports
- Settings

## Notebook

- `notebooks/01_zero_to_hero_productivity_agent.ipynb`
- Covers architecture, algorithms, prompt/structured outputs, tool calling, memory, reflection, evaluation

Execute sequentially:

```bash
uv run python scripts/run_notebook.py
```

## Testing

```bash
uv run pytest -q
```

## Configuration

Main config: `configs/config.yaml`

Configurable without code changes:
- Model families and fallback chains
- Planning strategy defaults
- Scheduler objective weights
- Calendar mode
- Memory paths
- API/UI host/port

## Monitoring and Logging

- Structured logging with Rich
- Runtime monitor: CPU, memory, GPU availability
- MLflow metric tracking
- Execution traces persisted in memory and reports

## Screenshots and Execution Artifacts

Store runtime evidence in:
- `artifacts/screenshots/streamlit_dashboard.png`
- `artifacts/screenshots/fastapi_docs.png`
- `artifacts/screenshots/timeline.png`
- `artifacts/screenshots/kanban.png`
- `artifacts/screenshots/gantt.png`
- `artifacts/screenshots/dependency_graph.png`
- `artifacts/screenshots/analytics_dashboard.png`
- `artifacts/screenshots/memory_search.png`
- `artifacts/logs/`

## Advanced Integrations

Implemented as production connector contracts with live-ready stubs:
- GitHub Issues
- Jira
- Notion
- Google Tasks
- Todoist
- Slack summarization
- Email summarization
- WhatsApp task extraction

Google Calendar adapter is stubbed by default until OAuth credentials are configured.

## Future Improvements

- Full Google Calendar OAuth integration path
- Live SaaS connector implementations behind existing contracts
- Real-time collaborative planning and websocket push updates
- Stronger optimization with mixed-integer scheduling backend
- Dedicated React frontend for enterprise UX
