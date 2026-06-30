# Project #30: Production-Grade LangGraph Multi-Agent Platform

Production-ready multi-agent workflow platform built with LangGraph, FastAPI, Streamlit, Typer CLI, SQLite memory, Chroma semantic retrieval, and enterprise execution controls.

This README is based on the **actual codebase** and a **real end-to-end run** completed on this environment.

## 1. What This Platform Does

The platform executes enterprise AI workflows with:

- typed shared graph state
- dynamic routing
- multi-stage reasoning and verification
- persistent memory (SQLite)
- semantic retrieval (Chroma)
- API + CLI + dashboard execution surfaces
- report export (Markdown / HTML / PDF / JSON)

## 2. Verified Real Run (No Mock Execution)

The following were executed successfully in this repository:

- dependency sync + build
- compile checks
- full pytest suite
- real CLI workflow run (strict live Ollama mode)
- real FastAPI server startup and live endpoint calls
- real Streamlit dashboard startup and live page response
- real PNG screenshot capture from live UI endpoints
- Ruff lint + format with zero findings
- real artifact generation and validation

### Verified run summary (generated from `artifacts/reports/verification_summary.json`)

- `generated_at_utc`: `2026-06-30T17:56:06.486508+00:00`
- `strict_live_llm_workflow_id`: `wf_d386a7326277`
- `strict_live_llm_confidence`: `0.85`
- `screenshots_count`: `11`
- `tests`: `7 passed`
- `ruff`: `All checks passed`

## 3. Architecture

## 3.1 Workflow Graph

```text
START
  -> planner
  -> parallel_research
  -> knowledge_merge
  -> consensus_reasoning
  -> writer
  -> fact_checker
  -> reflection
  -> critic
  -> qa
  -> citation
  -> supervisor
      -> writer (retry path)
      -> finalize (success path)
  -> END
```

## 3.2 Core Subsystems

- `src/langgraph_platform/state`: typed workflow contracts (`WorkflowState`, metadata, routing, citations, token usage)
- `src/langgraph_platform/agents`: enterprise agent registry (20 agent definitions)
- `src/langgraph_platform/engine`: graph nodes, routing, orchestration, runtime
- `src/langgraph_platform/tools`: pluggable tool registry + built-in tools
- `src/langgraph_platform/memory`: SQLite persistence + Chroma vector store
- `src/langgraph_platform/rag`: ingest/chunk/retrieve pipeline
- `src/langgraph_platform/api`: FastAPI endpoints
- `src/langgraph_platform/ui`: Streamlit dashboard and graph visualization
- `src/langgraph_platform/cli`: operational CLI commands
- `src/langgraph_platform/exporters`: report export to MD/HTML/PDF/JSON

## 3.3 Agent Organization

Implemented agents:

- Planner Agent
- Research Agent
- Technical Research Agent
- Web Search Agent
- Documentation Agent
- GitHub Agent
- RAG Agent
- Memory Agent
- Knowledge Manager
- Business Analyst
- Data Analyst
- Financial Analyst
- Technical Writer
- Report Writer
- Fact Checker
- Reflection Agent
- Critic Agent
- QA Agent
- Citation Agent
- Supervisor Agent

## 4. API Surface

Implemented endpoints:

- `POST /chat`
- `POST /workflow`
- `GET /graph`
- `GET /agents`
- `POST /tasks`
- `GET /memory`
- `POST /reports`
- `POST /knowledge`
- `POST /search`
- `GET /analytics`
- `GET /metrics`
- `GET /health`
- `GET /mcp/capabilities`
- `POST /mcp/call`

## 5. Dashboard Surface

Implemented Streamlit pages:

- Dashboard
- Workflow Graph
- Live Execution
- Agents
- Shared State
- Memory
- Knowledge Base
- Reports
- Analytics
- Configuration

## 6. CLI Surface

Implemented commands:

```bash
langgraph-platform run "<request>"
langgraph-platform graph
langgraph-platform state --limit 10
langgraph-platform memory "<query>" --limit 10
langgraph-platform report <workflow_id> <markdown_path>
langgraph-platform doctor
langgraph-platform dashboard
```

## 7. Zero-to-Hero Setup

## 7.1 Prerequisites

- Linux (validated on Ubuntu)
- Python 3.12+ (runtime here: 3.14.4)
- `uv`

Optional for full local LLM execution:

- Ollama daemon running at `http://localhost:11434`
- pulled models configured in `configs/config.yaml`

## 7.2 Install

```bash
uv venv .venv
source .venv/bin/activate
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

## 7.3 Build

```bash
UV_CACHE_DIR=/tmp/uv-cache uv build
```

Build artifacts:

- `dist/langgraph_multi_agent_platform-0.1.0.tar.gz`
- `dist/langgraph_multi_agent_platform-0.1.0-py3-none-any.whl`

## 7.4 Run Tests

```bash
UV_CACHE_DIR=/tmp/uv-cache .venv/bin/pytest tests -q
```

Observed result in this run: `7 passed`

## 7.5 Start API

```bash
UV_CACHE_DIR=/tmp/uv-cache .venv/bin/uvicorn apps.fastapi.main:app --host 127.0.0.1 --port 8010
```

## 7.6 Start Dashboard

```bash
UV_CACHE_DIR=/tmp/uv-cache .venv/bin/streamlit run apps/streamlit/app.py --server.address 127.0.0.1 --server.port 8510 --server.headless true --server.fileWatcherType none
```

## 7.7 Run CLI End-to-End

```bash
UV_CACHE_DIR=/tmp/uv-cache .venv/bin/langgraph-platform run "Generate an enterprise report about multi-agent workflow design with citations and risks"
```

Strict live-model mode (fails if Ollama cannot serve responses):

```bash
REQUIRE_LIVE_LLM=1 OLLAMA_TIMEOUT_SECONDS=120 UV_CACHE_DIR=/tmp/uv-cache .venv/bin/langgraph-platform run "Produce a concise enterprise update on this platform status with citations and confidence."
```

## 8. Verified Artifacts (Generated in Real Run)

Directory: `artifacts/reports/`

API outputs:

- `api_health.json`
- `api_agents.json`
- `api_graph.json`
- `api_metrics.json`
- `api_analytics.json`
- `api_workflow.json`
- `api_chat.json`
- `api_memory.json`
- `api_tasks_pause.json`
- `api_tasks_resume.json`
- `api_knowledge.json`
- `api_search.json`
- `api_reports_export.json`

CLI outputs:

- `cli_run.txt`
- `cli_run_live_llm.txt`
- `cli_graph.txt`
- `cli_state.txt`
- `cli_memory.txt`
- `cli_doctor.txt`
- `cli_report.txt`

Other generated outputs:

- `streamlit_home.html`
- `api_workflow_report.md`
- `verification_summary.json`
- `artifacts/langgraph_platform.db`
- `artifacts/chroma/chroma.sqlite3`
- `artifacts/reports/wf_e737a7ad6254.{md,html,pdf,json}`

Screenshots (`assets/screenshots/`):

- `fastapi_swagger.png`
- `streamlit_dashboard.png`
- `streamlit_workflow_graph.png`
- `streamlit_live_execution.png`
- `streamlit_shared_state.png`
- `streamlit_memory.png`
- `streamlit_knowledge_base.png`
- `streamlit_reports.png`
- `streamlit_analytics.png`
- `streamlit_configuration.png`
- `streamlit_agents.png`

## 9. Configuration

Main config file:

- `configs/config.yaml`

Config domains:

- models
- graph
- prompts
- retries
- routing
- memory
- embeddings
- chunking
- tools
- analytics
- mcp
- hitl

## 10. Runtime Behavior When Model Endpoint Is Unavailable

The engine now includes a deterministic fallback path in `src/langgraph_platform/engine/llm.py`.

Behavior:

- tries configured model chain first
- on endpoint connectivity failure, falls back to local deterministic generation
- keeps workflow execution non-blocking and result-generating

This makes end-to-end execution resilient in restricted or offline environments while still supporting live Ollama inference when available.

## 11. Troubleshooting

If API workflow calls are slow/hanging:

- verify Ollama availability (`ollama ps`)
- check model endpoint at `http://localhost:11434`
- if unavailable, fallback mode will be used automatically

If local networking is sandbox-restricted in your agent shell:

- run service checks with required permissions, or run directly in host terminal

## 12. Project Status

Project is functionally complete for the requested enterprise workflow scope:

- multi-agent orchestration
- typed state
- routing and retries
- memory + RAG
- API + dashboard + CLI
- report exports
- verified real execution artifacts
- strict live Ollama execution path verified
- dashboard and API screenshot evidence captured
