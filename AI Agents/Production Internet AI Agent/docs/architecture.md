# Production Internet AI Agent Architecture

## Overview

This platform combines LangGraph multi-agent reasoning, internet retrieval, semantic memory, verification loops, and multi-surface interfaces (FastAPI, Streamlit, CLI, MCP).

## Core Runtime

1. User query enters `InternetAgentWorkflow`.
2. User Intent + Planner classify problem and tool strategy.
3. Search Decision agent decides local-only vs internet retrieval.
4. Search + Web Extraction ingest and clean evidence.
5. Summarization drafts answer.
6. Verification checks confidence/conflicts and may trigger re-search.
7. Memory persists conversation/documents/tool history.
8. Reflection improves response quality.
9. Report agent emits export-ready payload.

## Storage

- SQLite (`artifacts/memory.db`) for history, cache, tool logs, reports.
- ChromaDB (`artifacts/chroma`) for semantic memory and retrieved chunk vectors.

## Interfaces

- FastAPI: `/chat`, `/search`, `/browse`, `/history`, `/memory`, `/report`, `/health`, `/metrics`.
- Streamlit multipage dashboard.
- Typer CLI (`internet-agent ...`).
- MCP-compatible stdio server (`internet-agent-mcp`).

## Observability

- Structured logs via Loguru.
- In-memory metrics store for tool/retrieval/agent latency and counters.
- MLflow experiment tracking for chat runs.
- System monitoring for CPU/RAM/GPU.
