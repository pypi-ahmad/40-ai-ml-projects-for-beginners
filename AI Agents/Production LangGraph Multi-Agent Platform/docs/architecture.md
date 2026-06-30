# Architecture Notes

## Core subsystems

- `engine`: LangGraph nodes + routing + retries + consensus + reflection loop
- `state`: shared Pydantic state contracts
- `agents`: 20 agent definitions + prompt contracts
- `tools`: pluggable tool registry with built-in enterprise tools
- `memory`: SQLite operational memory + Chroma semantic memory
- `rag`: ingestion and retrieval pipelines
- `api`: FastAPI service layer
- `ui`: Streamlit dashboard pages
- `cli`: Typer command surface
- `monitoring/analytics`: observability and charts
- `mcp`: external server client + internal exposed tool adapter

## State lifecycle

1. Create initial state from request
2. Planner decomposes request and sets routing
3. Parallel retrieval nodes collect evidence
4. Knowledge merge and consensus reasoning build synthesis context
5. Writer produces report draft
6. Verification loop (fact check -> reflection -> critic -> QA)
7. Supervisor decides retry or finalize
8. Persist artifacts and telemetry

## Persistence

### SQLite

- `workflow_runs`
- `agent_outputs`
- `tool_calls`
- `graph_states`

### Chroma

Single `knowledge` collection for ingestion/retrieval with source metadata.

## Reliability controls

- confidence threshold gate
- bounded retries
- model fallback chain
- structured output parsing with fallback behavior
- persisted node snapshots for audit and replay
