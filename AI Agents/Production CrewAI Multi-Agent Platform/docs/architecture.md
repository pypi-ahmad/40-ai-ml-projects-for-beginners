# Architecture Notes

## Core Runtime
- FastAPI service is system-of-record for runs/tasks/memory.
- Streamlit + CLI act as clients to API.
- Planner creates dependency-safe DAG.
- DAG executor handles parallel tasks, retries, and dynamic delegation.

## Orchestration Layers
1. Planning: dynamic task decomposition.
2. HITL approval gate.
3. Execution: CrewAI role-specialized task execution + LLM generation.
4. Verification: fact-check + QA + reflection.
5. Confidence decision: consensus reroute when low confidence.
6. Reporting + persistence.

## Data Stores
- SQLite: plans, tasks, reports, approvals, tool calls, conversations.
- ChromaDB: semantic memory for retrieval and shared knowledge reuse.

## Integration Surfaces
- FastAPI endpoints for automation and UI clients.
- Typer CLI for local operations.
- Streamlit pages for observability.
- MCP internal/external adapters for tool interoperability.
