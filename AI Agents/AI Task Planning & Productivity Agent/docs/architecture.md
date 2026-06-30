# Architecture — AI Task Planning & Productivity Agent

## Core workflow

1. **Input Parser / Task Extractor** normalize messy sources into structured tasks.
2. **Priority Analyzer** scores tasks with switchable frameworks.
3. **Dependency Planner** builds DAG and validates cycles.
4. **Schedule Optimizer** creates time blocks with deadline-aware heuristics.
5. **Validator** flags risk/conflicts.
6. **Reflection + Recommendation Agents** derive improvement actions.
7. **Memory Manager** persists structured + semantic memory.
8. **Reporter** builds user-facing report and analytics snapshot.

## LangGraph nodes

`Planner -> Scheduler -> Validator -> Reflection -> Memory -> Reporter`

## Storage

- SQLite: operational persistence (users, tasks, plans, preferences, reflections)
- ChromaDB: semantic retrieval
- DuckDB: analytics aggregation
- MLflow: run metric tracking

## Interfaces

- FastAPI for APIs (`/plan`, `/replan`, `/tasks`, `/calendar`, `/history`, `/search`, `/preferences`, `/report`, `/health`)
- Streamlit for dashboard
- Typer CLI for automation

## Integrations

- Calendar: ICS + local enabled, Google calendar adapter stub
- External systems: production connector contracts + live-ready stubs
- Tools: datetime, file reader, calculator, weather, web search, notes, reminders, timer
