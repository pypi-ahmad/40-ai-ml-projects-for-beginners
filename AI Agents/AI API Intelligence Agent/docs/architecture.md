# Architecture

## System Overview

```mermaid
flowchart LR
    U[User] --> C[CLI]
    U --> S[Streamlit]
    U --> F[FastAPI]
    C --> G[LangGraph Runtime]
    S --> G
    F --> G

    G --> L[Ollama LLM]
    G --> R[Connector Registry]
    G --> T[Tool Registry]
    G --> M[Memory Manager]
    G --> K[Cache Manager]
    G --> A[Analytics]
    G --> P[Report Generator]

    R --> APIs[Live APIs]
    M --> SQ[(SQLite)]
    M --> CH[(ChromaDB)]
    K --> KC[(SQLite/Redis Cache)]
    P --> AR[Report Artifacts]
```

## LangGraph Node Flow

```mermaid
flowchart LR
    planner --> router --> auth --> fetch --> validate --> reason --> report --> memory --> reflection
    reflection -->|retry| fetch
    reflection -->|end| done([Done])
```

## Data Contracts

- `AnalyzeRequest`: user query, model selection, APIs, cache/memory flags.
- `ConnectorResult`: normalized provider result with latency and errors.
- `AnalyzeResponse`: summary, insights, recommendations, sources, charts, artifacts.

## Reliability Controls

- Async HTTP with retry + exponential backoff.
- Provider-level graceful degradation on missing credentials.
- Cache-first for repeat calls.
- Reflection retry loop for retryable errors.
- Structured logs + runtime metrics + persistent history.
