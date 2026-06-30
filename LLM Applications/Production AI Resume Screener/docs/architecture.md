# Architecture

## System Overview

```mermaid
flowchart LR
    U[Recruiter] --> UI[Streamlit Dashboard]
    U --> CLI[Typer CLI]
    U --> API[FastAPI]
    E[MCP Client] --> MCP[MCP Server]
    MCP --> API
    CLI --> SVC[ResumeAIService]
    UI --> SVC
    API --> SVC

    SVC --> ING[Ingestion Pipeline]
    ING --> OCR[OCR Engine]
    ING --> PARSE[Resume/JD Parser]
    PARSE --> MATCH[Matching Engine]
    MATCH --> RANK[Ranking]
    SVC --> RAG[RAG Assistant]
    RAG --> VDB[ChromaDB]

    SVC --> DB[(SQLite)]
    SVC --> VDB
    SVC --> REP[Report Generator]
    SVC --> INT[Interview Generator]
```

## Data Flow

1. Resume uploaded through API/UI/CLI.
2. Reader extracts text or routes to OCR.
3. Parser builds `ResumeParseResult` schema.
4. Candidate/resume persisted in SQLite.
5. Embeddings pushed into ChromaDB.
6. JD parser creates normalized requirement schema.
7. Matching engine computes weighted explainable score.
8. RAG retriever serves recruiter semantic queries.
9. Report engine exports artifacts (md/html/pdf/json).

## Storage

- SQLite tables: candidates, resumes, jobs, scores, interviews, reports, notes, processing_jobs, metrics.
- Chroma collections: resume_chunks, job_descriptions, candidate_summaries, projects, experience, interview_feedback.

## Explainability Contract

Each score output returns:

- matched skills
- missing skills
- per-component weighted scores
- evidence snippets
- confidence value
