# Architecture

## 1) End-to-End Flow
```mermaid
flowchart LR
    A[Google Sheets API] --> B[Ingestion Layer]
    B --> C[Cache + State Store]
    C --> D[Quality Validation]
    D --> E[Cleaning Engine]
    E --> F[Deterministic Analytics]
    F --> G[Visualization Engine]
    F --> H[LLM Insight Engine]
    H --> I[Judge Evaluation]
    G --> J[Report Generator]
    H --> J
    J --> K[Exports: MD/HTML/PDF/XLSX/PPTX]
    J --> L[Optional Google Sheets Write-back]
    F --> M[Conversational Tool Router]
    M --> H
```

## 2) Authentication Path
```mermaid
sequenceDiagram
    participant U as User
    participant S as Service Account
    participant G as Google APIs
    U->>S: Provide JSON key via env path
    S->>G: Request token with scopes
    G-->>S: Access token
    S->>G: Read sheets / metadata
    G-->>S: Worksheet records
```

## 3) Conversational Analytics Pipeline
```mermaid
flowchart TD
    Q[User question] --> R[Intent Router]
    R --> T[Deterministic Data Tools]
    T --> E[Evidence Packet]
    E --> L[Local LLM Narrative]
    L --> A[Answer + Recommendations]
    A --> H[History Store]
```

## 4) Write-back Safety
- Default behavior: create new worksheet.
- Overwrite requires explicit `overwrite=true`.
- Source worksheets never modified by default.
