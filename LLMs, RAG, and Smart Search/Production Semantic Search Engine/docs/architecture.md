# Architecture

```mermaid
flowchart LR
    A[Hugging Face HuffPost Corpus] --> B[Ingestion Pipeline]
    B --> C[Dedup + Hash + Metadata + Language]
    C --> D[Chunking Layer\nRecursive / Token / Semantic]
    D --> E[Embedding Layer\nBGE / MiniLM / Nomic]
    E --> F[ChromaDB Primary Index]
    E --> G[FAISS Comparison Index]
    H[BM25 Lexical Index] --> I[Hybrid Retrieval + RRF]
    F --> I
    G --> I
    I --> J[MMR Diversification]
    J --> K[Cross-Encoder Reranker]
    K --> L[Search API + Streamlit UI]
    L --> M[Analytics + Eval + Benchmarks]
```

## Key Interfaces

- `semantic_search.service.SemanticSearchService`: orchestration entrypoint.
- `semantic_search.cli`: CLI execution surface.
- `app.py`: Streamlit UI with pages: Home, Index Builder, Documents, Search, Analytics, Benchmarks, Settings.
- `config/default.yaml`: runtime configuration.

## Storage Layout

- `data/raw/`: source dataset snapshots.
- `data/processed/documents.jsonl`: canonical documents.
- `data/processed/chunks.jsonl`: chunked corpus.
- `artifacts/chroma/`: persistent Chroma collections.
- `artifacts/faiss/`: FAISS index + metadata sidecars.
- `artifacts/reports/`: benchmark and evaluation outputs.
- `logs/search_events.jsonl`: search telemetry.
