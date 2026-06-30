# Hybrid Real-Time AI Research Assistant Architecture

## High-Level Flow

```text
User Question
  -> Intent Detection
  -> Router (local / web / hybrid)
  -> Retriever(s)
  -> Reranker
  -> Context Builder
  -> LLM Generation
  -> Judge
  -> Response + Citations
```

## LangGraph Nodes

- `intent_detection`
- `local_retrieve`
- `web_retrieve`
- `hybrid_retrieve`
- `rerank`
- `context_builder`
- `generation`
- `judge`
- `error_recovery`
- `response`

## Storage

- ChromaDB persistent collections (`vectordb/chroma`)
- Manifest for incremental indexing and versioning (`vectordb/index_manifest_<profile>.json`)
- Reports/metrics (`outputs/reports`, `outputs/benchmarks`)
- Web cache (`data/web_cache`)

## Core Guarantees

- Local-first execution for embeddings, indexing, retrieval, and generation
- Grounded generation with explicit fallback when evidence is insufficient
- Source-level citations with chunk IDs and confidence values
- Support for local, web, and hybrid retrieval routes
