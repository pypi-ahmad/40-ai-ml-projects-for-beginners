# Project #19: Production-Grade Semantic Search Engine

Production-ready semantic search engine with ChromaDB, FAISS comparison, BM25 hybrid retrieval, RRF fusion, MMR diversification, cross-encoder reranking, evaluation pipelines, and multi-page Streamlit interface.

## Highlights

- 50k-doc corpus pipeline (`khalidalt/HuffPost`) with optional URL enrichment.
- Multi-format document support: TXT, Markdown, PDF, CSV, JSON, HTML, DOCX (recursive loading).
- Chunking experiments: recursive, token, semantic with configurable size/overlap.
- Embedding models:
  - `BAAI/bge-small-en-v1.5` (primary)
  - `all-MiniLM-L6-v2`
  - `nomic-embed-text` (Ollama)
- Retrieval stack:
  - Semantic vector search (Chroma)
  - Lexical BM25 search
  - Hybrid fusion via Reciprocal Rank Fusion (RRF)
  - MMR diversification
  - Metadata filtering
- Reranking with `BAAI/bge-reranker-base`.
- Evaluation metrics: Precision@K, Recall@K, MRR, NDCG, latency.
- LLM-assisted judging with `granite4.1:3b`.
- Streamlit pages: Home, Index Builder, Documents, Search, Analytics, Benchmarks, Settings.

## Repository Structure

- `src/semantic_search/`: package source.
- `config/default.yaml`: project configuration.
- `scripts/`: pipeline scripts.
- `tests/`: unit tests.
- `notebooks/semantic_search_zero_to_hero.ipynb`: educational notebook.
- `docs/architecture.md`: architecture diagram and interfaces.

## Installation (uv + Python 3.12.10)

```bash
UV_CACHE_DIR=/tmp/uv-cache uv venv --python 3.12.10 .venv
source .venv/bin/activate
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev
```

## Ollama Requirements (hard requirement)

Required models:

- `nomic-embed-text`
- `granite4.1:3b`
- `qwen3.5:4b` (optional feature path but auto-provisioned)

```bash
uv run semantic-search check-models
```

## End-to-End Run

```bash
./scripts/run_full_pipeline.sh
```

Equivalent manual steps:

```bash
uv run semantic-search ingest --source huggingface
uv run semantic-search chunk
uv run semantic-search index --model primary
uv run python scripts/generate_eval_queries.py
uv run semantic-search evaluate --mode hybrid
uv run semantic-search benchmark
uv run python scripts/failure_analysis.py
```

Incremental operations:

```bash
uv run semantic-search incremental-index --input-jsonl path/to/new_docs.jsonl
uv run semantic-search delete-docs --document-ids doc-1,doc-2
```

## Streamlit App

```bash
uv run semantic-search app
```

## Evaluation Dataset Policy

- Generates 150 weak-label queries with category balancing:
  - technical, general, reasoning, comparison, multi-document
- Manual audit required for final quality gate.
- Stored at `data/processed/evaluation_queries.jsonl`.

## Quality Gate

- Baseline: BM25 lexical retrieval.
- Requirement: hybrid+r rerank must show relative lift over baseline on key quality metric (`ndcg_at_10`) and maintain p95 latency target (<= 1500 ms on local CPU for configured benchmark setup).

## Security Controls

- Safe local file loading with extension allowlist.
- Hidden-file blocking option.
- Path-restricted ingestion.
- UI search rate limiting.
- Environment-variable-based configuration (`.env` support).

## Benchmarks and Reports

Outputs:

- `artifacts/reports/benchmark_report.json`
- `artifacts/reports/evaluation_results.json`
- `artifacts/reports/collection_manifest.json`
- `logs/search_events.jsonl`

## Notebook

`notebooks/semantic_search_zero_to_hero.ipynb` covers:

- Embeddings and vector space intuition
- ChromaDB vs FAISS
- BM25 + semantic hybrid retrieval
- MMR and reranking
- Evaluation metrics and interpretation

## Limitations

- URL enrichment depends on live source URLs and may partially fail.
- Weak labels require manual audit for production-grade trust.
- Full 10k-100k long-form corpus migration path is documented but current default corpus is HuffPost headlines+descriptions with optional enrichment.

## Future Work

- Cross-lingual retrieval and multilingual reranking.
- OCR/image caption ingestion path.
- Voice search + speech-to-text query frontend.
- GitHub repository indexing extension.
