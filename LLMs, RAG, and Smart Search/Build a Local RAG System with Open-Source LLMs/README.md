# Project #14 — Local RAG System with Open-Source LLMs

Production-style, fully local Retrieval-Augmented Generation (RAG) stack built with Ollama + ChromaDB.  
No OpenAI/Anthropic/Gemini/cloud inference calls.

## Executive Summary

This project delivers an enterprise-style local AI assistant that:

- runs embeddings and generation locally through Ollama
- ingests private multi-format corpora (PDF/Markdown/TXT/man)
- persists vector indexes in ChromaDB across restarts
- supports incremental indexing (no full re-embed on unchanged data)
- exposes CLI + Streamlit UX + evaluation + audit commands
- includes tests, notebooks, reports, and verification artifacts

## Local AI Fundamentals

Why local inference:

- Privacy: documents stay on local infrastructure
- Compliance: easier control for regulated environments
- Latency: no network round-trip to hosted model APIs
- Cost: predictable hardware/runtime cost model
- Offline capability: usable without internet connectivity
- Ownership: full control over model/runtime/index lifecycle

## Architecture

```text
Query
 -> Embed (qwen3-embedding:4b)
 -> Chroma similarity search
 -> Top-K chunks (+ metadata filters)
 -> Grounded prompt construction
 -> Generate answer (qwen3.5:4b)
 -> Citations + timing + diagnostics
```

Core modules:

- `src/local_rag/loaders.py`: PDF/MD/TXT loading + metadata normalization
- `src/local_rag/splitter.py`: RecursiveCharacterTextSplitter chunking
- `src/local_rag/embeddings.py`: Ollama embedding client + normalization
- `src/local_rag/vectordb.py`: persistent Chroma lifecycle + integrity checks
- `src/local_rag/indexing.py`: incremental indexing service + manifest updates
- `src/local_rag/rag.py`: retrieval + prompting + generation orchestration
- `src/local_rag/evaluator.py`: retrieval/response metrics
- `src/local_rag/llm_judge.py`: local LLM-as-a-judge scoring (`granite4.1:3b`)
- `streamlit_app/app.py`: interactive UI for ingestion/query/inspection

## Models and Components

- Embedding model: `qwen3-embedding:4b`
- Generation model: `qwen3.5:4b`
- Judge model: `granite4.1:3b` (optional but supported)
- Vector DB: `chromadb` `PersistentClient`

## Dataset and Corpus Strategy

Domain: Linux/system documentation corpus with heterogeneous technical content.

Bootstrap sources:

- Linux man-pages archive
- systemd documentation
- Linux kernel docs
- GNU manuals (PDF + TXT)

Corpus profiles:

- `full`: large corpus (`data/documents`)
- `quickstart`: deterministic sampled subset (`data/documents_quickstart`)

Corpus quality gates enforce mixed formats and minimum file counts.

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv sync --all-groups
cp .env.example .env
```

Start Ollama and pull required models:

```bash
ollama serve
ollama pull qwen3-embedding:4b
ollama pull qwen3.5:4b
ollama pull granite4.1:3b
```

## CLI Workflow

### 1) Environment and local-only checks

```bash
uv run python -m local_rag doctor
uv run python -m local_rag validate-local
```

### 2) Bootstrap corpus and quickstart subset

```bash
uv run python -m local_rag bootstrap --build-quickstart
uv run python -m local_rag corpus-report --profile full
uv run python -m local_rag corpus-report --profile quickstart
```

### 3) Build/update persistent index

```bash
uv run python -m local_rag ingest --profile full
uv run python -m local_rag ingest --profile quickstart --rebuild
uv run python -m local_rag validate-index --profile full
```

### 4) Query

```bash
uv run python -m local_rag query "How does ACPI work?" --top-k 5 --profile full
uv run python -m local_rag query "What is ACPI?" --stream --source-type markdown
```

### 5) Evaluation and failure analysis

```bash
uv run python -m local_rag generate-eval-set --max-examples 200 --profile full
uv run python -m local_rag compile-eval-set
uv run python -m local_rag evaluate --profile full
uv run python -m local_rag run-experiments --profile full
uv run python -m local_rag failures
```

### 6) Local judge

```bash
uv run python -m local_rag judge \
  --query "What is ACPI?" \
  --answer "..." \
  --context "..."

uv run python -m local_rag judge-batch --input-path outputs/reports/judge_input.jsonl
```

### 7) Streamlit UI

```bash
uv run streamlit run streamlit_app/app.py
```

## Streamlit Features

- Full/quickstart profile switch
- Upload documents (PDF/TXT/MD)
- Build, rebuild, clear, and inspect index
- Query with top-k, metadata filters, and streaming
- Retrieval diagnostics: chunk text, metadata, scores, citations
- Latency breakdown and health panel

## Persistence and Incremental Indexing

Storage:

- Chroma DB path: `vectordb/chroma/`
- Profile-scoped collections: `linux_local_rag_full`, `linux_local_rag_quickstart`
- Profile-scoped manifests:
  - `vectordb/index_manifest_full.json`
  - `vectordb/index_manifest_quickstart.json`

Behavior:

- unchanged corpus + compatible config: skip re-embedding
- changed/new documents: embed/upsert only changed units
- removed documents: delete vectors by `doc_id`
- changed model/chunk/manifest compatibility fields: full rebuild

## Evaluation

Retrieval metrics:

- Precision@K
- Recall@K
- MRR
- NDCG
- Average retrieval latency

Response metrics:

- generation latency
- answer length
- citation count

LLM-as-a-judge rubric:

- correctness
- groundedness
- completeness
- faithfulness
- conciseness

## Performance and Profiling

Tracked via reports/benchmarks:

- embedding time
- indexing time
- retrieval time
- generation time
- total latency
- memory and disk footprint
- vector count

## Notebooks (Educational Track)

- `notebooks/01_local_rag_enterprise_foundations.ipynb`
- `notebooks/02_chunking_embeddings_retrieval_eval.ipynb`
- `notebooks/03_local_rag_evaluation_failures.ipynb`

## Testing

```bash
uv run pytest -v
uv run pytest -v -m integration
uv run ruff check .
```

## Reproducibility

Run end-to-end verification script (host environment):

```bash
bash scripts/verify_full_pipeline.sh
```

This script runs preflight, corpus, indexing, query, evaluation, judge/failure checks, and a real Streamlit run health validation.
Default mode runs a real `quickstart` profile for fast, repeatable local validation.
For full-corpus execution:

```bash
VERIFY_PROFILE=full bash scripts/verify_full_pipeline.sh
```

For slower hosts, a practical full-corpus indexing run can be executed directly with
larger chunks and higher embed batching (still local, still full corpus):

```bash
RAG_EMBEDDING_BATCH_SIZE=256 uv run python -m local_rag ingest --profile full --chunk-size 6144 --chunk-overlap 768
```

Audit report:

- `FINAL_PROJECT_VERIFICATION_REPORT.md`

## Scaling Considerations (10K / 100K / 1M Docs)

- 10K docs: current architecture workable on single node with batching and profile-aware indexes.
- 100K docs: add ingestion workers, shard collections, and periodic index compaction.
- 1M docs: move to distributed retrieval stack (sharded vector service + reranking tier + async pipelines).

Recommended upgrades for large scale:

- asynchronous ingestion queue
- metadata partitioning strategy
- hybrid retrieval (BM25 + dense)
- local reranker stage
- observability + SLO dashboards

## Project #13 vs #14

| Dimension | Project #13 | Project #14 |
|---|---|---|
| Objective | Learn RAG fundamentals | Build production-style local RAG |
| Corpus | Small/tutorial | Large mixed-format technical corpus |
| Storage | Often in-memory | Persistent Chroma + manifest lifecycle |
| Evaluation | Basic checks | Retrieval + response + judge + failure audits |
| Runtime | May use cloud APIs | Fully local inference/retrieval |

## Lessons Learned

- persistence and manifest compatibility are critical for practical RAG cost/latency
- grounding policy needs strict prompt contracts + citation validation
- evaluation must separate retrieval quality from generation quality
- local-only systems still require robust operational checks and reproducibility tooling

## Future Improvements

- local reranker integration
- conflict-aware citation consolidation
- richer UI dashboards for trend analysis
- async ingestion and backpressure handling
- benchmark suite for model quantization trade-offs
