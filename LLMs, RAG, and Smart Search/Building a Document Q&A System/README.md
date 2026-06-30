# Project #15 — Building a Production-Grade Document Q&A System

Production-quality, local-first **Document Question Answering (RAG)** system for multi-document enterprise intelligence.

This project upgrades prior roadmap projects into a complete document assistant that supports:
- Multi-document ingestion and lifecycle management
- Persistent vector + lexical indexing
- Vector / BM25 / hybrid retrieval
- Citation-aware grounded generation
- Retrieval + generation + judge evaluation
- Streamlit UI for document ops, QA, and evidence inspection

All runtime inference is local with Ollama. No OpenAI/Anthropic/Gemini APIs.

## 1) Stack

- Python: `3.12.10` (target)
- Package manager: `uv`
- Embeddings: `qwen3-embedding:4b`
- Generator: `qwen3.5:4b`
- Judge: `granite4.1:3b`
- Vector DB: ChromaDB (`PersistentClient`)
- Keyword retrieval: BM25 (local persisted lexical index)
- Hybrid retrieval: weighted reciprocal rank fusion (RRF)
- UI: Streamlit

## 2) Architecture

Pipeline stages:

`Question -> Embedding -> Retriever (Vector/BM25/Hybrid) -> Hybrid Rank -> Context Builder -> Prompt Template -> LLM -> Citations -> Evaluation`

Core modules (`src/local_rag/`):
- `loaders.py`: PDF/TXT/Markdown loaders + metadata extraction
- `document_manager.py`: document catalog, delete/update, version tracking
- `splitter.py`: recursive chunking with configurable size/overlap
- `embeddings.py`: Ollama embedding client (`qwen3-embedding:4b`)
- `vectordb.py`: Chroma persistence, filtering, CRUD
- `lexical.py`: persistent BM25 index
- `retriever.py`: vector, keyword, hybrid retrieval
- `prompts.py`: strict/citation/legal/technical templates
- `rag.py`: orchestration + citations + streaming
- `evaluator.py`: retrieval metrics + generation metrics + groundedness proxies
- `llm_judge.py`: local rubric judge (`granite4.1:3b`)
- `benchmarking.py`: latency/throughput benchmark runner
- `cli.py`: end-to-end operational commands

## 3) Corpus Strategy

The bootstrap pipeline collects a realistic mixed corpus (technical/policy/research/finance) from public sources and archive extracts, including:
- GNU/Linux technical manuals and docs (PDF/TXT/Markdown-rich archives)
- Public policy/regulatory PDFs (NIST, public governance documents)
- Research PDFs (public ML papers)
- Public finance reports

Formats supported end-to-end:
- `PDF`
- `Markdown`
- `TXT`

Why this corpus:
- Reflects enterprise multi-domain knowledge settings
- Supports cross-document comparison questions
- Exercises metadata filtering and citation provenance

## 4) Retrieval and QA Features

Implemented:
- Top-K retrieval (`1/3/5/10`)
- Metadata filtering (`source_type`, `section`, `domain`)
- Strategy switch: `vector`, `keyword`, `hybrid`
- Hybrid RRF fusion with configurable weighting
- Multi-turn conversation history in prompt construction
- Prompt template variants:
  - `strict_grounding`
  - `citation_focus`
  - `enterprise_qa`
  - `legal_qa`
  - `technical_qa`
  - `unknown_safe`

Citation payload per answer includes:
- `document_name`
- `source_path`
- `page_number`
- `chunk_id`
- `similarity_score`
- `evidence_text`

## 5) Evaluation

### Retrieval metrics
- Precision@K
- Recall@K
- MRR
- NDCG
- Retrieval latency

### Generation metrics
- BLEU
- ROUGE-1 / ROUGE-L
- METEOR
- BERTScore (local cached model)
- Answer length
- Groundedness (context overlap proxy)
- Faithfulness (groundedness threshold proxy)
- Context Precision / Context Recall
- Answer Relevancy
- Hallucination rate (low-groundedness proxy)

### LLM-as-a-Judge
Using local `granite4.1:3b`:
- Correctness
- Groundedness
- Completeness
- Faithfulness
- Conciseness
- Citation quality

## 6) Streamlit App

Run:

```bash
uv run streamlit run streamlit_app/app.py
```

Main tabs:
- `Document Management`: upload/delete/catalog/bootstrapping
- `Indexing`: chunk config, incremental indexing, rebuild, collection stats
- `Q&A`: strategy/prompt controls, streaming answers, citations, retrieved chunks
- `Analytics`: corpus and conversation summaries

## 7) Setup and Usage

### Install

```bash
uv venv .venv
source .venv/bin/activate
uv sync --all-groups
cp .env.example .env
```

### Pull local models

```bash
ollama serve
ollama pull qwen3-embedding:4b
ollama pull qwen3.5:4b
ollama pull granite4.1:3b
```

### CLI workflow

```bash
uv run python -m local_rag bootstrap
uv run python -m local_rag ingest --chunk-size 768 --chunk-overlap 100
uv run python -m local_rag query "Which document discusses encryption?" --strategy hybrid --top-k 5
uv run python -m local_rag evaluate --strategy hybrid
uv run python -m local_rag benchmark
uv run python -m local_rag failures
```

### Full run script

```bash
bash scripts/run_end_to_end.sh
```

### Notebook execution

```bash
uv run python scripts/execute_notebook.py
```

## 8) Tests and Quality

Run tests:

```bash
uv run pytest -q
```

Run lint:

```bash
uv run ruff check src tests streamlit_app scripts
```

Current local verification in this implementation pass:
- `pytest`: pass (`1 skipped` integration)
- `ruff`: pass

## 9) Project Progression (#13 -> #14 -> #15)

| Dimension | Project #13 | Project #14 | Project #15 (this) |
|---|---|---|---|
| Focus | RAG basics | Local production-style RAG | Enterprise document intelligence |
| Corpus | Small/tutorial | Large local technical | Mixed enterprise multi-domain |
| Retrieval | Primarily vector basics | Persistent vector retrieval | Vector + BM25 + hybrid RRF |
| Document lifecycle | Minimal | Ingestion + incremental index | Upload/delete/update/version tracking |
| Evidence | Basic | Citation-aware | Rich citation payload + inspector |
| Evaluation | Intro-level | Retrieval + judge baseline | Retrieval + generation + judge + benchmark suite |
| UI | Demo | Streamlit local RAG | Multi-tab document operations + QA analytics |

## 10) Repository Layout

```text
.
├── configs/
├── data/
├── notebooks/
│   └── 01_document_qa_zero_to_hero.ipynb
├── outputs/
├── scripts/
├── src/local_rag/
├── streamlit_app/
├── tests/
└── vectordb/
```

## 11) Limitations

- Full heavy-corpus benchmarking runtime depends on machine resources and local model throughput.
- BERTScore may trigger large local model/checkpoint downloads on first run.
- Some quality metrics use pragmatic local proxies (groundedness/faithfulness/context metrics) and should be complemented with human audits for high-stakes production use.

## 12) References

- ChromaDB docs
- Ollama docs
- BM25 / reciprocal rank fusion literature
- NIST public publications
- GNU/Linux documentation archives
