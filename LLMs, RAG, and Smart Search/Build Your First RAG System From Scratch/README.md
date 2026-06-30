# Build Your First RAG System From Scratch

Portfolio-grade, local-first, zero-to-hero RAG project built for serious beginners and early intermediate engineers.

This repository teaches and implements the complete RAG lifecycle:

- why hallucinations happen
- why retrieval matters
- how embeddings and vector databases work
- how to build a production-minded local RAG pipeline
- how to evaluate retrieval/generation rigorously
- how to audit groundedness and failure modes

All model inference runs locally via Ollama.

## Executive Summary

This project upgrades a basic tutorial RAG into a full engineering workflow with:

- split-aware, leakage-audited dataset pipeline
- multiple chunking strategies and fair retrieval comparison
- local embeddings (`qwen3-embedding:4b`) + ChromaDB indexing
- grounded generation (`qwen3.5:4b`) with abstention policy
- retrieval + generation + LLM-judge evaluation
- hallucination analysis (`No RAG` vs `RAG`)
- retrieval diagnostics and performance profiling
- transparent Gradio interface

## What Is RAG?

Retrieval-Augmented Generation (RAG) combines two systems:

1. Retrieval: fetch relevant external context.
2. Generation: answer using retrieved context.

Traditional LLM:
`query -> model weights -> answer`

RAG:
`query -> embedding -> vector search -> grounded prompt -> answer + citations`

RAG exists because model weights alone are not enough for reliable factual answers in dynamic or specialized domains.

## Why RAG Exists

### Hallucinations
Pure LLM responses may include plausible but unsupported claims.

### Knowledge limitations
Model weights are static after training and may be stale or incomplete.

### Context window constraints
Even good models fail if relevant evidence is not present in prompt context.

### Retrieval as control
Retrieval constrains generation to known context and enables evidence inspection.

## Architecture

Core runtime flow:

1. User query
2. Query embedding (`qwen3-embedding:4b`)
3. ChromaDB similarity search
4. Context assembly with citations
5. Prompt construction with grounding constraints
6. Generation (`qwen3.5:4b`, `think=False`)
7. Structured response + diagnostics

### Repository modules

- `src/rag_system/data.py`: split-aware dataset ingestion, persistence, leakage audit
- `src/rag_system/chunking.py`: fixed/recursive/semantic/parent-child chunking
- `src/rag_system/embeddings.py`: local embedding engine
- `src/rag_system/retrieval.py`: ChromaDB indexing/retrieval/filtering/diagnostics helpers
- `src/rag_system/advanced_retrieval.py`: query expansion + multi-query + reranking
- `src/rag_system/generation.py`: grounded generation + abstention policy + baseline
- `src/rag_system/metrics.py`: retrieval and generation metrics
- `src/rag_system/evaluation.py`: end-to-end evaluation + LLM-as-a-judge
- `src/rag_system/diagnostics.py`: embedding/index/retrieval diagnostic reports
- `app/gradio_app.py`: interactive transparency interface

## Dataset Choice and Audit

Primary dataset: **Hugging Face `rajpurkar/squad_v2`**.

Why it was selected:

- high-quality public QA benchmark
- non-trivial scale (tens of thousands of contexts/questions)
- supports retrieval and generation evaluation with labels
- includes unanswerable cases for robustness analysis

### Split policy (leakage-aware)

- Retrieval corpus documents: `train + validation` (document-only corpus)
- Evaluation queries: `validation` questions

Why this design: evaluation questions must retrieve their supporting documents, so validation documents are included in corpus. We avoid contamination by never training on evaluation labels and by auditing split policy + gold-reference integrity.

Artifacts include leakage evidence:

- `data/processed/split_manifest.json`
- `data/eval/leakage_audit.json`

## Chunking

Implemented chunking strategies:

- fixed-size chunking
- recursive chunking
- semantic chunking
- parent-child chunking

Each strategy is benchmarked under the same conditions (same query set, same `top_k`, same metric suite).

## Embeddings

Embedding model: `qwen3-embedding:4b` via Ollama.

Concepts covered in code/notebooks:

- vector representations
- cosine similarity and distance behavior
- nearest-neighbor retrieval
- embedding integrity checks (dimension, NaN/Inf, duplication, batch consistency)

## ChromaDB

Primary vector database: **ChromaDB**.

Usage includes:

- persistent local collections
- chunk + metadata upserts
- top-k similarity search
- optional metadata filtering
- collection diagnostics

Why ChromaDB here:

- local-first workflow
- low operational overhead for beginners
- clear API for educational vector retrieval

Comparison context is included for FAISS/Pinecone/Weaviate/Qdrant.

## Retrieval Pipeline

Implemented retrieval capabilities:

- similarity search
- top-k retrieval
- metadata filter path
- score distribution summary
- retrieval failure bucket classification

Failure buckets include:

- `no_hit`
- `low_score`
- `hit_top1`
- `hit_not_top1`
- `wrong_document`

## Prompt Engineering and Generation

Prompting emphasizes grounding constraints:

- use retrieved context only
- cite evidence as `[1], [2], ...`
- abstain when evidence is insufficient

Generation path:

`Retrieve -> Augment -> Generate`

Abstention behavior is explicit and test-covered.

## Evaluation Framework

### Retrieval metrics

- Precision@K
- Recall@K
- F1@K
- MRR
- NDCG

### Generation metrics

- Exact Match
- BLEU
- ROUGE-1 / ROUGE-L
- METEOR
- BERTScore

### LLM-as-a-Judge

Judge model: `granite4.1:3b`

Scored dimensions:

- relevance
- correctness
- groundedness
- completeness
- faithfulness

### Hallucination analysis

Compares:

- baseline LLM answer without retrieval
- grounded RAG answer with retrieval

with per-query groundedness/faithfulness deltas.

## Advanced Retrieval

Implemented:

- query expansion
- multi-query retrieval (RRF merge)
- lexical-semantic reranking

Advanced path is benchmarked against baseline retrieval so complexity is justified by evidence.

## Visualizations

Generated figures include:

- RAG architecture diagrams
- LLM vs RAG diagram
- document length distributions
- retrieval metric charts
- generation metric charts
- hallucination delta distributions

## Interactive Interface

Gradio app (`app/gradio_app.py`) provides:

- user query input
- retrieved chunks with scores/distances
- retrieved context panel
- RAG vs no-RAG comparison
- diagnostics block (abstention + top score)

## Notebooks (Zero-to-Hero Curriculum)

1. `01_rag_foundations.ipynb`
2. `02_modern_rag_flow.ipynb`
3. `03_dataset_eda.ipynb`
4. `04_chunking_deep_dive.ipynb`
5. `05_embeddings_deep_dive.ipynb`
6. `06_chromadb_retrieval.ipynb`
7. `07_prompt_and_generation.ipynb`
8. `08_evaluation.ipynb`
9. `09_advanced_retrieval_and_faiss_appendix.ipynb`
10. `10_gradio_demo_and_production_notes.ipynb`

Each notebook follows an explain-first structure:

- definition
- theory
- motivation
- architecture
- implementation and code explanation
- best practices
- common mistakes

## Setup (Local)

### Prerequisites

- Linux
- Python `3.12.10`
- `uv`
- local Ollama server (`http://127.0.0.1:11434`)
- pulled models:
  - `qwen3-embedding:4b`
  - `qwen3.5:4b`
  - `granite4.1:3b`

### Environment

```bash
uv venv .venv --python 3.12.10
source .venv/bin/activate
uv sync
```

## Runbook

### Generate notebooks

```bash
.venv/bin/python scripts/generate_notebooks.py
```

### Canonical Real Audit (primary)

```bash
.venv/bin/python scripts/run_final_audit.py --profile max_depth
```

This command is the canonical source for published portfolio metrics and artifacts.

### Practical Real Audit (bounded, still all-real)

```bash
.venv/bin/python scripts/run_final_audit.py \
  --profile fast \
  --chunking-docs 300 \
  --chunking-queries 60 \
  --advanced-queries 20 \
  --retrieval-limit 60 \
  --generation-limit 12 \
  --judge-limit 10 \
  --hallucination-limit 8 \
  --strict-gates
```

Use this during rapid iteration when max-depth runtime is too long; it still runs fully local real inference with Ollama and enforces strict audit gates.

### Full project run (real, no smoke shortcuts)

```bash
.venv/bin/python scripts/run_project.py --profile max_depth --rebuild-index --strict-audit
```

### Optional smoke sanity check (non-canonical)

```bash
.venv/bin/python scripts/run_smoke_evaluation.py
.venv/bin/python scripts/generate_smoke_figures.py
```

### Execute notebooks end-to-end

```bash
.venv/bin/python scripts/execute_notebooks.py --path notebooks --timeout 3600
```

### Launch interface

```bash
.venv/bin/python app/gradio_app.py
```

### Run tests

```bash
.venv/bin/python -m pytest -q
```

## Output Artifacts

- `data/processed/`: corpus tables, EDA summary, split manifest
- `data/eval/`: query labels, leakage audit
- `data/artifacts/`: metrics tables, visualizations, summaries
- `data/artifacts/final_audit/`: canonical real-run report and benchmark CSVs
- `chroma_db/`: persistent vector index

## Lessons Learned

- Retrieval quality is often the dominant bottleneck in RAG.
- Prompt constraints are required for grounded answers.
- Local-first systems can still implement serious evaluation rigor.
- Split policy and leakage checks are non-negotiable for trustworthy metrics.

## Future Improvements

- stronger cross-encoder reranking
- hybrid sparse+dense retrieval
- caching and async inference for throughput
- broader domain transfer benchmark dataset
- judge reliability calibration across models

## Verification Report

See `FINAL_PROJECT_VERIFICATION_REPORT.md` for:

- repository audit findings and fixes
- retrieval/generation/evaluation validation
- hallucination analysis review
- interface and reproducibility checks
- final scoring and residual limitations
