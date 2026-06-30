# Project #18 — Hybrid Real-Time AI Research Assistant

Production-grade local-first AI Research Assistant built with **LangGraph**, **ChromaDB**, **Ollama**, and optional **live web search**. It routes queries across local knowledge, web, or both, then generates grounded answers with citations.

## Features

- LangGraph workflow (intent -> routing -> retrieval -> rerank -> context -> generation -> judge -> response)
- Local RAG on PDF/TXT/MD/HTML/DOCX + OCR images
- Web search adapters (DuckDuckGo default, Tavily/Brave optional)
- Hybrid retrieval with merge + dedupe + reranking
- Persistent ChromaDB with incremental indexing, deletion, and updates
- Metadata-rich citations (source, page, URL, chunk ID, confidence)
- Conversation memory + semantic response cache
- LLM-as-a-judge evaluation pipeline (`granite4.1:3b`)
- Multi-page Streamlit UI
- Benchmark tooling for chunking and embedding model comparison

## Tech Stack

- Python 3.12.10
- uv
- LangGraph / LangChain Core
- ChromaDB
- Ollama (`qwen3.5:4b`, `granite4.1:3b`, `glm-ocr:latest`)
- Embeddings: `BAAI/bge-small-en-v1.5` (default), `all-MiniLM-L6-v2`, `nomic-embed-text`

## Why `BAAI/bge-small-en-v1.5` as default embedding

`BAAI/bge-small-en-v1.5` provides a strong quality/latency/memory tradeoff for local RAG:

- Better retrieval quality than lightweight baseline models in many technical corpora
- Lower latency and memory footprint than larger BGE variants
- Fixed-size vectors that are practical for persistent local Chroma indexing
- Works well for semantic search before cross-encoder reranking

The project still supports benchmarking and switching to `all-MiniLM-L6-v2` or `nomic-embed-text`.

## Project Structure

```text
.
├── configs/
├── data/
│   ├── documents/
│   ├── eval/
│   ├── notes/
│   └── web_cache/
├── docs/
├── notebooks/
├── outputs/
│   ├── benchmarks/
│   ├── diagrams/
│   ├── reports/
│   └── screenshots/
├── scripts/
├── src/hybrid_research_assistant/
├── streamlit_app/
└── tests/
```

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv sync --all-groups
cp .env.example .env
```

Pull local models in Ollama:

```bash
ollama serve
ollama pull qwen3.5:4b
ollama pull granite4.1:3b
ollama pull glm-ocr:latest
ollama pull nomic-embed-text
```

## Configure

- Main config: `configs/app.yaml`
- Env overrides: `.env`
- Optional provider keys: `TAVILY_API_KEY`, `BRAVE_API_KEY`

## CLI Usage

### Bootstrap corpus structure

```bash
uv run hybrid-ra bootstrap-corpus
```

### Build/update index

```bash
uv run hybrid-ra ingest --strategy recursive --chunk-size 768 --chunk-overlap 100
```

### Query assistant

```bash
uv run hybrid-ra query "What is LangGraph?" --mode auto --prompt research_assistant --provider duckduckgo
```

### Generate benchmark dataset (100 questions)

```bash
uv run hybrid-ra generate-eval-set
```

### Evaluate

```bash
uv run hybrid-ra evaluate
```

### Run chunking + embedding benchmarks

```bash
uv run hybrid-ra benchmark
```

### Run failure analysis

```bash
uv run hybrid-ra run-failure-analysis
```

### Export LangGraph workflow diagram

```bash
uv run hybrid-ra export-workflow
```

## Streamlit App

```bash
uv run streamlit run streamlit_app/Home.py
```

Pages:

- Home
- Knowledge Base
- Upload Documents
- Index Builder
- Search
- Chat
- Retrieved Chunks
- Sources
- Evaluation
- Settings
- Analytics

## Notebook

- `notebooks/project_18_hybrid_real_time_ai_research_assistant.ipynb`

Execute end-to-end:

```bash
bash scripts/run_notebook.sh
```

## Testing

```bash
uv run pytest -q
```

## Security Notes

- Local-first execution by default
- No hardcoded secrets
- Optional web APIs only through env vars
- Citation-only response policy for grounding
- Fallback when evidence is insufficient:

```text
I don't know based on the retrieved information.
```

## Limitations

- Live web quality depends on provider availability
- OCR quality depends on image quality/model support
- Reranker/model downloads can be heavy on first run
- Full-scale benchmark runtime depends on local compute and corpus size

## Future Work (v2)

- Voice input and speech synthesis
- Rich multimodal retrieval
- GitHub repository indexing workflows

## References

- LangGraph docs
- ChromaDB docs
- Ollama docs
- DuckDuckGo Search
- Sentence-Transformers
