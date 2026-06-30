# Project #8: Build a Live Machine Learning App in 5 Minutes

Portfolio-grade, local-first Gradio application that shows how to move from model inference to a usable AI product in minutes.

## Executive Summary
This project teaches end-to-end AI application engineering, not just model wrapping. It includes:
- Multi-tab Gradio product UI
- Local model serving via Ollama
- OCR + document Q&A workflow
- Translation and chat with persistent state
- Real benchmark pipeline with artifact export
- Beginner-friendly tutorial notebooks
- Preflight checks and test suite

Audience fit:
- AI / ML / GenAI / MLOps portfolios
- Interview demos
- Teaching workshops

## What Is Gradio?
Gradio is a Python-first framework for building interactive ML interfaces quickly.

Key primitives used in this project:
- `Blocks`: app-level composition system
- Components: `Textbox`, `Dropdown`, `Chatbot`, `File`, `Image`, `DataFrame`, `Markdown`
- Events: `.click()`, `.submit()`, `.change()`
- State: persistent per-session memory (`gr.State`)
- Themes: visual style applied at app construction

Why Gradio became popular in AI:
- Fast iteration loop
- Native fit for model demos and prototypes
- Easy local and temporary public sharing

## Traditional Software vs ML Applications
Traditional software:
- `Input -> Deterministic Rules -> Output`

ML application:
- `Input -> Preprocessing -> Model Inference -> Postprocessing -> UI Rendering`

In ML apps, quality depends on:
- Model-task alignment
- Prompt/contract design
- Latency and reliability
- UX clarity and failure handling

## AI Application Architecture
Layered architecture implemented in this repository:

1. UI Layer (`app.py`)
- Gradio Blocks, tabs, components, event bindings

2. Business/Event Layer ([src/ui_handlers.py](src/ui_handlers.py))
- Callback orchestration
- Model availability checks
- User-facing error formatting

3. Inference Layer (`src/*.py` task modules)
- Prompt design
- Parsing and validation
- Structured result schemas

4. Serving Layer ([src/ollama_client.py](src/ollama_client.py))
- Ollama REST calls
- Retry/backoff and timing metrics

5. Data/Artifact Layer (`outputs/`)
- Benchmark JSON/CSV/Markdown
- Figure generation
- Executed notebook outputs

Architecture figure:
- `outputs/figures/ml_app_architecture.png`

## Request-Response Flow
Every task follows this flow:
- User Input
- Validation / preprocessing
- Ollama inference call
- Structured parsing / postprocessing
- UI rendering with metadata + graceful errors

## Model Serving with Ollama
Serving is local through Ollama HTTP API:
- `POST /api/generate`
- `POST /api/chat`
- `POST /api/embed`
- `GET /api/tags`

Local serving advantages:
- Data stays local
- No per-request cloud cost
- Works offline if models are pulled

Tradeoff:
- Throughput/latency bounded by local hardware

## Supported Models
Core models in this project:
- `qwen3.5:2b` (fast sentiment)
- `qwen3.5:4b` (chat + QA)
- `granite4.1:3b` (summarization)
- `nemotron-3-nano:4b` (benchmark reference)
- `translategemma:4b` (translation)
- `glm-ocr:latest` / `deepseek-ocr:latest` (OCR primary/fallback)
- `qwen3-embedding:4b` (embeddings)

## Feature Tabs
1. Sentiment Analysis
- Structured label + confidence + explanation
- Robust JSON parsing fallback

2. Text Summarization
- Summary + bullet key points
- Fallback extraction from non-JSON responses

3. Translation
- Source/target language controls
- Input length and language validation

4. Local LLM Chat
- Per-model chat memory state
- Reset and model-switch history behavior

5. Document Analyzer
- PDF/image upload validation
- Native PDF extraction + OCR fallback
- Summary + grounded Q&A + warning surfacing

6. Benchmarking + Visualization
- Cold/warm latency, p95, tokens/s, memory, quality
- Exported JSON/CSV/Markdown + manifest

## OCR and Document Processing
Document pipeline supports:
- PDF (`.pdf`)
- Images (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, `.tif`, `.tiff`)

Hardening implemented:
- File type allowlist
- Max file-size guard
- Max image-pixel guard
- Corrupted image handling
- OCR primary/fallback model routing
- Warning propagation to UI

## Translation System
Translation module uses `translategemma:4b` through deterministic generation settings.

Validation includes:
- Empty input checks
- Max input length
- Supported source/target language checks
- Clear errors for unsupported language pairs

## Benchmarking Methodology
Benchmarks run repeated real inference calls and export artifacts.

Metrics:
- Mean latency (ms)
- P95 latency (ms)
- Tokens/sec
- Mean memory delta (MB)
- Cold-start latency (ms)
- Warm-start latency (ms)
- Quality score (judge-model + heuristic fallback)

Artifact contract:
- `outputs/benchmarks/short_results.json|csv`
- `outputs/benchmarks/medium_results.json|csv`
- `outputs/benchmarks/long_results.json|csv`
- `outputs/benchmarks/benchmark_results.json|csv`
- `outputs/benchmarks/benchmark_table.md`
- `outputs/benchmarks/artifact_manifest.json`

## Setup (uv + Python 3.12.10)
### Prerequisites
- Python `3.12.10`
- `uv`
- Ollama daemon running locally

### Environment
```bash
uv venv --python 3.12.10 .venv
source .venv/bin/activate
uv sync
```

### Pull required Ollama models
```bash
ollama pull qwen3.5:2b
ollama pull qwen3.5:4b
ollama pull granite4.1:3b
ollama pull nemotron-3-nano:4b
ollama pull translategemma:4b
ollama pull qwen3-embedding:4b
ollama pull glm-ocr:latest
ollama pull deepseek-ocr:latest
```

## Run the App
```bash
uv run python app.py
```

Environment knobs:
- `GRADIO_SERVER_NAME` (default `0.0.0.0`)
- `GRADIO_SERVER_PORT` (default `7860`, auto-fallback if busy)
- `GRADIO_SHARE` (`1/true/yes` to enable share link)
- `OLLAMA_BASE_URL` (default `http://localhost:11434`)

## Testing and Verification
### Lint
```bash
uv run ruff check app.py src scripts tests
```

### Tests
```bash
uv run pytest -ra
```

Notes:
- Integration tests are marked `integration` and auto-skip when Ollama is unreachable.
- This allows CI/sandbox runs without false failures while preserving live checks locally.

### Runtime verification (strict)
```bash
uv run python -m scripts.verify_runtime
```

### Preflight profiles
Fast profile (local quick confidence):
```bash
uv run python -m scripts.preflight --profile fast
```

Full profile (deep validation):
```bash
uv run python -m scripts.preflight --profile full
```

Optional flags:
- `--skip-runtime-check`
- `--skip-benchmarks`
- `--skip-notebooks`

## Notebooks (Zero-to-Hero Curriculum)
- `notebooks/01_ml_application_foundations.ipynb`
- `notebooks/02_gradio_fundamentals.ipynb`
- `notebooks/03_ollama_model_serving.ipynb`
- `notebooks/04_building_multi_tab_gradio_app.ipynb`
- `notebooks/05_benchmarking_and_visualization.ipynb`
- `notebooks/06_sharing_deployment_and_production.ipynb`

Execute all notebooks:
```bash
uv run python -m scripts.execute_notebooks
```

Executed copies are written to:
- `outputs/executed_notebooks/`

## Deployment Options and Tradeoffs
1. Gradio Share
- Fastest external demo link
- Temporary, not production-grade

2. Streamlit Cloud
- Easy demo hosting
- Less backend flexibility

3. Hugging Face Spaces
- Strong OSS AI visibility
- Platform constraints by tier/runtime

4. Docker + API + Frontend
- Best control, security, scalability
- Highest engineering effort

## Production Considerations
For production beyond demos, add:
- Authentication/authorization
- Rate limits and abuse controls
- Prompt/output safety policies
- Centralized monitoring and alerting
- Structured logging/tracing
- Horizontal scaling and failover

## Results and Artifacts
Current repository includes generated artifacts in:
- `outputs/benchmarks/`
- `outputs/figures/`
- `outputs/executed_notebooks/`

## Lessons Learned
- UI + error design matters as much as model quality.
- Model specialization improves user outcomes.
- Per-model state handling avoids chat context leaks.
- Benchmark artifact contracts improve reproducibility.

## Future Improvements
- Streaming token output in chat tab
- Chunked document retrieval before Q&A
- Prompt regression test harness
- Dockerized deployment profile
- Auth and audit logging for shared environments

## Security Notes
- Keep sensitive documents off public share links.
- Use file size/type restrictions for uploads.
- Treat local model outputs as untrusted user-facing content.

## Troubleshooting
1. Runtime check fails (`verify_runtime`)
- Ensure Ollama is running: `ollama serve`
- Confirm model list: `ollama list`

2. Integration tests skipped
- Ollama is unreachable from current shell/sandbox.
- Re-run on local host with Ollama daemon active.

3. Notebook execution fails with socket permission errors
- Run notebook execution outside restricted sandboxed environments.
