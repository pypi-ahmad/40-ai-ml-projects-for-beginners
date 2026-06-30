# Project #6: Build Your First AI App UI with Streamlit

Portfolio-grade, local-first AI application that teaches how to transform ML/LLM models into interactive user-facing software.

This project is intentionally built as both:
- A production-style Streamlit app (`streamlit_app/`)
- A Zero-to-Hero tutorial mini-book (notebooks in `notebooks/`)

---

## 1) What Is an AI Application?

An AI application is software where core behavior is produced by model inference (probabilistic) instead of only deterministic if/else rules.

### Traditional Software vs AI-Powered Software

| Dimension | Traditional Software | AI-Powered Software |
|---|---|---|
| Core logic | Hard-coded rules | Model inference + prompts |
| Output type | Deterministic | Probabilistic / context-sensitive |
| Failure mode | Bugs/exceptions | Hallucination, drift, weak confidence |
| Quality strategy | Unit tests | Tests + evals + prompt engineering + telemetry |

### AI App Layering (implemented in this repo)

```text
Frontend (Streamlit widgets + layouts)
  -> Backend orchestration (validation, routing, session state)
    -> Model layer (prompt templates + output parsing)
      -> Inference layer (local Ollama models)
        -> Data/artifact layer (session state + outputs/metrics + outputs/figures)
```

---

## 2) Why Streamlit for AI App Development?

Streamlit is ideal for fast AI product iteration:
- Python-native UI
- Built-in state and caching primitives
- Easy local deployment
- Rapid experimentation with model prompts

### Streamlit vs Flask/FastAPI/Django/Gradio

| Framework | Best for | Tradeoff |
|---|---|---|
| Streamlit | Data/AI interactive apps | Less control than full frontend stack |
| Flask | Lightweight web backends | More manual UI work |
| FastAPI | High-performance APIs | You still need frontend stack |
| Django | Full web platform | Higher setup overhead for AI demos |
| Gradio | Quick model demos | Less flexible app architecture than Streamlit for multi-workflow SaaS feel |

---

## 3) Implemented AI Mini-Apps

The app supports multi-page navigation with local inference:

1. **Sentiment Analysis** (`granite4.1:3b`)
2. **Text Summarization** (`qwen3.5:4b`)
3. **Text Classification** (`granite4.1:3b`)
4. **Translation** (`translategemma:4b`)
5. **Local LLM Chat** (`qwen3.5:4b`, optional fast model `qwen3.5:2b`)
6. **PDF/OCR Analysis** (`glm-ocr:latest` with fallback `deepseek-ocr:latest`)
7. **Benchmark Dashboard** (`qwen3.5:2b`, `qwen3.5:4b`, `granite4.1:3b`, `nemotron-3-nano:4b`)

---

## 4) Architecture and Code Design

### High-level responsibilities

- `streamlit_app/app.py`
  - Router and page lifecycle
- `streamlit_app/components/`
  - Sidebar and reusable charts/UI cards
- `streamlit_app/pages/`
  - Feature pages (one module per use case)
- `streamlit_app/utils/models.py`
  - Ollama API wrappers, OCR flow, benchmark metrics
- `streamlit_app/utils/caching.py`
  - `st.cache_resource` and `st.cache_data` wrappers
- `streamlit_app/utils/helpers.py`
  - Session defaults, validation, artifact paths, theme CSS
- `streamlit_app/services/prompts.py`
  - Prompt templates and teaching examples
- `streamlit_app/services/schemas.py`
  - Typed benchmark/sentiment/classification structures

### Request flow when user clicks button

```text
User input
  -> Validation (empty input / malformed categories / bad files)
    -> Prompt template
      -> Local model inference (Ollama)
        -> Output parsing + fallback handling
          -> Result rendering + metrics + artifact save
```

---

## 5) Streamlit Fundamentals Used in Real App

### Execution model

- Script reruns top-to-bottom on every interaction.
- Persistent state stored in `st.session_state`.

### Components used

- Text/UI: `st.title`, `st.header`, `st.subheader`, `st.markdown`, `st.write`
- Input: `st.text_area`, `st.text_input`, `st.selectbox`, `st.checkbox`, `st.radio`, `st.slider`, `st.file_uploader`, `st.chat_input`
- Output: `st.metric`, `st.dataframe`, `st.plotly_chart`, `st.image`, `st.expander`

### Caching

- `st.cache_resource`: model client metadata and long-lived resources
- `st.cache_data`: deterministic model task outputs

---

## 6) Caching, Optimization, and State Management

### Why caching matters

Model calls are expensive compared with pure Python logic. Cache avoids repeated identical inference and improves UX responsiveness.

### What is persisted in state

- Chat history
- User preferences (theme, teaching notes)
- Model settings (temperature, max tokens)
- Saved prompts
- Last benchmark run data

### Cache demonstration

Sentiment page includes `uncached` vs `cached first` vs `cached second` timing comparison.

---

## 7) Prompt Engineering Patterns Included

The project explicitly teaches:
- System prompt vs user prompt roles
- Structured JSON output prompts for safe parsing
- Good prompt vs bad prompt examples
- Context-window control via trimmed chat history

---

## 8) Benchmarking and Visualization

Benchmark pipeline measures:
- Latency (mean/median/min/max/std)
- Throughput (words/sec)
- Process memory proxy (MB)
- Quality proxy (lexical diversity + length balance)

Charts generated:
- Mean latency bar chart
- Latency distribution box plot
- Throughput bar chart
- Model tradeoff radar chart

Artifacts are saved into:
- `outputs/metrics/*.csv|*.json`
- `outputs/figures/*.html` (and PNG best-effort when local image export available)

---

## 9) OCR and File Processing Workflow

Supported uploads:
- PDF, DOCX, PNG, JPG, JPEG, BMP, TIFF

Flow:
1. Extract text by file type (`pdfplumber`, `python-docx`, `pytesseract`)
2. Analyze extracted text with primary OCR LLM (`glm-ocr:latest`)
3. Fallback to secondary OCR LLM (`deepseek-ocr:latest`) on failure

---

## 10) Setup and Run (uv + local .venv)

```bash
cd "MLOps, UI, and Deployment/Build Your First AI App UI with Streamlit"

uv venv .venv
source .venv/bin/activate
uv sync

# Start local Ollama daemon in another terminal
ollama serve

# Pull required local models
ollama pull qwen3.5:2b
ollama pull qwen3.5:4b
ollama pull granite4.1:3b
ollama pull nemotron-3-nano:4b
ollama pull qwen3-embedding:4b
ollama pull glm-ocr:latest
ollama pull deepseek-ocr:latest
ollama pull translategemma:4b

# Launch app
streamlit run main.py
```

Automation scripts:

```bash
bash scripts/setup_env.sh
bash scripts/run_full_pipeline.sh
```

---

## 11) Testing

```bash
# Fast unit/component tests
UV_CACHE_DIR=.uv_cache uv run pytest

# Optional live integration tests (requires running Ollama)
UV_CACHE_DIR=.uv_cache uv run pytest -m integration
```

Test categories cover:
- Input validation utilities
- Caching wrappers
- Model API wrappers (mocked)
- UI component rendering smoke tests
- Optional live local-model inference

---

## 12) Notebook Mini-Book (Zero-to-Hero)

1. `notebooks/00_ai_app_foundations.ipynb`
2. `notebooks/01_streamlit_fundamentals.ipynb`
3. `notebooks/02_ai_app_architecture_and_integration.ipynb`
4. `notebooks/03_caching_benchmarking_and_visualization.ipynb`
5. `notebooks/04_testing_deployment_and_production_readiness.ipynb`

Execute all end-to-end:

```bash
bash scripts/execute_notebooks.sh
```

---

## 13) Deployment Readiness Checklist

- Reproducible environment via `uv`
- Local model dependency documented and explicit
- Error handling for invalid input/model failures
- Artifacts exported for auditability
- Modular app architecture for maintainability
- Tests included for core utility and model integration logic

For remote deployment, containerize Streamlit app and expose `OLLAMA_HOST` to local/remote inference endpoint.

---

## 14) Known Runtime Constraint in Restricted Environments

If your environment blocks local socket binding to `127.0.0.1:11434`, live Ollama inference cannot run.
In that case, unit tests and notebooks still execute with graceful checks, but live benchmark metrics require running Ollama on a host with socket access.

---

## 15) Lessons Learned and Future Improvements

### Lessons
- Local inference improves privacy and cost control, but requires explicit performance management.
- Streamlit session state + caching can approximate SaaS-quality UX quickly.
- Prompt design quality strongly affects output reliability.

### Next improvements
- Add RAG tab with local vector store (`qwen3-embedding:4b`)
- Add structured eval harness (human + automatic rubric)
- Add auth/tenant controls for multi-user deployment
- Add Prometheus/Grafana telemetry for production monitoring
