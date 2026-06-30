# Multi-Model LLM Playground

A local-first, multi-tab Gradio application that demonstrates six LLM-powered tasks (sentiment analysis, summarization, translation, chat, document OCR + Q&A, and benchmarking) using Ollama-local models. Each tab runs a different model optimized for its task — no cloud APIs, no API keys, no internet required after pulling the models.

---

## Table of Contents

1. [Overview & Vision](#1-overview--vision)
2. [Architecture Deep-Dive](#2-architecture-deep-dive)
3. [Tech Stack & Design Decisions](#3-tech-stack--design-decisions)
4. [Setup Guide](#4-setup-guide)
5. [Running the App](#5-running-the-app)
6. [Testing](#6-testing)
7. [Tutorial Notebooks](#7-tutorial-notebooks)
8. [Extension Guide](#8-extension-guide)
9. [Known Limitations & Tradeoffs](#9-known-limitations--tradeoffs)

---

## 1. Overview & Vision

**gradio-llm-playground** is a teaching-oriented project that wraps multiple local LLM workflows into a single Gradio UI. It was designed around four principles:

- **Local-first.** Everything runs through Ollama. No API keys, no cloud bills, no data leaving your machine.
- **Model-task alignment.** Each tab pairs with a model chosen for its size and strength at that task — not a one-size-fits-all approach.
- **Pedagogical clarity.** The code is intentionally unabstracted. Each module is independent so you can read, modify, or replace one without untangling a framework.
- **Measurable performance.** The benchmarking tab runs latency and throughput tests across four models, producing comparison charts.

### What you can do

| Tab | Task | Default Model |
|-----|------|---------------|
| Sentiment | Classify text as positive/negative/neutral | qwen3.5:2b |
| Summarization | Condense long text into TL;DR + key points | granite4.1:3b |
| Translation | Translate between 18 languages | translategemma:4b |
| Chat | Multi-turn conversational agent (20-turn memory) | qwen3.5:4b |
| Document Analyzer | OCR images/PDFs + ask questions about extracted text | glm-ocr → qwen3.5:4b |
| Benchmarking | Latency, throughput, scaling, and radar charts | All 4 models |

---

## 2. Architecture Deep-Dive

### 2.1 Module Map

```
app.py                          ← Gradio entry point, 6 tabs
│
├── src/ollama_client.py        ← HTTP client shared by all modules
│
├── src/sentiment.py            ← SentimentAnalyzer (qwen3.5:2b)
├── src/summarization.py        ← Summarizer       (granite4.1:3b)
├── src/translation.py          ← Translator       (translategemma:4b)
├── src/chat.py                 ← ChatEngine       (qwen3.5:4b)
├── src/document_analyzer.py    ← DocumentAnalyzer (glm-ocr + qwen3.5:4b)
│
├── src/benchmarking.py         ← BenchmarkRunner  (4 models)
├── src/visualization.py        ← BenchmarkVisualizer → outputs/figures/
│
├── tests/                      ← 3 test files
│   ├── test_sentiment.py       ← Unit tests (no Ollama needed)
│   ├── test_integration.py     ← Integration tests (requires Ollama)
│   └── test_validation.py      ← Import/docstring/model validation
│
└── notebooks/                  ← 4 Jupyter notebooks
    ├── 01_gradio_fundamentals.ipynb
    ├── 02_model_integration.ipynb
    ├── 03_benchmarking.ipynb
    └── 04_visualization.ipynb
```

### 2.2 `OllamaClient` — The HTTP Base

**File:** `src/ollama_client.py`

A lightweight wrapper around Ollama's REST API. No LangChain, no wrappers — just `httpx` and JSON.

```python
class OllamaClient:
    BASE = "http://localhost:11434"

    def generate(self, model: str, prompt: str, **kwargs) -> dict:
        """Single-turn text generation via POST /api/generate."""
        ...

    def chat(self, model: str, messages: list, **kwargs) -> dict:
        """Multi-turn conversation via POST /api/chat."""
        ...

    def embed(self, model: str, text: str) -> list[float]:
        """Text embeddings via POST /api/embed."""
        ...

    def measure_inference_time(self, model: str, prompt: str, **kwargs) -> dict:
        """Returns {'latency_s': float, 'tokens': int}."""
        ...
```

Key design choice: each consumer module creates and manages its own `OllamaClient` instance. Clients are independent — no shared state, no global connection pool. This means modules can be imported and tested in isolation, and resource cleanup is deterministic per module.

### 2.3 Sentiment Analyzer

**File:** `src/sentiment.py`

Prompts the model to return structured JSON and strips markdown fences from the raw output:

```python
SYSTEM_PROMPT = """You are a sentiment analyzer. Return ONLY valid JSON:
{"label": "positive"|"negative"|"neutral", "score": float, "explanation": string}
No markdown, no extra text."""

class SentimentAnalyzer:
    def analyze(self, text: str) -> dict:
        raw = self._client.generate(self.model, text, system=SYSTEM_PROMPT)
        return self._parse(raw["response"])
```

The `_parse` method handles three real-world failure modes: plain JSON, JSON inside ` ```json ` fences, and complete garbage (falls back to `{"label": "neutral", "score": 0.0}`).

### 2.4 Summarizer

**File:** `src/summarization.py`

Same JSON-prompt pattern but with three output keys: `summary`, `key_points`, and `tldr`. Skips the API call entirely for inputs under 50 characters (returns the input as-is).

### 2.5 Translator

**File:** `src/translation.py`

Supports 18 languages. Uses the `generate` endpoint (not `chat`) because TranslateGemma is a text-to-text model that responds better to low-temperature generation. Language codes are stored in a `LANGUAGES` dict mapping display names to ISO codes.

```python
def translate(self, text: str, target_language: str) -> dict:
    prompt = f"Translate this to {target_language}:\n{text}"
    result = self._client.generate(self.model, prompt, system=SYSTEM_PROMPT)
    return self._parse(result["response"])
```

### 2.6 Chat Engine

**File:** `src/chat.py`

Manages a conversation history list and uses Ollama's `/api/chat` endpoint (not `/api/generate`). History is capped at `max_turns=20` — `_trim()` drops the oldest exchanges beyond the limit.

```python
class ChatEngine:
    def __init__(self, model: str = "qwen3.5:4b", max_turns: int = 20):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        ...

    def send(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})
        reply = self._client.chat(self.model, self.history)
        self.history.append({"role": "assistant", "content": reply_text})
        self._trim()
        return reply_text
```

### 2.7 Document Analyzer

**File:** `src/document_analyzer.py`

A two-stage pipeline: extract text via OCR, then answer questions via a Q&A model.

1. **Stage 1 (OCR):** Sends the image as a base64-encoded string via Ollama's `images` param with `raw=False` to skip template wrapping — required for multimodal models like `glm-ocr`.
2. **Stage 2 (Q&A):** Feeds extracted text + user question to `qwen3.5:4b` using the chat endpoint.

```python
def extract_text(self, file_path: str) -> str:
    image_b64 = base64.b64encode(open(file_path, "rb").read()).decode()
    payload = {"model": self.ocr_model, "prompt": "Extract text.", "images": [image_b64], "raw": False}
    resp = httpx.post(f"{self.BASE}/api/generate", json=payload, timeout=60)
    ...

def answer_question(self, context: str, question: str) -> str:
    messages = [{"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}]
    resp = self._qa_client.chat(self.qa_model, messages)
    ...
```

### 2.8 Benchmarking Runner

**File:** `src/benchmarking.py`

Defines three prompt length buckets (short, medium, long) and runs inference on all four models. Results are returned as a nested dict keyed by model → prompt label → `{latency_s, tokens}`.

```python
PROMPTS = {
    "short":  "What is AI?",
    "medium": "Explain the difference between supervised and unsupervised learning...",
    "long":   "Write a detailed essay on the history of artificial intelligence...",
}

class BenchmarkRunner:
    def run_all(self) -> dict:
        for model in MODELS:
            for label, prompt in PROMPTS.items():
                m = client.measure_inference_time(model, prompt)
                ...
```

### 2.9 Benchmark Visualizer

**File:** `src/visualization.py`

Uses matplotlib with the `Agg` backend (no display required) to generate four chart types saved as PNGs in `outputs/figures/`:

| Chart | What it shows |
|-------|---------------|
| `latency_bar` | Per-model latency across prompt lengths |
| `throughput_bar` | Tokens-per-second comparison |
| `prompt_scale_line` | How latency scales as input grows |
| `radar_chart` | Multi-dimensional model comparison |

### 2.10 `app.py` — The Entry Point

Six tabs, each with a handler function that catches exceptions and returns formatted markdown strings (never crashes the UI). The app binds to `0.0.0.0:7860` with Gradio's `Soft` theme, making it accessible from any device on the local network.

---

## 3. Tech Stack & Design Decisions

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Inference | Ollama REST API | No cloud, no API keys, teach raw HTTP integration |
| HTTP | `httpx` | Async-capable, clean API, fewer deps than requests |
| UI | Gradio ≥5.0 | Fast prototyping, built-in file uploads, image display |
| Charts | matplotlib + Agg | Works on headless servers, no display needed |
| Packaging | `uv` + pyproject.toml | Modern Python packaging, fast dependency resolution |
| Type checking | mypy strict | Catches type errors at CI time |
| Linting | ruff | Fast, drop-in replacement for flake8 + isort + pyupgrade |
| Tests | pytest + asyncio + timeout | Covers unit, integration, and validation layers |

**Why not LangChain?** The project intentionally avoids LangChain to teach the underlying API mechanics. Each module constructs its own payloads and parses responses, making the request/response cycle visible and debuggable.

**Why so many models?** The project demonstrates model-task specialization. A 2B model is fast enough for sentiment; TranslateGemma produces better translations; a 4B model handles multi-turn chat more naturally than a smaller one. The benchmarking tab quantifies these differences.

---

## 4. Setup Guide

### Prerequisites

- Python ≥3.12
- [Ollama](https://ollama.com/download) installed and running
- `uv` (recommended) or pip

### 4.1 Clone and install

```bash
git clone https://github.com/your-username/gradio-llm-playground.git
cd gradio-llm-playground

# Option A — uv (recommended)
uv venv .venv && source .venv/bin/activate
uv sync

# Option B — pip
python -m venv .venv && source .venv/bin/activate
pip install .
```

### 4.2 Pull models

```bash
ollama pull qwen3.5:2b          # sentiment (fast)
ollama pull qwen3.5:4b          # chat + document Q&A + benchmarking
ollama pull granite4.1:3b       # summarization
ollama pull translategemma:4b   # translation
ollama pull glm-ocr:latest      # document OCR
ollama pull qwen3-embedding:4b  # embeddings (integration tests)
ollama pull nemotron-3-nano:4b  # benchmarking reference
```

Seven models total (~12 GB disk). You can substitute any model name in the module constructors if you prefer alternatives.

### 4.3 Verify connectivity

```bash
ollama list                  # confirm models are pulled
curl http://localhost:11434  # should return "Ollama is running"
```

---

## 5. Running the App

```bash
uv run python app.py
```

Opens at `http://localhost:7860`. The app binds to `0.0.0.0` so you can reach it from any device on your LAN via your machine's IP address.

### Tab-by-tab walkthrough

1. **Sentiment** — Type a sentence, click Analyze. Returns label, confidence score, and explanation.
2. **Summarization** — Paste in 50+ characters. Returns TL;DR, bullet-point summary, and key points.
3. **Translation** — Select a target language (18 available). The output is a plain textbox (not markdown) because the result is pure translated text.
4. **Chat** — Multi-turn conversation with memory. Click Reset Conversation to clear history.
5. **Document Analyzer** — Upload a PNG, JPG, or PDF. Optionally type a question. Returns extracted text and an AI-generated answer.
6. **Benchmarking** — Click Run Benchmark. Runs all 4 models across 3 prompt lengths. Displays latency, throughput, prompt-scaling, and radar charts.

---

## 6. Testing

Three test suites, each with a different purpose:

```bash
uv run pytest -v                    # all tests
uv run pytest tests/test_sentiment.py tests/test_validation.py -v  # unit only (no Ollama)
uv run pytest tests/test_integration.py -v  # requires Ollama running
```

| Test file | Kind | What it covers |
|-----------|------|----------------|
| `test_sentiment.py` | Unit | JSON parsing, markdown fence stripping, fallback on garbage, `analyze_batch`, empty input |
| `test_integration.py` | Integration | `generate`, `chat`, `embed`, `measure_inference_time` against live Ollama |
| `test_validation.py` | Validation | All 8 module imports, package `__all__` exports, default model names, translator languages, benchmark model count |

### What makes a good test

**Unit tests** (`test_sentiment.py`) exercise parsing logic without any network calls. They mock nothing — the `_parse` static method is pure Python. These are fast (<10ms each) and run anywhere.

**Integration tests** (`test_integration.py`) hit a live Ollama instance. They use `temperature=0.0` for deterministic output. These validate that the API contract is working but depend on the local environment.

**Validation tests** (`test_validation.py`) check module structure: imports resolve, attributes exist, magic numbers are documented as constants. These catch drift during refactoring.

---

## 7. Tutorial Notebooks

Four Jupyter notebooks in `notebooks/` walk through the codebase step by step:

```bash
uv run jupyter notebook notebooks/
```

| Notebook | Covers |
|----------|--------|
| `01_gradio_fundamentals.ipynb` | Gradio Blocks, layout, event handlers, state management |
| `02_model_integration.ipynb` | Using the Ollama API from Python, structured prompting |
| `03_benchmarking.ipynb` | Running benchmarks programmatically, interpreting results |
| `04_visualization.ipynb` | Generating benchmark charts with matplotlib |

Each notebook is self-contained and references the corresponding `src/` module.

---

## 8. Extension Guide

The architecture makes adding a new tab straightforward. Here is the pattern:

### Adding a new task tab

1. **Create a module** in `src/` following the existing pattern:

```python
# src/your_task.py
import json
from src.ollama_client import OllamaClient

SYSTEM_PROMPT = """Return ONLY valid JSON with key "result"."""

class YourTask:
    def __init__(self, model: str = "qwen3.5:2b") -> None:
        self.model = model
        self._client = OllamaClient()

    def run(self, text: str) -> dict:
        result = self._client.generate(self.model, text, system=SYSTEM_PROMPT)
        return self._parse(result["response"])

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _parse(raw: str) -> dict:
        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(cleaned)
```

2. **Export from `src/__init__.py`** — add the class name to `__all__`.
3. **Wire in `app.py`** — instantiate the class, write a handler function, and add a `gr.Tab`.
4. **Write tests** — at minimum a unit test for `_parse` and a validation test for the import.
5. **Pull a suitable model** — `ollama pull <model>`.

### Adding a new model to the benchmark

Edit `src/benchmarking.py` and add the model name to the `MODELS` list:

```python
MODELS = ["qwen3.5:2b", "qwen3.5:4b", "granite4.1:3b", "nemotron-3-nano:4b", "your-model:tag"]
```

The benchmarking tab and all four chart types will pick it up automatically.

### Customizing the UI theme

Gradio themes are passed at launch time. Edit the `app.launch()` call in `app.py`:

```python
app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
```

Gradio provides `Soft()`, `Default()`, `Monochrome()`, `Glass()`, and `Base()`. You can also use `gr.themes.ThemeClass` to set primary/secondary colors, font scales, and radius values.

### Switching to a cloud LLM backend

Replace `OllamaClient` with an OpenAI-compatible client:

```python
class OpenAIClient:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, model: str, prompt: str, **kwargs) -> dict:
        resp = self._client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], **kwargs
        )
        return {"response": resp.choices[0].message.content}
```

Swap the client in any module's `__init__` and no other code changes are needed — the rest of the module only calls `self._client.generate()`.

---

## 9. Known Limitations & Tradeoffs

| Area | Limitation | Mitigation |
|------|------------|------------|
| **Concurrency** | Each module creates its own `OllamaClient`. No connection pooling or async batching. | Fine for single-user local use. For multi-user, add `httpx.AsyncClient` and connection reuse. |
| **Session state** | Chat history is in-memory — lost on server restart. | Add SQLite or Redis-backed persistence for production. |
| **OCR fidelity** | `glm-ocr` produces raw text without layout structure. Tables, columns, and handwriting degrade quickly. | For structured documents, swap in a vision model (LLaVA, LLaMA 3.2 Vision) with layout-aware prompting. |
| **Model availability** | Models must be pulled before first use. `ollama pull` downloads multiple GB. | The setup guide lists all required models. A health-check endpoint could verify availability at startup. |
| **Translation endpoint** | Uses `generate` (not `chat`) because TranslateGemma expects text-to-text format. This means no conversation history for translation. | If using a chat-capable translation model, switch to `self._client.chat()`. |
| **Benchmark precision** | Single-shot measurements (no warmup, no median across runs). | For production benchmarking, run each prompt multiple times and report median + IQR. |
| **Error surface** | All errors surface as markdown strings in the UI — no structured error logging. | Add `loguru` and route Gradio errors to a `logging` handler with tracebacks. |
| **Security** | Server binds to `0.0.0.0` with no authentication. | Add Gradio's `auth` parameter or a reverse proxy (Caddy, nginx) for shared deployments. |

---

## Reference: Default Models

| Module | Class | Default Model | Endpoint |
|--------|-------|---------------|----------|
| `src/ollama_client.py` | `OllamaClient` | — | `/api/generate`, `/api/chat`, `/api/embed` |
| `src/sentiment.py` | `SentimentAnalyzer` | `qwen3.5:2b` | generate |
| `src/summarization.py` | `Summarizer` | `granite4.1:3b` | generate |
| `src/translation.py` | `Translator` | `translategemma:4b` | generate |
| `src/chat.py` | `ChatEngine` | `qwen3.5:4b` | chat |
| `src/document_analyzer.py` | `DocumentAnalyzer` | `glm-ocr` + `qwen3.5:4b` | generate (OCR), chat (QA) |
| `src/benchmarking.py` | `BenchmarkRunner` | 4 models | generate via `measure_inference_time` |
| `src/visualization.py` | `BenchmarkVisualizer` | — | matplotlib output |

---

## License

MIT
