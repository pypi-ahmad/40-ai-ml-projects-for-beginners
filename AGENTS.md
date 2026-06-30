# AGENTS.md — Local LLM Playground

## Project Overview
Multi-tab Gradio ML application showcasing local LLM inference via Ollama. Six functionality tabs (sentiment, summarization, translation, chat, document analysis, benchmarking) plus 4 tutorial notebooks.

## Architecture
- **Pattern**: Module-per-tab, shared `OllamaClient` base
- **Client lifecycle**: Each module creates/closes its own client — independent resource management
- **API Layer**: Direct Ollama REST API (http://localhost:11434), no langchain
- **UI**: Gradio Blocks with tabs, state management, file uploads
- **Testing**: pytest (unit + integration + validation)

## Module Reference

| Module | Class | Default Model | Key Methods |
|--------|-------|---------------|-------------|
| src/ollama_client.py | OllamaClient | — | generate, chat, embed, measure_inference_time |
| src/sentiment.py | SentimentAnalyzer | qwen3.5:2b | analyze, analyze_batch |
| src/summarization.py | Summarizer | granite4.1:3b | summarize |
| src/translation.py | Translator | translategemma:4b | translate, supported_languages |
| src/chat.py | ChatEngine | qwen3.5:4b (max_turns=20) | send, reset, history, close |
| src/document_analyzer.py | DocumentAnalyzer | glm-ocr + qwen3.5:4b | extract_text, answer_question, close |
| src/benchmarking.py | BenchmarkRunner | 4 models | run_all, run_single |
| src/visualization.py | BenchmarkVisualizer | — | latency_bar, throughput, prompt_scale, radar, generate_all |

## Key Design Decisions
1. Direct Ollama API over langchain — fewer deps, teaches fundamentals
2. JSON-structured prompting — parseable output across models
3. Agg matplotlib backend — headless figure generation
4. Soft theme, 0.0.0.0 binding — accessible from any device on network
5. translation uses generate (not chat) — lower temp, more predictable output
6. DocumentAnalyzer splits OCR and Q&A — uses specialized model for each step
7. OCR sends image base64 via `images` param with `raw=False` — bypasses template wrapping for multimodal models
8. ChatEngine caps history at `max_turns=20` — `_trim()` drops oldest exchanges beyond limit
9. All tab functions wrapped in try/except — errors surface as markdown strings, never crash UI

## Critical Paths
- **Benchmarking flow**: BenchmarkRunner.run_all() → dict → BenchmarkVisualizer.generate_all() → PNG files
- **Document analysis flow**: Upload → extract_text (glm-ocr) → answer_question (qwen3.5:4b) → response
- **Chat flow**: send_message → append history → Ollama chat API → parse response → return + store

## Models
- qwen3.5:2b — sentiment, quick tasks (fast)
- qwen3.5:4b — chat, Q&A, general reasoning (balanced)
- granite4.1:3b — summarization (good output quality)
- nemotron-3-nano:4b — benchmarking reference (baseline)
- translategemma:4b — translation (specialized)
- glm-ocr:latest — document OCR (primary)
- deepseek-ocr:latest — document OCR (fallback)
- qwen3-embedding:4b — text embeddings

## Testing
```bash
uv run pytest -v              # all tests
uv run pytest tests/test_sentiment.py tests/test_validation.py -v  # unit only
uv run pytest tests/test_integration.py -v  # requires Ollama
```

## Common Operations
```bash
uv venv .venv && source .venv/bin/activate  # setup
uv sync                                       # install deps
uv run python app.py                          # launch app
uv run jupyter notebook notebooks/            # notebooks
uv run pytest -v                              # tests
```

## Ollama API Reference
- `POST /api/generate` — single-turn generation
- `POST /api/chat` — multi-turn conversation
- `POST /api/embed` — text embeddings
- `GET /api/ps` — running models
- Streaming: SSE lines with `{"response": "..."}` chunks, final line has `"done": true`
- Structured output: Use `raw=True` to skip Ollama's template wrapping
