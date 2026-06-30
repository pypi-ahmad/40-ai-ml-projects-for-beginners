# Production-Grade AI Reasoning Agent Framework

Production-oriented local AI agent framework using LangGraph-compatible architecture, dynamic tool orchestration, memory tiers, structured outputs, observability, benchmarking, and a multi-page Streamlit UI.

## Executive Summary

This project implements a modular reasoning agent with:

- Planner, router, executor, observation processor, reflector, response generator, error handler, and execution logger.
- Dynamic `ToolRegistry` with tool metadata, schema validation, discovery, and invocation telemetry.
- ReAct-style loop (`thought -> action -> observation -> reflection -> answer`) with iteration history.
- Session memory + optional ChromaDB semantic memory retrieval.
- Structured parsing with Pydantic and retry/fallback handling.
- Benchmark/evaluation pipeline for 120 prompts across multiple Ollama models.

The codebase is designed as a lightweight production framework, not a tutorial-only demo.

## Architecture

Detailed architecture docs:

- [docs/architecture.md](docs/architecture.md)
- [docs/agent_lifecycle.md](docs/agent_lifecycle.md)

Core package layout:

```text
src/reasoning_agent/
  agent/            # planner/router/executor/reflector/graph/runner
  tools/            # base tool contracts, registry, required/optional tools
  memory/           # session + Chroma semantic memory
  llm/              # Ollama provider
  evals/            # benchmark dataset, evaluator, judge, reports
  observability/    # metrics, tracing, charts
  prompts/          # planning/tool-selection/reflection/response prompts
streamlit_app/      # Streamlit multipage UI
configs/            # YAML settings
scripts/            # demo + benchmark runners
notebooks/          # educational notebook
```

## Agent Lifecycle

1. Build plan from query and available tools.
2. Route the current step to a valid tool (or no-tool step).
3. Execute tool with schema validation.
4. Process observation and update loop state.
5. Reflect on errors and apply retry/fallback policy.
6. Generate final response from observations/reflection.
7. Persist memory and trace telemetry.

## LangGraph Workflow

Defined in `src/reasoning_agent/agent/graph.py` as:

- `planner -> tool_router -> executor -> observation -> reflection`
- Conditional branch:
  - `continue -> tool_router`
  - `respond -> response -> END`
  - `error -> error_handler -> (continue|respond)`

Runtime modes (`agent.runtime_mode`):

- `fallback` (default): deterministic loop for maximum stability.
- `graph`: execute graph runtime directly.
- `auto`: attempt graph and fallback on failures/timeouts.

## Tool System

Implemented required tools:

- calculator, duckduckgo_search, wikipedia, datetime
- unit_converter, currency_converter, weather
- file_reader, csv_analyzer, json_explorer, markdown_reader, webpage_reader
- document_search, local_rag
- semantic_search/vector_search (when Chroma memory is enabled)
- python_repl (disabled by default; explicitly opt-in)

Optional tools:

- sqlite_query, github_search, news_search, arxiv_search

Tool controls:

- `tools.enabled_tools`: allowlist (`["*"]` by default)
- `tools.optional_tools`: opt-in optional tool list
- `tools.enable_python_tool`: strict gate for Python execution tool

## Memory System

- Session memory: rolling conversation window.
- Semantic memory (optional): ChromaDB store for prior context retrieval.
- Query-time memory retrieval is bounded by `memory.memory_top_k`.

## Security and Safety

- No `eval` in calculator.
- Python REPL hardened with:
  - AST validation
  - blocked dangerous names/dunder attributes
  - restricted builtins and import allowlist
  - process timeout + memory limit
- File-based tools are workspace scoped.
- Tool IO and LLM structured outputs validated with Pydantic.

## Setup

```bash
uv venv .venv
source .venv/bin/activate
UV_CACHE_DIR=.uv-cache uv sync --all-groups
```

## Run

```bash
# demo query
.venv/bin/python scripts/run_demo.py

# tests
.venv/bin/pytest -q

# tests + coverage
.venv/bin/pytest --cov=src/reasoning_agent --cov-report=term-missing -q

# benchmark + dashboards
AGENT_OFFLINE_MODE=1 .venv/bin/python scripts/run_benchmarks.py

# streamlit app
.venv/bin/python app.py
```

## Configuration

Primary config file: `configs/settings.yaml`

Key fields:

- `agent.runtime_mode`
- `agent.graph_timeout_seconds`
- `agent.graph_fallback_on_error`
- `agent.reasoning_mode`
- `agent.use_llm_for_planning`
- `agent.use_llm_for_response`
- `tools.enabled_tools`
- `tools.enable_python_tool`
- `memory.chroma_enabled`
- `memory.memory_top_k`
- `retries.max_retries`
- `retries.backoff_seconds`

Environment override convention:

- `AGENT__SECTION__KEY=value` (example: `AGENT__LLM__BASE_URL=http://127.0.0.1:11434`)

## Benchmark Snapshot (June 27, 2026)

Source: `artifacts/benchmarks/summary_20260627T150834Z.csv`

| Model | Prompts | Success Rate | Avg Latency (ms) | Avg Tool Calls | Avg Keyword Score | Avg Tool Selection | Avg Judge Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3:8b | 120 | 1.00 | 6.53 | 0.75 | 0.38 | 0.56 | 0.00 |
| llama3.1:8b | 120 | 1.00 | 6.38 | 0.75 | 0.38 | 0.56 | 0.00 |
| granite4.1:3b | 120 | 1.00 | 6.45 | 0.75 | 0.38 | 0.56 | 0.00 |
| deepseek-r1 | 120 | 1.00 | 7.54 | 0.75 | 0.38 | 0.56 | 0.00 |

Generated dashboards:

- `artifacts/benchmarks/latency.html`
- `artifacts/benchmarks/success_rate.html`
- `artifacts/benchmarks/tool_usage.html`
- `artifacts/benchmarks/quality_radar.html`

## Streamlit UI

Pages:

- Chat
- Execution Trace
- Tool Calls
- Memory
- Analytics
- Settings

## Notebook

- `notebooks/zero_to_hero_reasoning_agent.ipynb`
- Executed counterpart:
  - `notebooks/zero_to_hero_reasoning_agent.executed.ipynb`

## Known Limitations

- Graph runtime may behave differently across environments; default runtime mode is `fallback` for stability.
- LLM judge scores are `0.0` in offline benchmark runs where judge model execution is unavailable.
- Streamlit startup verification may require unsandboxed execution on restricted hosts.

## Future Improvements

- Parallel multi-tool execution and async fan-out planning.
- Robust graph-runtime lifecycle management across all hosting environments.
- Higher coverage on UI/evals modules and richer integration tests.
- Human approval checkpoints and multi-agent specializations.

## References

- LangGraph: https://langchain-ai.github.io/langgraph/
- LangChain Core: https://python.langchain.com/docs/
- Ollama: https://github.com/ollama/ollama
- ChromaDB: https://docs.trychroma.com/
- Streamlit: https://docs.streamlit.io/
