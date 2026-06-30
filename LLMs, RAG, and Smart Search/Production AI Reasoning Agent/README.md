# Production-Grade AI Reasoning Agent Framework

Production-ready local AI reasoning agent built with **LangGraph**, **LangChain Core**, **Ollama**, **ChromaDB**, **dynamic MCP-style tools**, and **Streamlit**.

## Highlights

- LangGraph-first orchestration (no monolithic while loop)
- ReAct-style loop with thought/action/observation/reflection tracking
- Dynamic tool registry with metadata + input/output schema validation
- Persistent semantic memory via ChromaDB with embedding fallback
- Structured outputs with Pydantic validation + automatic retry repair
- Professional Streamlit interface (chat, traces, tools, memory, analytics)
- Benchmark suite (120 prompts) with model auto-skip for unavailable models
- Hybrid evaluation (deterministic + LLM-as-a-Judge using `granite4.1:3b`)
- Structured logging, trace persistence, metrics, and report export

## Architecture

See [docs/architecture.md](docs/architecture.md).

Core pipeline:

1. Planner
2. Tool Router
3. Executor
4. Observation Processor
5. Reflector
6. Error Handler
7. Response Generator

## Project Structure

```text
src/reasoning_agent/
  agent/          # Runner + state
  graph/          # LangGraph wiring
  planner/        # Planning + reflection
  routing/        # Tool selection
  executor/       # Tool execution bridge
  tooling/        # Registry + tools
  memory/         # Short-term + Chroma semantic memory
  parsing/        # Structured output parser
  evaluation/     # Benchmark + judge + visualization
  observability/  # Logs, traces, metrics
streamlit_app/
  Home.py
  pages/
benchmarks/
  prompts.jsonl   # 120 evaluation prompts
notebooks/
  project_20_production_reasoning_agent.ipynb
```

## Installation

```bash
uv python install 3.12.10
uv sync --all-extras
cp .env.example .env
```

## Configuration

- Environment variables: `.env`
- YAML config: `configs/config.yaml`
- Effective precedence: `env > yaml > defaults`

Key config groups:

- Models (primary/compare/embedding)
- Graph iterations/retries/timeouts
- Tools and provider adapters
- Memory (Chroma path, retention)
- Logging and trace visibility
- Benchmark dataset/output paths

## Run

### CLI chat

```bash
uv run reasoning-agent chat "Calculate 12*7 and explain"
```

### Full benchmark (100+ prompts)

```bash
uv run reasoning-agent benchmark --output-dir artifacts/reports
```

### Streamlit app

```bash
uv run streamlit run streamlit_app/Home.py
```

## Streamlit Pages

- Chat
- Execution Trace
- Tool Calls
- Memory
- Analytics
- Settings

## Tool System

Required implemented tools:

- Calculator
- DuckDuckGo Search
- Wikipedia
- Python REPL (sandboxed)
- Datetime
- Unit Converter
- Currency Converter (adapter)
- Weather (adapter)
- File Reader
- CSV Analyzer
- JSON Explorer
- Webpage Reader
- Markdown Reader
- Document Search
- Semantic Search
- Vector Search
- Local RAG

Optional stubs included:

- SQLite
- GitHub Search
- News Search
- arXiv Search

## Memory

Implemented memory tiers:

- Conversation memory
- Working/short-term memory
- Tool/observation memory
- Session memory
- Semantic memory (Chroma)

Retention and purge behavior configurable.

## Evaluation and Benchmarking

- Dataset: `benchmarks/prompts.jsonl` (120 prompts)
- Models: `qwen3:8b`, `llama3.1:8b`, `granite4.1:3b`, `deepseek-r1` (optional)
- Auto-skip missing models with explicit `skip_reason`
- Metrics: hybrid accuracy, success rate, latency, tool calls, retries
- Outputs:
  - `artifacts/reports/benchmark_summary.json`
  - `artifacts/reports/benchmark_details.json`
  - `artifacts/plots/*.html`

## Security and Safety

- No arbitrary shell tool
- Python sandbox with AST restrictions + process limits
- Workspace-restricted filesystem tools
- Schema validation for every tool call
- Graceful external API unavailability handling

## Testing

```bash
uv run pytest
```

Coverage gate is enforced at `>=80%` for core package.

## Notebook

Tutorial notebook:

- `notebooks/project_20_production_reasoning_agent.ipynb`

Covers architecture, ReAct flow, tools, memory, benchmarks, observability, and UI workflow.

## Screenshots and Artifacts

Screenshots folder ready at `docs/screenshots/`.

Generate real screenshots after running Streamlit and benchmark in local environment.

## Limitations

- Full benchmark runtime depends on local model availability and hardware.
- Network-backed tools depend on provider uptime/connectivity.
- Judge quality depends on local `granite4.1:3b` behavior.

## Future Improvements (Phase 2)

- Async + parallel tool execution
- Streaming token responses
- MCP-compatible tool interface adapter
- Human approval checkpoints for risky actions

## References

- LangGraph documentation
- LangChain Core documentation
- Ollama API docs
- ChromaDB docs
- Streamlit docs
