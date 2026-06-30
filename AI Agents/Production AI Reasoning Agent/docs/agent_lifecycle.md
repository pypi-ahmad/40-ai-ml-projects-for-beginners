# Agent Lifecycle

1. Load config + tool registry + memory managers.
2. Planner decomposes task into steps with tool metadata.
3. Tool Router validates tool availability and schemas.
4. Executor invokes tool and captures output, latency, status.
5. Observation Processor updates state and iteration counters.
6. Reflector decides retry vs continue vs stop.
7. Response Generator builds grounded final answer.
8. Memory stores user/assistant turns and optional semantic artifacts.
9. Observability layer exports JSONL traces and aggregate metrics.
