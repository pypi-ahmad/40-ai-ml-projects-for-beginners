#!/usr/bin/env python3
"""Generate architecture and flow diagrams as markdown + mermaid."""

from __future__ import annotations

from pathlib import Path

from ai_sql_assistant.constants import DIAGRAMS_DIR


def _write(name: str, body: str) -> Path:
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    path = DIAGRAMS_DIR / name
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return path


def main() -> None:
    architecture = """
# Architecture Diagram

```mermaid
flowchart LR
    User[User Question] --> UI[Streamlit App]
    UI --> Pipeline[AISQLAssistant Pipeline]
    Pipeline --> Schema[Schema Cache + Glossary]
    Pipeline --> Gen1[LangChain SQL Generator]
    Pipeline --> Gen2[Direct Ollama Generator]
    Gen1 --> Validator[SQL Validator]
    Gen2 --> Validator
    Validator --> Executor[Read-only SQLite Executor]
    Executor --> Explain[SQL Explainer]
    Executor --> Viz[Visualization Recommender]
    Explain --> UI
    Viz --> UI
    Pipeline --> Memory[History + Conversation Store]
```
"""

    eval_flow = """
# Evaluation Workflow

```mermaid
flowchart TD
    Cases[Benchmark Cases 100] --> Matrix[Model x Approach Matrix]
    Matrix --> Run[Generate SQL]
    Run --> Validate[Validate + Execute]
    Validate --> Compare[Compare vs Ground Truth]
    Compare --> Judge[LLM Judge granite4.1:3b]
    Judge --> Metrics[Latency + Correctness + Safety Metrics]
    Metrics --> Reports[JSON/Markdown Reports]
```
"""

    for name, body in [
        ("architecture_diagram.md", architecture),
        ("evaluation_workflow.md", eval_flow),
    ]:
        _write(name, body)


if __name__ == "__main__":
    main()
