"""Direct prompting SQL generator using Ollama REST API."""

from __future__ import annotations

from ai_sql_assistant.config import AppSettings
from ai_sql_assistant.llm.ollama_client import OllamaDeterministicClient
from ai_sql_assistant.types import GenerationCandidate
from ai_sql_assistant.utils.sql_utils import clean_sql_text, format_sql


class DirectSQLGenerator:
    """Generate SQL directly through deterministic Ollama prompts."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.client = OllamaDeterministicClient(settings)

    def close(self) -> None:
        """Close network resources."""
        self.client.close()

    def generate(
        self,
        question: str,
        schema_context: str,
        persona: str,
        memory_context: str,
        model: str,
    ) -> GenerationCandidate:
        """Generate SQL candidate from direct prompt."""
        compact_schema = "\n".join(schema_context.splitlines()[:20])
        prompt = f"""
You are deterministic SQLite query generator for enterprise analytics.
Return only one SQL query beginning with SELECT or WITH.
Never emit markdown or explanation.
Read-only SQL only.
Persona: {persona}

Conversation context:
{memory_context}

Schema (compact):
{compact_schema}

Question:
{question}
""".strip()

        response = self.client.generate(prompt=prompt, model=model)
        cleaned = clean_sql_text(response.text)

        # Retry once with stricter instruction if output is not SQL-like.
        if not cleaned.lower().startswith(("select", "with")):
            strict_prompt = f"""
You are SQLite query generator.
Return exactly one SQL query starting with SELECT or WITH.
Never return explanation text.
Use read-only analytics SQL only.

Schema (compact):
{compact_schema}

Question:
{question}
""".strip()
            response = self.client.generate(prompt=strict_prompt, model=model)
            cleaned = clean_sql_text(response.text)

        sql = format_sql(cleaned)
        return GenerationCandidate(
            sql=sql,
            approach="direct",
            model=model,
            prompt_name=persona,
            latency_ms=response.latency_ms,
            raw_response=response.text,
        )
