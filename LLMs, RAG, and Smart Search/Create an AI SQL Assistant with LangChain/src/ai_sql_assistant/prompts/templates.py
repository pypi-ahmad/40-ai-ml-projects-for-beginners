"""Prompt templates for personas and SQL generation tasks."""

from __future__ import annotations

from dataclasses import dataclass

from ai_sql_assistant.glossary import glossary_context


@dataclass(slots=True)
class PromptTemplateSpec:
    """Prompt template metadata."""

    name: str
    role_instruction: str


PERSONA_TEMPLATES: dict[str, PromptTemplateSpec] = {
    "Business Analyst": PromptTemplateSpec(
        name="Business Analyst",
        role_instruction="Focus on clear business KPIs, trends, and segmentation.",
    ),
    "Finance": PromptTemplateSpec(
        name="Finance",
        role_instruction="Prioritize revenue, margin proxy, cost proxy, and period-over-period comparisons.",
    ),
    "Sales": PromptTemplateSpec(
        name="Sales",
        role_instruction="Prioritize top customers, product mix, region performance, and quota-style rankings.",
    ),
    "HR": PromptTemplateSpec(
        name="HR",
        role_instruction="Focus on employee productivity and territory/customer coverage.",
    ),
    "Inventory": PromptTemplateSpec(
        name="Inventory",
        role_instruction="Focus on stock levels, category mix, supplier dependency, and slow movers.",
    ),
    "Marketing": PromptTemplateSpec(
        name="Marketing",
        role_instruction="Focus on customer cohorts, segment trends, and campaign-like geo insights.",
    ),
}


def sql_generation_prompt(
    question: str,
    schema_context: str,
    persona: str,
    memory_context: str,
) -> str:
    """Create deterministic SQL generation prompt."""
    persona_spec = PERSONA_TEMPLATES.get(persona, PERSONA_TEMPLATES["Business Analyst"])
    glossary = glossary_context(question)

    return f"""
You are enterprise SQL assistant. Generate only valid SQLite SELECT query.

Rules:
1) Read-only SQL only. Never use INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/CREATE.
2) Use explicit joins with ON clauses.
3) Prefer CTEs for complex logic.
4) Use aliases and deterministic ordering when top-N requested.
5) Return SQL only. No markdown fences, no explanation text.
6) If question ambiguous, choose safest common interpretation and include LIMIT 200.
7) Use temperature-0 style determinism by avoiding random constructs.

Persona objective:
{persona_spec.role_instruction}

Conversation context:
{memory_context}

Schema context:
{schema_context}

Business glossary hints:
{glossary}

Question:
{question}
""".strip()


def explanation_prompt(question: str, sql: str, execution_plan: str) -> str:
    """Prompt to generate beginner-friendly SQL explanation."""
    return f"""
You are SQL teacher for analytics beginners.

Given question and SQL, provide:
1) Step-by-step SQL logic.
2) Business interpretation.
3) Execution plan overview using provided SQLite EXPLAIN output.
4) Complexity notes and optimization tips.

Question: {question}
SQL: {sql}
Execution plan: {execution_plan}
""".strip()


def judge_prompt(question: str, generated_sql: str, ground_truth_sql: str) -> str:
    """Prompt for LLM-as-judge with strict JSON output."""
    return f"""
You evaluate SQL quality. Output strict JSON object with keys:
sql_correctness, business_correctness, completeness, readability, efficiency, safety, rationale.
All score keys must be float between 0 and 1.

Question: {question}
Generated SQL: {generated_sql}
Ground truth SQL: {ground_truth_sql}
""".strip()
