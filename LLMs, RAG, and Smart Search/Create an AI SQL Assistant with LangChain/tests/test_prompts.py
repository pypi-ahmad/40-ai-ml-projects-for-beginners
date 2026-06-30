from __future__ import annotations

from ai_sql_assistant.prompts.templates import sql_generation_prompt


def test_prompt_contains_safety_rules() -> None:
    prompt = sql_generation_prompt(
        question="Show monthly revenue",
        schema_context="TABLE orders(order_id INTEGER, order_date TEXT)",
        persona="Finance",
        memory_context="No prior context",
    )

    assert "Read-only SQL only" in prompt
    assert "Return SQL only" in prompt
    assert "Persona objective" in prompt


def test_prompt_uses_glossary() -> None:
    prompt = sql_generation_prompt(
        question="Show sales by customer segment",
        schema_context="TABLE customers(customer_id TEXT, segment TEXT)",
        persona="Marketing",
        memory_context="No prior context",
    )
    assert "Business glossary hints" in prompt
    assert "sales" in prompt.lower()
