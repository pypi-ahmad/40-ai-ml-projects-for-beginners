"""Prompt templates and builders."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """
You are an AI business analytics assistant.
Rules:
1. Never invent numeric values.
2. Use deterministic evidence provided by Python analytics only.
3. If evidence missing, state uncertainty explicitly.
4. Keep recommendations practical and prioritized.
5. Return concise markdown.
""".strip()

ROLE_PROMPTS: dict[str, str] = {
    "executive": "Provide executive summary with strategic priorities and risk flags.",
    "sales_analyst": "Focus on revenue drivers, top products, and conversion opportunities.",
    "finance_analyst": "Focus on financial performance, variance, cost/revenue signals, and risk.",
    "marketing_analyst": "Focus on acquisition, retention, channel performance, and campaign insights.",
    "operations_analyst": "Focus on process bottlenecks, throughput, efficiency, and SLA risks.",
    "product_manager": "Focus on customer behavior, feature adoption, and product opportunities.",
    "data_scientist": "Focus on signal quality, assumptions, potential leakage, and next experiments.",
}


def explain_temperature_zero() -> str:
    """Explain deterministic decoding requirement."""
    return (
        "Temperature=0 keeps token choice deterministic. "
        "In business analytics this improves reproducibility, auditability, regression testing, "
        "and consistency across benchmark/judge runs."
    )


def build_prompt(role: str, deterministic_payload: dict[str, Any], max_words: int = 180) -> str:
    """Build role-specific prompt.

    Args:
        role: Prompt role name.
        deterministic_payload: Computed analytics evidence.
        max_words: Maximum summary words.

    Returns:
        Prompt text for LLM.
    """
    instruction = ROLE_PROMPTS.get(role, ROLE_PROMPTS["executive"])
    return (
        f"Role: {role}\n"
        f"Task: {instruction}\n"
        f"Word limit: {max_words}\n"
        "Use markdown sections:\n"
        "- Executive Summary\n- Key Findings\n- Risks\n- Opportunities\n- Recommendations\n"
        "Use only evidence below:\n"
        f"{deterministic_payload}\n"
    )
