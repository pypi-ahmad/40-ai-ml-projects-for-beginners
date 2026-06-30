from __future__ import annotations

from ai_spreadsheet_analytics.prompts import build_prompt, explain_temperature_zero


def test_prompt_builder_includes_role_and_limit() -> None:
    prompt = build_prompt("executive", {"kpi": 1}, max_words=120)
    assert "Role: executive" in prompt
    assert "Word limit: 120" in prompt


def test_temperature_explanation_mentions_reproducibility() -> None:
    explanation = explain_temperature_zero().lower()
    assert "reproduc" in explanation
