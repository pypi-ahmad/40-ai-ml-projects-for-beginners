"""Explainability placeholders for attention/hidden-state diagnostics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ExplainabilitySummary:
    attention_available: bool
    hidden_states_available: bool
    token_importance_available: bool
    notes: str


def summarize_explainability_support(model_name: str) -> ExplainabilitySummary:
    lower = model_name.lower()
    attention = "bert" not in lower
    hidden_states = True
    token_importance = True
    return ExplainabilitySummary(
        attention_available=attention,
        hidden_states_available=hidden_states,
        token_importance_available=token_importance,
        notes="Use transformer output hooks for detailed visualizations in notebook workflows.",
    )
