"""Agent specification contracts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AgentSpec:
    """Definition of one enterprise workflow agent."""

    name: str
    role: str
    objective: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    output_schema: dict[str, str] = field(default_factory=dict)
    retry_strategy: str = "retry_with_fallback"
