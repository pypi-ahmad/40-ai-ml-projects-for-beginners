"""Agent catalog and dynamic agent generation."""

from __future__ import annotations

from dataclasses import dataclass

from crew_platform.config import AgentCatalogConfig, AgentProfile


@dataclass(slots=True)
class AgentCatalog:
    """In-memory agent catalog."""

    config: AgentCatalogConfig

    def all(self) -> list[AgentProfile]:
        return self.config.agents

    def roles(self) -> list[str]:
        return [agent.role for agent in self.config.agents]

    def get_by_role(self, role: str) -> AgentProfile | None:
        role_norm = role.strip().lower()
        for agent in self.config.agents:
            if agent.role.lower() == role_norm:
                return agent
        return None

    def ensure_role(self, role: str) -> AgentProfile:
        existing = self.get_by_role(role)
        if existing is not None:
            return existing

        # Dynamic crew generation for unknown roles.
        dynamic = AgentProfile(
            id=f"dynamic_{role.lower().replace(' ', '_')}",
            role=role,
            goal=f"Handle specialized tasks for {role}",
            backstory="Auto-generated specialist agent from planner requirements",
            tools=["memory_search", "web_search"],
            constraints=["Return structured JSON", "State assumptions"],
            output_schema="dynamic_output",
        )
        self.config.agents.append(dynamic)
        return dynamic
