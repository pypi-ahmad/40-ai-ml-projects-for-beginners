"""CrewAI adapter for role-specialized task execution."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from loguru import logger

from crew_platform.config import AgentProfile


class CrewAIRunner:
    """Best-effort CrewAI execution adapter with graceful fallback."""

    def __init__(
        self,
        model: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
        os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
        os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
        os.environ.setdefault("OTEL_SDK_DISABLED", "true")

    async def run_task(
        self,
        profile: AgentProfile,
        description: str,
        context: str,
    ) -> dict[str, Any]:
        try:
            from crewai import Agent, Crew, LLM, Process, Task
        except Exception:  # pragma: no cover - optional runtime dependency
            Agent = Crew = LLM = Process = Task = None  # type: ignore[assignment]

        if Agent is None or Crew is None or Task is None or LLM is None:
            return {
                "mode": "fallback",
                "output": f"CrewAI unavailable. Fallback output for role={profile.role}.",
            }

        model_name = self.model if self.model.startswith("ollama/") else f"ollama/{self.model}"
        try:
            llm = LLM(
                model=model_name,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            agent = Agent(
                role=profile.role,
                goal=profile.goal,
                backstory=profile.backstory,
                llm=llm,
                verbose=False,
                allow_delegation=True,
            )
            task = Task(
                description=f"{description}\n\nContext:\n{context}",
                expected_output=f"JSON-like content for schema {profile.output_schema}",
                agent=agent,
            )
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )
            if hasattr(crew, "kickoff_async"):
                result = await crew.kickoff_async()
            else:
                result = await asyncio.to_thread(crew.kickoff)
            return {"mode": "crewai", "output": str(result)}
        except Exception as exc:  # noqa: BLE001
            logger.warning("CrewAI task failed, fallback used: {}", exc)
            return {"mode": "fallback", "output": f"Fallback due to CrewAI error: {exc}"}
