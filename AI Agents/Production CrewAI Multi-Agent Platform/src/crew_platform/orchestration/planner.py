"""Dynamic task planner."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from crew_platform.config import Settings
from crew_platform.llm.ollama import OllamaProvider
from crew_platform.orchestration.catalog import AgentCatalog
from crew_platform.orchestration.models import PlanProposal, TaskSpec


class PlannerService:
    """Builds dependency-aware task plans from user objectives."""

    def __init__(self, settings: Settings, llm: OllamaProvider, catalog: AgentCatalog) -> None:
        self.settings = settings
        self.llm = llm
        self.catalog = catalog

    async def build_plan(self, query: str) -> PlanProposal:
        run_id = f"run-{uuid.uuid4().hex[:10]}"
        llm_tasks = await self._try_llm_plan(query)
        tasks = llm_tasks or self._fallback_plan(query)

        # Ensure all roles exist in catalog (dynamic crew generation support).
        normalized: list[TaskSpec] = []
        for task in tasks:
            profile = self.catalog.ensure_role(task.agent_role)
            normalized.append(
                task.model_copy(update={"agent_role": profile.role, "output_schema": profile.output_schema})
            )

        return PlanProposal(
            run_id=run_id,
            objective=query,
            tasks=normalized,
            assumptions=[
                "Planner enforces dependency-safe DAG",
                "Low-confidence outputs trigger additional research",
                "Network tools degrade gracefully when unavailable",
            ],
        )

    async def _try_llm_plan(self, query: str) -> list[TaskSpec]:
        """Try model-based plan creation with strict JSON contract."""

        prompt = self._planner_prompt(query)
        try:
            response = await self.llm.generate(
                prompt=prompt,
                model=self.settings.llm.planner_model,
                temperature=0.1,
                max_tokens=min(1400, self.settings.llm.max_tokens),
                raw=True,
            )
        except Exception:
            return []

        payload = self._extract_json(response.text)
        if not payload:
            return []

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return []

        tasks: list[TaskSpec] = []
        for item in data.get("tasks", []):
            if not isinstance(item, dict):
                continue
            try:
                task = TaskSpec(
                    task_id=str(item["task_id"]),
                    title=str(item["title"]),
                    description=str(item["description"]),
                    agent_role=str(item["agent_role"]),
                    dependencies=[str(v) for v in item.get("dependencies", [])],
                    tools=[str(v) for v in item.get("tools", [])],
                    output_schema=str(item.get("output_schema", "generic")),
                    retries_allowed=self.settings.orchestration.default_retry_limit,
                )
                tasks.append(task)
            except Exception:
                continue

        return tasks

    def _fallback_plan(self, query: str) -> list[TaskSpec]:
        """Deterministic plan generator for offline/failed planning."""

        q = query.lower()
        research_role = "Market Research Analyst"
        analysis_role = "Business Analyst"
        if any(key in q for key in ["dataset", "csv", "sql", "metrics", "analysis"]):
            analysis_role = "Data Analyst"
        if any(key in q for key in ["finance", "budget", "roi", "cost"]):
            analysis_role = "Financial Analyst"
        if any(key in q for key in ["seo", "keyword", "ranking"]):
            research_role = "SEO Expert"

        return [
            TaskSpec(
                task_id="plan_scope",
                title="Refine scope and constraints",
                description="Turn objective into explicit success criteria and constraints",
                agent_role="Executive Planner",
                tools=["memory_search", "calculator"],
                output_schema="plan_scope",
                retries_allowed=self.settings.orchestration.default_retry_limit,
            ),
            TaskSpec(
                task_id="research",
                title="Collect evidence",
                description="Gather external and internal evidence relevant to objective",
                agent_role=research_role,
                dependencies=["plan_scope"],
                tools=["web_search", "wikipedia", "url_fetcher"],
                output_schema="research_note",
                retries_allowed=self.settings.orchestration.default_retry_limit,
            ),
            TaskSpec(
                task_id="analysis",
                title="Synthesize findings",
                description="Convert evidence into actionable analysis",
                agent_role=analysis_role,
                dependencies=["research"],
                tools=["memory_search", "calculator", "csv_reader"],
                output_schema="analysis",
                retries_allowed=self.settings.orchestration.default_retry_limit,
            ),
            TaskSpec(
                task_id="draft",
                title="Draft deliverable",
                description="Create structured draft response",
                agent_role="Technical Writer",
                dependencies=["analysis"],
                tools=["memory_search", "markdown_reader"],
                output_schema="draft_report",
                retries_allowed=self.settings.orchestration.default_retry_limit,
            ),
            TaskSpec(
                task_id="fact_check",
                title="Fact check and verify",
                description="Validate factual claims and references",
                agent_role="Fact Checker",
                dependencies=["draft"],
                tools=["web_search", "wikipedia", "url_fetcher"],
                output_schema="fact_check_report",
                retries_allowed=self.settings.orchestration.default_retry_limit,
            ),
            TaskSpec(
                task_id="qa",
                title="QA and reflection",
                description="Perform quality review before final report",
                agent_role="QA Agent",
                dependencies=["fact_check"],
                tools=["memory_search", "json_reader"],
                output_schema="qa_report",
                retries_allowed=self.settings.orchestration.default_retry_limit,
            ),
            TaskSpec(
                task_id="final_report",
                title="Generate final report",
                description="Merge verified outputs into final artifacts",
                agent_role="Report Generator",
                dependencies=["qa"],
                tools=["memory_search", "markdown_reader", "json_reader"],
                output_schema="final_report",
                retries_allowed=self.settings.orchestration.default_retry_limit,
            ),
        ]

    def _planner_prompt(self, query: str) -> str:
        roles = ", ".join(self.catalog.roles())
        return (
            "You are Planner Agent for enterprise multi-agent platform.\n"
            "Create dependency-safe task DAG in JSON only.\n"
            "Available roles: "
            f"{roles}.\n"
            "Output schema:\n"
            "{\"tasks\":[{\"task_id\":\"string\",\"title\":\"string\","
            "\"description\":\"string\",\"agent_role\":\"string\","
            "\"dependencies\":[\"task_id\"],\"tools\":[\"tool\"],"
            "\"output_schema\":\"string\"}]}\n"
            "Rules: include verification and reflection tasks.\n"
            f"Objective: {query}\n"
        )

    @staticmethod
    def _extract_json(text: str) -> str:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0) if match else ""
