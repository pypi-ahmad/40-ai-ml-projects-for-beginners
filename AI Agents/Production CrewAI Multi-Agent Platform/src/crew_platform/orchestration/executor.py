"""Dependency-aware multi-agent task executor."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from crew_platform.config import Settings
from crew_platform.llm.ollama import OllamaProvider
from crew_platform.orchestration.catalog import AgentCatalog
from crew_platform.orchestration.crewai_runner import CrewAIRunner
from crew_platform.orchestration.models import TaskExecution, TaskSpec, TaskStatus
from crew_platform.tools.registry import ToolRegistry


class DAGExecutor:
    """Executes task DAG with parallelism, retries, and dynamic delegation."""

    def __init__(
        self,
        settings: Settings,
        llm: OllamaProvider,
        catalog: AgentCatalog,
        tool_registry: ToolRegistry,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.catalog = catalog
        self.tool_registry = tool_registry
        self.use_crewai_execution = settings.orchestration.use_crewai_execution
        self.crewai = (
            CrewAIRunner(
                model=settings.llm.default_model,
                base_url=settings.llm.base_url,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
            )
            if self.use_crewai_execution
            else None
        )

    async def execute(
        self,
        run_id: str,
        objective: str,
        tasks: list[TaskSpec],
        event_log: list[dict[str, Any]],
    ) -> list[TaskExecution]:
        state = {task.task_id: TaskExecution(**task.model_dump()) for task in tasks}
        dependencies = {task.task_id: set(task.dependencies) for task in tasks}
        dependents: dict[str, set[str]] = defaultdict(set)
        for task in tasks:
            for dep in task.dependencies:
                dependents[dep].add(task.task_id)

        semaphore = asyncio.Semaphore(self.settings.orchestration.max_parallel_tasks)

        async def run_task(task_id: str) -> None:
            task = state[task_id]
            async with semaphore:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now(timezone.utc)
                event_log.append({"event": "task_started", "task_id": task_id, "role": task.agent_role})
                for attempt in range(task.retries_allowed + 1):
                    task.attempt = attempt + 1
                    try:
                        task.result = await self._execute_single_task(run_id, objective, task, state)
                        task.status = TaskStatus.COMPLETED
                        task.error = None
                        task.confidence = 0.7 if task.result else 0.4
                        break
                    except Exception as exc:  # noqa: BLE001
                        task.error = str(exc)
                        task.status = TaskStatus.FAILED
                        if attempt >= task.retries_allowed:
                            break
                        await asyncio.sleep(self.settings.orchestration.retry_backoff_seconds)
                task.finished_at = datetime.now(timezone.utc)
                event_log.append(
                    {
                        "event": "task_finished",
                        "task_id": task_id,
                        "status": task.status.value,
                        "attempt": task.attempt,
                    }
                )

        completed: set[str] = set()

        while len(completed) < len(tasks):
            ready = [
                task_id
                for task_id, deps in dependencies.items()
                if task_id not in completed and all(dep in completed for dep in deps)
            ]
            if not ready:
                # Circular dependency or deadlock; mark unfinished tasks as failed.
                for task_id in dependencies:
                    if task_id not in completed:
                        state[task_id].status = TaskStatus.FAILED
                        state[task_id].error = "Dependency deadlock"
                        completed.add(task_id)
                break

            await asyncio.gather(*(run_task(task_id) for task_id in ready))
            completed.update(ready)

        return list(state.values())

    async def _execute_single_task(
        self,
        run_id: str,
        objective: str,
        task: TaskExecution,
        state: dict[str, TaskExecution],
    ) -> dict[str, Any]:
        profile = self.catalog.ensure_role(task.agent_role)
        dep_context = self._dependency_context(task, state)
        tool_context = await self._collect_tool_context(task, run_id)
        prompt = self._build_agent_prompt(objective, profile.goal, task.description, dep_context, tool_context)

        llm_output = ""
        try:
            response = await self.llm.generate(
                prompt=prompt,
                model=self.settings.llm.default_model,
                temperature=self.settings.llm.temperature,
                max_tokens=self.settings.llm.max_tokens,
            )
            llm_output = response.text
        except Exception:
            llm_output = "LLM unavailable. Using CrewAI fallback path."

        if self.crewai is None:
            crew_output = {
                "mode": "disabled",
                "output": "CrewAI execution disabled by config for stable runtime",
            }
        else:
            try:
                crew_output = await asyncio.wait_for(
                    self.crewai.run_task(
                        profile=profile,
                        description=task.description,
                        context=f"Objective: {objective}\nDependencies:\n{dep_context}\nTools:\n{tool_context}",
                    ),
                    timeout=90,
                )
            except TimeoutError:
                crew_output = {"mode": "timeout", "output": "CrewAI execution timed out"}

        merged = {
            "role": profile.role,
            "content": llm_output,
            "crewai_output": crew_output.get("output", ""),
            "tools_used": task.tools,
            "schema": task.output_schema,
        }
        if not merged["content"] and not merged["crewai_output"]:
            raise RuntimeError("Empty agent output")
        return merged

    async def _collect_tool_context(self, task: TaskExecution, run_id: str) -> dict[str, Any]:
        context: dict[str, Any] = {}
        for tool_name in task.tools:
            try:
                if tool_name == "calculator":
                    context[tool_name] = await self.tool_registry.invoke(
                        "calculator", {"expression": "2+2"}, run_id=run_id
                    )
                elif tool_name == "memory_search":
                    context[tool_name] = await self.tool_registry.invoke(
                        "memory_search", {"query": task.description, "top_k": 3}, run_id=run_id
                    )
                elif tool_name == "web_search":
                    context[tool_name] = await self.tool_registry.invoke(
                        "duckduckgo_search", {"query": task.description, "max_results": 3}, run_id=run_id
                    )
            except Exception as exc:  # noqa: BLE001
                context[tool_name] = {"error": str(exc)}
        return context

    @staticmethod
    def _dependency_context(task: TaskExecution, state: dict[str, TaskExecution]) -> str:
        lines: list[str] = []
        for dep in task.dependencies:
            dep_task = state.get(dep)
            if dep_task and dep_task.result:
                lines.append(f"[{dep}] {dep_task.result.get('content', '')[:500]}")
        return "\n".join(lines)

    @staticmethod
    def _build_agent_prompt(
        objective: str,
        goal: str,
        description: str,
        dependency_context: str,
        tool_context: dict[str, Any],
    ) -> str:
        return (
            f"Objective: {objective}\n"
            f"Agent goal: {goal}\n"
            f"Task: {description}\n"
            f"Dependency context:\n{dependency_context}\n"
            f"Tool context:\n{tool_context}\n"
            "Return concise JSON with keys: summary, key_points, risks, references."
        )
