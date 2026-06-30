"""LangGraph node implementations."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from task_planning_agent.analytics.metrics import AnalyticsEngine
from task_planning_agent.dependencies.planner import DependencyPlanner
from task_planning_agent.extraction.extractor import TaskExtractor
from task_planning_agent.memory.manager import MemoryManager
from task_planning_agent.llm.client import LocalLLMClient
from task_planning_agent.llm.model_registry import ModelRegistry
from task_planning_agent.prioritization.registry import score_tasks
from task_planning_agent.recommendations.engine import RecommendationAgent
from task_planning_agent.reflection.reflector import ReflectionAgent
from task_planning_agent.reports.generator import ReportGenerator
from task_planning_agent.scheduling.optimizer import ScheduleOptimizer
from task_planning_agent.schemas import PlanSession, PriorityStrategy, TaskStatus
from task_planning_agent.schemas import Task


class WorkflowNodes:
    """Container for graph node functions and shared runtime dependencies."""

    def __init__(
        self,
        memory: MemoryManager,
        analytics: AnalyticsEngine,
        scheduler: ScheduleOptimizer,
        llm_client: LocalLLMClient | None = None,
        model_registry: ModelRegistry | None = None,
        llm_enabled: bool = False,
    ) -> None:
        self.extractor = TaskExtractor()
        self.dependency_planner = DependencyPlanner()
        self.reflection = ReflectionAgent()
        self.recommendation = RecommendationAgent()
        self.report_generator = ReportGenerator()
        self.memory = memory
        self.analytics = analytics
        self.scheduler = scheduler
        self.llm_client = llm_client
        self.model_registry = model_registry
        self.llm_enabled = llm_enabled

    def planner(self, state: dict[str, object]) -> dict[str, object]:
        raw_input = str(state["raw_input"])
        strategy = state.get("strategy", PriorityStrategy.WSJF)
        if isinstance(strategy, str):
            strategy = PriorityStrategy(strategy)

        extraction = self.extractor.extract(raw_input)
        tasks = score_tasks(extraction.tasks, strategy)
        dependencies, issues = self.dependency_planner.build_dependencies(tasks)
        llm_meta = self._llm_refinement(tasks=tasks, issues=issues)

        return {
            "tasks": tasks,
            "dependencies": dependencies,
            "issues": issues,
            "plan_id": str(uuid4()),
            "llm_meta": llm_meta,
        }

    def _llm_refinement(self, tasks: list[Task], issues: list[str]) -> dict[str, str] | None:
        if not self.llm_enabled or self.llm_client is None or self.model_registry is None:
            return None
        prompt_lines = ["Refine reasoning for productivity tasks. Return concise guidance."] + [
            f"- {task.name} | estimate={task.estimated_minutes} | score={task.priority_score}"
            for task in tasks[:12]
        ]
        prompt = "\n".join(prompt_lines)

        try:
            content, model = self.llm_client.generate(
                prompt=prompt,
                candidate_models=self.model_registry.candidates(),
            )
            guidance = content[:400]
            for task in tasks:
                task.reasoning = f"{task.reasoning} | llm_hint={guidance}"
            return {"model": model}
        except Exception as exc:
            issues.append(f"LLM refinement skipped: {exc}")
            return None

    def scheduler_node(self, state: dict[str, object]) -> dict[str, object]:
        tasks = list(state.get("tasks", []))
        timezone = str(state.get("timezone", "Asia/Kolkata"))
        blocks = self.scheduler.schedule(tasks=tasks, timezone=timezone)

        for task in tasks:
            if any(block.task_id == task.id for block in blocks):
                task.status = TaskStatus.SCHEDULED

        return {"schedule_blocks": blocks, "tasks": tasks}

    def validator(self, state: dict[str, object]) -> dict[str, object]:
        issues = list(state.get("issues", []))
        tasks = list(state.get("tasks", []))
        blocks = list(state.get("schedule_blocks", []))

        if not tasks:
            issues.append("No tasks extracted from input")
        if tasks and not blocks:
            issues.append("Tasks extracted but schedule generation returned no blocks")

        high_risk = [block for block in blocks if block.risk_level.value == "high"]
        if high_risk:
            issues.append(f"{len(high_risk)} high-risk blocks may miss deadlines")

        return {"issues": issues}

    def reflection_node(self, state: dict[str, object]) -> dict[str, object]:
        plan_id = str(state["plan_id"])
        blocks = list(state.get("schedule_blocks", []))
        reflection = self.reflection.reflect(plan_id=plan_id, blocks=blocks)
        recommendations = self.recommendation.suggest(blocks)
        return {"reflection": reflection, "recommendations": recommendations}

    def memory_node(self, state: dict[str, object]) -> dict[str, object]:
        session = PlanSession(
            plan_id=str(state["plan_id"]),
            user_id=str(state["user_id"]),
            raw_input=str(state["raw_input"]),
            tasks=list(state.get("tasks", [])),
            dependencies=list(state.get("dependencies", [])),
            schedule_blocks=list(state.get("schedule_blocks", [])),
            reflection=state.get("reflection"),
            recommendations=list(state.get("recommendations", [])),
        )
        self.memory.persist_plan(session)
        if session.reflection is not None:
            self.memory.sqlite.save_reflection(session.reflection)
        analytics = self.analytics.snapshot(
            user_id=session.user_id,
            plan_id=session.plan_id,
            tasks=session.tasks,
            blocks=session.schedule_blocks,
        )
        return {"analytics": analytics}

    def reporter_node(self, state: dict[str, object]) -> dict[str, object]:
        session = PlanSession(
            plan_id=str(state["plan_id"]),
            user_id=str(state["user_id"]),
            raw_input=str(state["raw_input"]),
            tasks=list(state.get("tasks", [])),
            dependencies=list(state.get("dependencies", [])),
            schedule_blocks=list(state.get("schedule_blocks", [])),
            reflection=state.get("reflection"),
            recommendations=list(state.get("recommendations", [])),
            analytics=state.get("analytics"),
            metadata={"generated_at": datetime.now(timezone.utc).isoformat(), "issues": state.get("issues", [])},
        )
        report = self.report_generator.generate(session)
        return {"report": report}
