"""High-level collaboration service orchestrating planning, execution, verification, and reporting."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from crew_platform.config import Settings, load_agent_catalog
from crew_platform.config.models import AgentProfile
from crew_platform.llm.ollama import OllamaProvider
from crew_platform.memory.persistence import PersistenceStore
from crew_platform.memory.runtime import RuntimeMemory
from crew_platform.orchestration.catalog import AgentCatalog
from crew_platform.orchestration.consensus import ConsensusService
from crew_platform.orchestration.executor import DAGExecutor
from crew_platform.orchestration.models import (
    CrewRunRequest,
    CrewRunResult,
    PlanApproval,
    RunStatus,
    TaskExecution,
)
from crew_platform.orchestration.planner import PlannerService
from crew_platform.orchestration.verification import VerificationService
from crew_platform.reports.generator import ReportGenerator
from crew_platform.tools.factory import create_default_registry
from crew_platform.observability.metrics import MetricsStore
from crew_platform.observability.mlflow_tracker import MLflowTracker
from crew_platform.observability.tracer import JsonlTracer
from crew_platform.plugins import PluginLoader


@dataclass(slots=True)
class RunRecord:
    result: CrewRunResult
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    approved: bool = False


class CollaborationService:
    """Main application service backing API, CLI, and dashboard clients."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        Path(settings.reports.output_dir).mkdir(parents=True, exist_ok=True)
        Path(Path(settings.logging.run_log_path).parent).mkdir(parents=True, exist_ok=True)

        self.llm = OllamaProvider(
            base_url=settings.llm.base_url,
            timeout_seconds=settings.llm.request_timeout_seconds,
        )
        self.catalog = AgentCatalog(load_agent_catalog())
        self._load_marketplace_agents()
        self.persistence = PersistenceStore(settings.memory.sqlite_path)
        self.runtime_memory = RuntimeMemory(settings)

        self.metrics = MetricsStore()
        tracer = JsonlTracer(Path(settings.logging.run_log_path)) if settings.logging.json_logs else None

        self.tool_registry = create_default_registry(
            workspace_root=Path(settings.tools.workspace_root),
            tracer=tracer,
            metrics=self.metrics,
            memory_store=self.runtime_memory.semantic_store,
            python_timeout=settings.tools.python_timeout_seconds,
            python_memory_mb=settings.tools.python_memory_limit_mb,
            optional_tools=settings.tools.optional_tools,
            enabled_tools=settings.tools.enabled_tools,
            enable_python_tool=settings.tools.enable_python_tool,
        )

        self.planner = PlannerService(settings, self.llm, self.catalog)
        self.executor = DAGExecutor(settings, self.llm, self.catalog, self.tool_registry)
        self.verifier = VerificationService(settings, self.llm)
        self.consensus = ConsensusService(settings, self.llm)
        self.report_generator = ReportGenerator(settings)
        self.mlflow = MLflowTracker()

        self.runs: dict[str, RunRecord] = {}

    async def create_plan(self, request: CrewRunRequest) -> CrewRunResult:
        plan = await self.planner.build_plan(request.query)
        status = RunStatus.AWAITING_APPROVAL if self.settings.orchestration.plan_approval_required else RunStatus.PLANNED
        result = CrewRunResult(run_id=plan.run_id, status=status, plan=plan)
        self.runs[plan.run_id] = RunRecord(result=result, approved=not self.settings.orchestration.plan_approval_required)

        self.persistence.save_plan(plan.run_id, request.session_id, request.query, plan.model_dump(mode="json"))
        logger.info("Created plan run_id={} tasks={}", plan.run_id, len(plan.tasks))

        if request.auto_execute and (self.runs[plan.run_id].approved or not self.settings.orchestration.plan_approval_required):
            return await self.execute_run(plan.run_id)
        return result

    async def apply_approval(self, run_id: str, approval: PlanApproval) -> CrewRunResult:
        record = self._must_get_run(run_id)
        record.approved = approval.approved

        if not approval.approved:
            record.result.status = RunStatus.REJECTED
            if approval.feedback:
                record.result.events.append({"event": "plan_rejected", "feedback": approval.feedback})
            self.persistence.save_approval(run_id, approval.reviewer, approval.approved, approval.feedback)
            return record.result

        record.result.status = RunStatus.PLANNED
        record.result.events.append({"event": "plan_approved", "reviewer": approval.reviewer})
        self.persistence.save_approval(run_id, approval.reviewer, approval.approved, approval.feedback)
        return record.result

    async def execute_run(self, run_id: str, force_consensus: bool = False) -> CrewRunResult:
        record = self._must_get_run(run_id)
        if self.settings.orchestration.plan_approval_required and not record.approved:
            record.result.status = RunStatus.AWAITING_APPROVAL
            record.result.error = "Run is awaiting human approval"
            return record.result

        record.result.status = RunStatus.RUNNING
        event_log = record.result.events

        tasks = await self.executor.execute(
            run_id=run_id,
            objective=record.result.plan.objective,
            tasks=record.result.plan.tasks,
            event_log=event_log,
        )
        record.result.tasks = tasks
        self._persist_task_outputs(run_id, tasks)

        verification = await self.verifier.verify(tasks)
        record.result.verification = verification

        if verification.needs_rerun:
            event_log.append({"event": "verification_low_confidence", "confidence": verification.confidence})

        should_consensus = force_consensus or (
            self.settings.orchestration.consensus_enabled
            and verification.confidence < self.settings.orchestration.consensus_trigger_confidence
        )
        if should_consensus:
            context = "\n".join(str(task.result or "") for task in tasks)
            consensus = await self.consensus.run(record.result.plan.objective, context)
            event_log.append(
                {
                    "event": "consensus_completed",
                    "selected_answer": consensus.selected_answer[:200],
                }
            )

        report = self.report_generator.generate(
            run_id=run_id,
            objective=record.result.plan.objective,
            tasks=tasks,
            verification=verification,
        )
        record.result.report = report

        self.persistence.save_report(run_id, report.model_dump(mode="json"))
        self.runtime_memory.remember_run(run_id, record.result.plan.objective, report.summary)

        record.result.metrics = self._build_metrics(tasks, verification.confidence)
        self.mlflow.log_run(
            run_name=run_id,
            params={
                "objective": record.result.plan.objective,
                "task_count": len(tasks),
                "consensus_enabled": self.settings.orchestration.consensus_enabled,
            },
            metrics=record.result.metrics,
        )
        record.result.status = RunStatus.COMPLETED if all(t.status.value == "completed" for t in tasks) else RunStatus.FAILED
        return record.result

    def list_agents(self) -> list[dict[str, Any]]:
        return [agent.model_dump(mode="json") for agent in self.catalog.all()]

    def list_runs(self) -> list[CrewRunResult]:
        return [record.result for record in self.runs.values()]

    def get_run(self, run_id: str) -> CrewRunResult | None:
        record = self.runs.get(run_id)
        return record.result if record else None

    def pause_run(self, run_id: str) -> CrewRunResult:
        record = self._must_get_run(run_id)
        record.result.status = RunStatus.PAUSED
        record.result.events.append({"event": "paused"})
        return record.result

    def resume_run(self, run_id: str) -> CrewRunResult:
        record = self._must_get_run(run_id)
        if record.result.status == RunStatus.PAUSED:
            record.result.status = RunStatus.PLANNED
            record.result.events.append({"event": "resumed"})
        return record.result

    async def rerun_task(self, run_id: str, task_id: str, feedback: str | None = None) -> CrewRunResult:
        record = self._must_get_run(run_id)
        task = next((t for t in record.result.tasks if t.task_id == task_id), None)
        if task is None:
            raise KeyError(f"Unknown task_id {task_id}")

        task.status = task.status.PENDING
        if feedback:
            record.result.events.append({"event": "task_feedback", "task_id": task_id, "feedback": feedback})

        rerun = await self.executor.execute(
            run_id=run_id,
            objective=record.result.plan.objective,
            tasks=record.result.plan.tasks,
            event_log=record.result.events,
        )
        record.result.tasks = rerun
        return record.result

    async def chat(self, query: str, session_id: str = "default") -> dict[str, Any]:
        request = CrewRunRequest(query=query, session_id=session_id, auto_execute=False)
        planned = await self.create_plan(request)
        if self.settings.orchestration.plan_approval_required:
            await self.apply_approval(planned.run_id, PlanApproval(approved=True, reviewer="auto_chat"))
        result = await self.execute_run(planned.run_id)
        return {
            "run_id": result.run_id,
            "answer": result.report.summary if result.report else "",
            "confidence": result.verification.confidence if result.verification else 0.0,
            "status": result.status.value,
        }

    async def search(self, query: str) -> dict[str, Any]:
        try:
            out = await self.tool_registry.invoke(
                "duckduckgo_search",
                {"query": query, "max_results": 5},
                run_id=f"search-{datetime.now(timezone.utc).timestamp()}",
            )
            return out
        except Exception as exc:  # noqa: BLE001
            return {"results": [], "error": str(exc)}

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "runs": len(self.runs),
            "agents": len(self.catalog.all()),
            "sqlite_path": self.settings.memory.sqlite_path,
        }

    def metrics_snapshot(self) -> dict[str, Any]:
        return self.metrics.snapshot()

    def _must_get_run(self, run_id: str) -> RunRecord:
        if run_id not in self.runs:
            raise KeyError(f"Unknown run_id {run_id}")
        return self.runs[run_id]

    def _persist_task_outputs(self, run_id: str, tasks: list[TaskExecution]) -> None:
        for task in tasks:
            self.persistence.save_task(
                run_id=run_id,
                task_id=task.task_id,
                agent_role=task.agent_role,
                status=task.status.value,
                payload=(task.result or {}),
                error=task.error,
                attempt=task.attempt,
            )

    @staticmethod
    def _build_metrics(tasks: list[TaskExecution], confidence: float) -> dict[str, float]:
        total = len(tasks)
        completed = sum(task.status.value == "completed" for task in tasks)
        failed = sum(task.status.value == "failed" for task in tasks)
        retries = sum(max(0, task.attempt - 1) for task in tasks)
        return {
            "task_total": float(total),
            "task_completed": float(completed),
            "task_failed": float(failed),
            "retries": float(retries),
            "confidence": float(confidence),
        }

    def _load_marketplace_agents(self) -> None:
        loader = PluginLoader(plugin_dir="plugins")
        manifests = loader.manifests()
        for manifest in manifests:
            for item in manifest.get("agents", []):
                try:
                    profile = AgentProfile.model_validate(item)
                    if self.catalog.get_by_role(profile.role) is None:
                        self.catalog.config.agents.append(profile)
                except Exception:
                    continue


def create_service(settings: Settings) -> CollaborationService:
    """Create initialized collaboration service."""

    return CollaborationService(settings)


async def run_sync(coro):
    """Run async coroutine from sync contexts (CLI/tests)."""

    return await coro
