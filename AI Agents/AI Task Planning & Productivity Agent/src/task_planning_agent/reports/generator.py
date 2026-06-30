"""Report generation helpers."""

from __future__ import annotations

from datetime import datetime, timezone

from task_planning_agent.schemas import PlanReport, PlanSession, Recommendation


class ReportGenerator:
    """Build user-facing report objects from plan sessions."""

    def generate(self, session: PlanSession) -> PlanReport:
        schedule = [
            {
                "task": block.task_name,
                "priority": block.priority,
                "deadline": next(
                    (task.deadline for task in session.tasks if task.id == block.task_id),
                    None,
                ),
                "estimated_duration": int(
                    (block.suggested_end_time - block.suggested_start_time).total_seconds() / 60
                ),
                "dependencies": next(
                    (task.dependencies for task in session.tasks if task.id == block.task_id),
                    [],
                ),
                "suggested_start_time": block.suggested_start_time,
                "suggested_end_time": block.suggested_end_time,
                "confidence": block.confidence,
                "reasoning": block.reasoning,
                "risk_level": block.risk_level,
            }
            for block in session.schedule_blocks
        ]

        reflections = session.reflection.recommendations if session.reflection else []
        recommendations = [
            Recommendation(category=item.category, suggestion=item.suggestion, impact=item.impact)
            for item in session.recommendations
        ]

        return PlanReport(
            plan_id=session.plan_id,
            user_id=session.user_id,
            generated_at=datetime.now(timezone.utc),
            summary=f"Generated {len(session.schedule_blocks)} blocks for {len(session.tasks)} tasks",
            schedule=schedule,  # type: ignore[arg-type]
            reflections=reflections,
            recommendations=recommendations,
            analytics=session.analytics,
        )
