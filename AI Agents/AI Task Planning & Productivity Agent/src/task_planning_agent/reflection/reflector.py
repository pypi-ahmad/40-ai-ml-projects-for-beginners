"""Reflection agent for schedule quality feedback."""

from __future__ import annotations

from datetime import datetime

from task_planning_agent.schemas import ReflectionRecord, ScheduleBlock


class ReflectionAgent:
    """Generate post-plan reflection insights."""

    def reflect(self, plan_id: str, blocks: list[ScheduleBlock]) -> ReflectionRecord:
        worked: list[str] = []
        failed: list[str] = []
        missed: list[str] = []
        overruns: list[str] = []
        switching: list[str] = []

        context_changes = 0
        prev_project_token = ""
        for block in blocks:
            token = block.task_name.split(" ")[0].lower()
            if prev_project_token and token != prev_project_token:
                context_changes += 1
            prev_project_token = token

            if block.risk_level.value == "high":
                missed.append(block.task_name)
            else:
                worked.append(block.task_name)

            if (block.suggested_end_time - block.suggested_start_time).total_seconds() > 3 * 3600:
                overruns.append(block.task_name)

        if context_changes > max(1, len(blocks) // 3):
            switching.append("High context-switch count; batch similar tasks")

        if not blocks:
            failed.append("No schedule blocks generated")

        recs = [
            "Protect first deep-work block from meetings.",
            "Cap daily context switches to <= 5.",
            "Re-estimate repeated overrun tasks by +20%.",
        ]

        return ReflectionRecord(
            plan_id=plan_id,
            what_worked=worked[:10],
            what_failed=failed,
            missed_deadlines=missed,
            overruns=overruns,
            context_switching_issues=switching,
            recommendations=recs,
            created_at=datetime.utcnow(),
        )
