"""Deadline-aware schedule optimizer."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from task_planning_agent.schemas import RiskLevel, ScheduleBlock, Task


class ScheduleOptimizer:
    """Generate daily and weekly schedules with lightweight heuristics."""

    def __init__(
        self,
        workday_start: str = "09:00",
        workday_end: str = "18:00",
        break_minutes: int = 15,
        deep_work_block_minutes: int = 90,
    ) -> None:
        self.workday_start = workday_start
        self.workday_end = workday_end
        self.break_minutes = break_minutes
        self.deep_work_block_minutes = deep_work_block_minutes

    def schedule(self, tasks: list[Task], timezone: str = "Asia/Kolkata") -> list[ScheduleBlock]:
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)
        day_cursor = now.replace(
            hour=int(self.workday_start.split(":")[0]),
            minute=int(self.workday_start.split(":")[1]),
            second=0,
            microsecond=0,
        )

        blocks: list[ScheduleBlock] = []
        for task in sorted(tasks, key=lambda t: t.priority_score, reverse=True):
            duration = timedelta(minutes=max(15, task.estimated_minutes))
            if day_cursor.time() >= time(
                hour=int(self.workday_end.split(":")[0]),
                minute=int(self.workday_end.split(":")[1]),
            ):
                day_cursor = (day_cursor + timedelta(days=1)).replace(
                    hour=int(self.workday_start.split(":")[0]),
                    minute=int(self.workday_start.split(":")[1]),
                )

            start = day_cursor
            end = start + duration

            if task.deadline and end.replace(tzinfo=None) > task.deadline:
                task.risk_level = RiskLevel.HIGH
                task.reasoning = f"{task.reasoning} | projected completion after deadline"

            block = ScheduleBlock(
                task_id=task.id,
                task_name=task.name,
                suggested_start_time=start,
                suggested_end_time=end,
                priority=task.priority_score,
                confidence=task.confidence,
                reasoning=task.reasoning,
                risk_level=task.risk_level,
            )
            blocks.append(block)

            # Insert small recovery break between heavy tasks.
            day_cursor = end + timedelta(minutes=self.break_minutes)

        return blocks
