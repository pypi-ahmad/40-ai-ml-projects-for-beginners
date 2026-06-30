"""Calendar orchestration service."""

from __future__ import annotations

from datetime import datetime

from task_planning_agent.calendar.adapters.google_stub import GoogleCalendarAdapterStub
from task_planning_agent.calendar.adapters.ics import ICSCalendarAdapter
from task_planning_agent.schemas import ScheduleBlock


class CalendarService:
    """Calendar imports/exports and conflict checks."""

    def __init__(self) -> None:
        self.ics = ICSCalendarAdapter()
        self.google = GoogleCalendarAdapterStub()

    def import_ics(self, path: str) -> list[dict[str, str]]:
        return self.ics.import_events(path)

    def export_ics(self, path: str, blocks: list[ScheduleBlock]) -> str:
        return self.ics.export_schedule(path, blocks)

    def detect_conflicts(
        self,
        candidate_blocks: list[ScheduleBlock],
        existing_events: list[dict[str, str]],
    ) -> list[str]:
        conflicts: list[str] = []
        parsed_events: list[tuple[datetime, datetime, str]] = []
        for event in existing_events:
            try:
                start = datetime.fromisoformat(event["start"])
                end = datetime.fromisoformat(event["end"])
            except ValueError:
                continue
            parsed_events.append((start, end, event.get("summary", "event")))

        for block in candidate_blocks:
            for start, end, summary in parsed_events:
                if block.suggested_start_time < end and block.suggested_end_time > start:
                    conflicts.append(
                        f"Conflict: {block.task_name} overlaps with {summary} ({start.isoformat()})"
                    )
        return conflicts

    def available_slots(
        self,
        date: datetime,
        existing_events: list[dict[str, str]],
        start_hour: int = 9,
        end_hour: int = 18,
    ) -> list[dict[str, datetime]]:
        slots: list[dict[str, datetime]] = []
        day_start = date.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        day_end = date.replace(hour=end_hour, minute=0, second=0, microsecond=0)
        pointer = day_start

        busy: list[tuple[datetime, datetime]] = []
        for event in existing_events:
            try:
                busy.append((datetime.fromisoformat(event["start"]), datetime.fromisoformat(event["end"])))
            except ValueError:
                continue

        while pointer < day_end:
            next_pointer = pointer.replace(minute=pointer.minute + 30 if pointer.minute < 30 else 0)
            if next_pointer <= pointer:
                next_pointer = pointer + (day_end - pointer)
            overlap = any(pointer < end and next_pointer > start for start, end in busy)
            if not overlap:
                slots.append({"start": pointer, "end": next_pointer})
            pointer = pointer + (next_pointer - pointer)
        return slots
