"""ICS import/export adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from icalendar import Calendar, Event

from task_planning_agent.schemas import ScheduleBlock


class ICSCalendarAdapter:
    """Read/write scheduling events in ICS format."""

    def import_events(self, path: str) -> list[dict[str, str]]:
        calendar = Calendar.from_ical(Path(path).read_bytes())
        events: list[dict[str, str]] = []
        for component in calendar.walk():
            if component.name != "VEVENT":
                continue
            events.append(
                {
                    "summary": str(component.get("summary", "")),
                    "start": str(component.decoded("dtstart", datetime.now(timezone.utc))),
                    "end": str(component.decoded("dtend", datetime.now(timezone.utc))),
                }
            )
        return events

    def export_schedule(self, path: str, blocks: list[ScheduleBlock]) -> str:
        calendar = Calendar()
        calendar.add("prodid", "-//Task Planning Agent//")
        calendar.add("version", "2.0")

        for block in blocks:
            event = Event()
            event.add("summary", block.task_name)
            event.add("dtstart", block.suggested_start_time)
            event.add("dtend", block.suggested_end_time)
            event.add("description", block.reasoning)
            calendar.add_component(event)

        output_path = Path(path)
        output_path.write_bytes(calendar.to_ical())
        return str(output_path)
