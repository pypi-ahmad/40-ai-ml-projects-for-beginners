from datetime import datetime, timedelta
from pathlib import Path

from task_planning_agent.calendar.service import CalendarService
from task_planning_agent.schemas import RiskLevel, ScheduleBlock


def test_calendar_export_import_ics(tmp_path: Path) -> None:
    service = CalendarService()
    block = ScheduleBlock(
        task_id="1",
        task_name="Task",
        suggested_start_time=datetime.utcnow(),
        suggested_end_time=datetime.utcnow() + timedelta(minutes=30),
        priority=80,
        confidence=0.8,
        reasoning="",
        risk_level=RiskLevel.LOW,
    )
    output = service.export_ics(str(tmp_path / "schedule.ics"), [block])
    assert Path(output).exists()
    events = service.import_ics(output)
    assert len(events) >= 1
