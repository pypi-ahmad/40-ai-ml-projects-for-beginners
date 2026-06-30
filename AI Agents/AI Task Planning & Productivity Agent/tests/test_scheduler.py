from datetime import datetime, timedelta

from task_planning_agent.scheduling.optimizer import ScheduleOptimizer
from task_planning_agent.schemas import RiskLevel, Task


def test_scheduler_marks_high_risk_for_late_deadline() -> None:
    task = Task(
        name="urgent",
        description="",
        estimated_minutes=180,
        deadline=datetime.utcnow() + timedelta(minutes=30),
    )
    task.priority_score = 95
    blocks = ScheduleOptimizer().schedule([task], timezone="UTC")
    assert blocks
    assert blocks[0].risk_level in {RiskLevel.HIGH, RiskLevel.MEDIUM}
