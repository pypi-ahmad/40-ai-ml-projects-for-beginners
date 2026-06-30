from task_planning_agent.prioritization.registry import score_tasks
from task_planning_agent.schemas import PriorityStrategy, Task


def test_wsjf_prioritization_sorts_descending() -> None:
    tasks = [
        Task(name="low", description="", estimated_minutes=120),
        Task(name="high", description="must do critical", estimated_minutes=30),
    ]
    scored = score_tasks(tasks, PriorityStrategy.WSJF)
    assert scored[0].priority_score >= scored[1].priority_score
