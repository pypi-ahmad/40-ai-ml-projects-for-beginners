from task_planning_agent.dependencies.planner import DependencyPlanner
from task_planning_agent.schemas import Task


def test_dependency_cycle_detection() -> None:
    a = Task(id="a", name="A", dependencies=["b"])
    b = Task(id="b", name="B", dependencies=["a"])
    deps, issues = DependencyPlanner().build_dependencies([a, b])
    assert deps
    assert any("cycle" in issue.lower() for issue in issues)
