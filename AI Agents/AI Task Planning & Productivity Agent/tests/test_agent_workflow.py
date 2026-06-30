from task_planning_agent.agent.service import PlanningService
from task_planning_agent.config import AppConfig
from task_planning_agent.schemas import PriorityStrategy


def test_agent_workflow_generates_report(tmp_path) -> None:
    cfg = AppConfig(
        raw={
            "paths": {
                "sqlite_path": str(tmp_path / "db.sqlite"),
                "chroma_dir": str(tmp_path / "chroma"),
            },
            "memory": {"chroma_collection": "test_collection"},
            "scheduling": {
                "workday_start": "09:00",
                "workday_end": "18:00",
                "default_break_minutes": 5,
                "deep_work_block_minutes": 90,
            },
            "analytics": {"mlflow_tracking_uri": f"file:{tmp_path / 'mlruns'}"},
        }
    )
    service = PlanningService(cfg)
    report = service.plan(
        user_id="u1",
        raw_input="- Finish API docs by tomorrow 60min",
        strategy=PriorityStrategy.WSJF,
        timezone="UTC",
    )
    assert report.plan_id
    assert report.schedule
