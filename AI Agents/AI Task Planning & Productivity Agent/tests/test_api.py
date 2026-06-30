from starlette.testclient import TestClient

from task_planning_agent.api.app import create_app
from task_planning_agent.api.deps import get_service
from task_planning_agent.config import AppConfig
from task_planning_agent.agent.service import PlanningService


class _ServiceFactory:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path

    def __call__(self):
        cfg = AppConfig(
            raw={
                "paths": {
                    "sqlite_path": str(self.tmp_path / "db.sqlite"),
                    "chroma_dir": str(self.tmp_path / "chroma"),
                },
                "memory": {"chroma_collection": "api_collection"},
                "scheduling": {
                    "workday_start": "09:00",
                    "workday_end": "18:00",
                    "default_break_minutes": 5,
                    "deep_work_block_minutes": 90,
                },
                "analytics": {"mlflow_tracking_uri": f"file:{self.tmp_path / 'mlruns'}"},
                "security": {"jwt_algorithm": "HS256", "token_expiry_minutes": 60},
            }
        )
        return PlanningService(cfg)


def test_health_endpoint(tmp_path) -> None:
    app = create_app()
    app.dependency_overrides[get_service] = _ServiceFactory(tmp_path)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
