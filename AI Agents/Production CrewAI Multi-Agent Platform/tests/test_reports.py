from __future__ import annotations

from pathlib import Path

from crew_platform.config import load_settings
from crew_platform.orchestration.models import TaskExecution, VerificationResult
from crew_platform.reports import ReportGenerator


def test_report_generator_outputs_files(tmp_path: Path) -> None:
    settings = load_settings("configs/settings.yaml")
    settings.reports.output_dir = str(tmp_path)
    generator = ReportGenerator(settings)

    tasks = [
        TaskExecution(
            task_id="t1",
            title="Task",
            description="desc",
            agent_role="Technical Writer",
            dependencies=[],
            tools=[],
            output_schema="x",
            result={"content": "Summary with http://example.com"},
            status="completed",
        )
    ]
    verification = VerificationResult(confidence=0.8)

    artifact = generator.generate("run-test", "objective", tasks, verification)

    assert (tmp_path / "run-test.md").exists()
    assert (tmp_path / "run-test.json").exists()
    assert artifact.references
