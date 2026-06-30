from __future__ import annotations

from typer.testing import CliRunner

from crew_platform.cli.main import app


def test_cli_agents_local() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["agents"])
    assert result.exit_code == 0
    assert "Executive Planner" in result.stdout
