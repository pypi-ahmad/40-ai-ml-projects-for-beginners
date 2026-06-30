from typer.testing import CliRunner

from resume_ai.cli.main import app


runner = CliRunner()


def test_dashboard_command() -> None:
    result = runner.invoke(app, ["dashboard"])
    assert result.exit_code == 0
    assert "streamlit run" in result.stdout
