from __future__ import annotations

import pytest

pytest.importorskip("typer")
from typer.testing import CliRunner

from peft_platform.cli.main import app


runner = CliRunner()


def test_cli_models() -> None:
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "TinyLlama" in result.stdout


def test_cli_dataset_smoke() -> None:
    result = runner.invoke(app, ["dataset-smoke"])
    assert result.exit_code == 0
    assert "dataset size=" in result.stdout
