from __future__ import annotations

from pathlib import Path

from reasoning_agent.tooling.base import ToolContext
from reasoning_agent.tooling.tools.python_repl import PythonREPLInput, run_python


def test_python_repl_runs_safe_code() -> None:
    out = run_python(
        PythonREPLInput(code="print(sum([1, 2, 3]))"),
        ToolContext(session_id="s", run_id="r", workspace_root=Path(".")),
    )
    assert out.ok is True
    assert "6" in out.stdout


def test_python_repl_blocks_unsafe_import() -> None:
    out = run_python(
        PythonREPLInput(code="import os\nprint(os.getcwd())"),
        ToolContext(session_id="s", run_id="r", workspace_root=Path(".")),
    )
    assert out.ok is False
    assert out.error is not None
