"""Restricted Python execution tool with timeout and safety guards."""

from __future__ import annotations

import ast
import io
import signal
import traceback
from contextlib import redirect_stdout

from pydantic import BaseModel, Field

from reasoning_agent.tooling.base import ToolContext, ToolSpec


class PythonREPLInput(BaseModel):
    """Python REPL input payload."""

    code: str = Field(min_length=1, max_length=4000)
    timeout_seconds: int = Field(default=5, ge=1, le=20)
    memory_mb: int = Field(default=512, ge=64, le=2048)


class PythonREPLOutput(BaseModel):
    """Python REPL output payload."""

    ok: bool
    stdout: str
    error: str | None = None


_ALLOWED_IMPORTS = {
    "math",
    "statistics",
    "random",
    "itertools",
    "functools",
    "collections",
    "datetime",
    "json",
}

_BLOCKED_NODES = {
    ast.Attribute,
    ast.With,
    ast.AsyncWith,
    ast.Lambda,
    ast.ClassDef,
    ast.Global,
    ast.Nonlocal,
    ast.Try,
    ast.Raise,
}


def _validate_python(code: str) -> None:
    tree = ast.parse(code, mode="exec")

    for node in ast.walk(tree):
        if type(node) in _BLOCKED_NODES:
            raise ValueError(f"Unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in _ALLOWED_IMPORTS:
                    raise ValueError(f"Import blocked: {alias.name}")
        if isinstance(node, ast.ImportFrom):
            if node.module is None or node.module.split(".")[0] not in _ALLOWED_IMPORTS:
                raise ValueError(f"Import blocked: {node.module}")


def run_python(payload: PythonREPLInput, _: ToolContext) -> PythonREPLOutput:
    """Execute guarded Python snippet.

    Notes:
        `memory_mb` is reserved for future process-isolated runner; current implementation
        enforces strict AST/import guards and execution timeout.
    """

    try:
        _validate_python(payload.code)
    except Exception as exc:
        return PythonREPLOutput(ok=False, stdout="", error=str(exc))

    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "round": round,
        "set": set,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    globals_ns = {"__builtins__": safe_builtins}
    locals_ns: dict[str, object] = {}

    stream = io.StringIO()

    def _timeout_handler(_signum: int, _frame: object) -> None:
        raise TimeoutError("Execution timeout")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, float(payload.timeout_seconds))

    try:
        with redirect_stdout(stream):
            exec(payload.code, globals_ns, locals_ns)  # noqa: S102 - sandboxed builtins + AST guard
        return PythonREPLOutput(ok=True, stdout=stream.getvalue())
    except TimeoutError as exc:
        return PythonREPLOutput(ok=False, stdout=stream.getvalue(), error=str(exc))
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=1)}"
        return PythonREPLOutput(ok=False, stdout=stream.getvalue(), error=err)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


spec = ToolSpec(
    name="python_repl",
    description="Restricted Python execution with AST/import guards and timeout",
    input_model=PythonREPLInput,
    output_model=PythonREPLOutput,
    tags=["python", "code", "sandbox"],
)
