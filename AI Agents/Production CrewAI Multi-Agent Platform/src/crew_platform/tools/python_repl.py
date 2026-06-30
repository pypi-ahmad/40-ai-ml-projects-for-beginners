"""Restricted Python execution tool."""

from __future__ import annotations

import ast
import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from crew_platform.tools.base import BaseTool


class PythonInput(BaseModel):
    """Python tool input."""

    code: str = Field(min_length=1)


class PythonOutput(BaseModel):
    """Python tool output."""

    success: bool
    stdout: str
    stderr: str
    return_code: int


_ALLOWED_IMPORTS = {
    "math",
    "statistics",
    "datetime",
    "json",
    "re",
    "itertools",
    "collections",
    "functools",
    "decimal",
    "fractions",
    "random",
}

_BLOCKED_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "input",
    "__import__",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "dir",
    "help",
    "breakpoint",
}

_BLOCKED_ATTRIBUTES = {
    "__class__",
    "__bases__",
    "__mro__",
    "__subclasses__",
    "__globals__",
    "__code__",
    "__dict__",
    "__getattribute__",
    "__setattr__",
    "__delattr__",
}

_ALLOWED_BUILTINS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "pow",
    "print",
    "range",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "Exception",
    "ValueError",
    "TypeError",
}


class PythonREPLTool(BaseTool[PythonInput, PythonOutput]):
    """Run restricted Python snippets with timeout and memory limits."""

    name = "python_repl"
    description = "Execute restricted Python code safely"
    input_model = PythonInput
    output_model = PythonOutput

    def __init__(self, timeout_seconds: int = 5, memory_limit_mb: int = 128) -> None:
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb

    async def run(self, payload: PythonInput) -> PythonOutput:
        self._validate_code(payload.code)
        try:
            result = self._run_subprocess(payload.code)
            return PythonOutput(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            return PythonOutput(
                success=False,
                stdout=exc.stdout or "",
                stderr="Execution timed out",
                return_code=124,
            )

    def _validate_code(self, code: str) -> None:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [alias.name.split(".")[0] for alias in node.names]
                for module in modules:
                    if module not in _ALLOWED_IMPORTS:
                        raise ValueError(f"Import not allowed: {module}")
            if isinstance(node, ast.Attribute):
                if node.attr in _BLOCKED_ATTRIBUTES or node.attr.startswith("__"):
                    raise ValueError(f"Attribute access not allowed: {node.attr}")
            if isinstance(node, ast.Name):
                if node.id in _BLOCKED_NAMES or node.id.startswith("__"):
                    raise ValueError(f"Name not allowed: {node.id}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in _BLOCKED_NAMES:
                    raise ValueError(f"Call not allowed: {node.func.id}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in _BLOCKED_ATTRIBUTES or node.func.attr.startswith("__"):
                    raise ValueError(f"Call not allowed: {node.func.attr}")

    def _run_subprocess(self, code: str) -> subprocess.CompletedProcess[str]:
        sandbox_code = (
            "import builtins as _builtins\n"
            f"_ALLOWED_IMPORTS = {sorted(_ALLOWED_IMPORTS)!r}\n"
            f"_ALLOWED_BUILTINS = {sorted(_ALLOWED_BUILTINS)!r}\n"
            "_real_import = _builtins.__import__\n"
            "def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):\n"
            "    root = name.split('.')[0]\n"
            "    if root not in _ALLOWED_IMPORTS:\n"
            "        raise ImportError(f'Import not allowed: {root}')\n"
            "    return _real_import(name, globals, locals, fromlist, level)\n"
            "_sandbox_builtins = {\n"
            "    key: getattr(_builtins, key)\n"
            "    for key in _ALLOWED_BUILTINS\n"
            "    if hasattr(_builtins, key)\n"
            "}\n"
            "_sandbox_builtins['__import__'] = _safe_import\n"
            f"_code = {code!r}\n"
            "exec(compile(_code, '<sandbox>', 'exec'), {'__builtins__': _sandbox_builtins}, {})\n"
        )
        with tempfile.TemporaryDirectory(prefix="pytool-") as tmpdir:
            path = Path(tmpdir) / "snippet.py"
            path.write_text(sandbox_code, encoding="utf-8")

            def _limit_resources() -> None:
                limit_bytes = self.memory_limit_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
                resource.setrlimit(resource.RLIMIT_CPU, (self.timeout_seconds, self.timeout_seconds + 1))
                os.setsid()

            return subprocess.run(
                [sys.executable, "-I", "-S", str(path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                preexec_fn=_limit_resources,
                cwd=tmpdir,
                env={"PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8"},
            )
