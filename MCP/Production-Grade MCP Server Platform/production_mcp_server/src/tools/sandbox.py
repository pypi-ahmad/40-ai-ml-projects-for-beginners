from __future__ import annotations

import ast
import asyncio
import os
import resource
import shlex
import subprocess
import tempfile
from pathlib import Path


_ALLOWED_IMPORTS = {
    "math",
    "statistics",
    "json",
    "datetime",
    "itertools",
    "collections",
    "functools",
    "re",
}

_BLOCKED_NAMES = {
    "__import__",
    "eval",
    "exec",
    "open",
    "compile",
    "input",
    "globals",
    "locals",
    "vars",
    "os",
    "sys",
    "subprocess",
    "socket",
}


def _validate_python_code(code: str) -> None:
    tree = ast.parse(code, mode="exec")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif node.module:
                modules = [node.module.split(".")[0]]
            for module in modules:
                if module not in _ALLOWED_IMPORTS:
                    raise ValueError(f"Import not allowed: {module}")

        if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            raise ValueError(f"Blocked symbol used: {node.id}")

        if isinstance(node, ast.Attribute) and str(node.attr).startswith("__"):
            raise ValueError("Dunder attribute access is not allowed")


async def execute_python_sandboxed(code: str, timeout_seconds: int = 5, memory_limit_mb: int = 128) -> dict:
    _validate_python_code(code)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write(code)
        script_path = Path(handle.name)

    def _preexec() -> None:
        memory_bytes = memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (timeout_seconds, timeout_seconds + 1))
        os.setsid()

    process = await asyncio.create_subprocess_exec(
        "python3",
        "-I",
        str(script_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        preexec_fn=_preexec,
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds + 0.5)
    except TimeoutError:
        process.kill()
        await process.wait()
        return {"ok": False, "error": "Python execution timeout"}
    finally:
        script_path.unlink(missing_ok=True)

    if process.returncode != 0:
        return {"ok": False, "error": stderr.decode("utf-8", errors="ignore").strip()}
    return {"ok": True, "stdout": stdout.decode("utf-8", errors="ignore").strip()}


def run_shell_whitelisted(command: str, allowed: set[str], cwd: str | None = None, timeout: int = 10) -> dict:
    args = shlex.split(command)
    if not args:
        return {"ok": False, "error": "Command is empty"}

    binary = args[0]
    if binary not in allowed:
        return {"ok": False, "error": f"Command not allowed: {binary}"}

    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    if completed.returncode != 0:
        return {
            "ok": False,
            "error": completed.stderr.strip() or f"Exit code: {completed.returncode}",
            "stdout": completed.stdout.strip(),
        }

    return {
        "ok": True,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "returncode": completed.returncode,
    }
