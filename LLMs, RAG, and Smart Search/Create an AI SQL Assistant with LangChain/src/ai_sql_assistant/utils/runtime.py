"""Runtime checks and guardrails."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass

from ai_sql_assistant.logging_utils import logger


@dataclass(slots=True)
class RuntimeCheckResult:
    """Result of runtime environment checks."""

    python_ok: bool
    message: str


def ensure_python_version(required: tuple[int, int, int] = (3, 12, 10)) -> RuntimeCheckResult:
    """Validate exact Python runtime as requested by project constraints."""
    current = sys.version_info[:3]
    if current != required:
        return RuntimeCheckResult(
            python_ok=False,
            message=(
                f"Python {required[0]}.{required[1]}.{required[2]} required, "
                f"found {platform.python_version()}."
            ),
        )
    return RuntimeCheckResult(python_ok=True, message="Python version check passed.")


def assert_runtime_or_raise() -> None:
    """Raise runtime error if environment constraints are not met."""
    result = ensure_python_version()
    if not result.python_ok:
        logger.error(result.message)
        raise RuntimeError(result.message)
