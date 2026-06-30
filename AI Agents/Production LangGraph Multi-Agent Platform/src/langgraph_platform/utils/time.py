"""Time and ID helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4


def utc_now() -> datetime:
    """Return UTC-aware now timestamp."""

    return datetime.now(UTC)


def new_workflow_id() -> str:
    """Generate workflow ID."""

    return f"wf_{uuid4().hex[:12]}"


def new_session_id() -> str:
    """Generate session ID."""

    return f"session_{uuid4().hex[:10]}"
