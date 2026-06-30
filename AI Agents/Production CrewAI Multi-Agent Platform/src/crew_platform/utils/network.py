"""Network mode utilities."""

from __future__ import annotations

import os


def offline_mode() -> bool:
    """Return true when runtime should skip external network calls."""

    return os.getenv("AGENT_OFFLINE_MODE", "0") in {"1", "true", "TRUE", "yes", "YES"}
