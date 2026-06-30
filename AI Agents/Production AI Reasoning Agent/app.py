"""Entrypoint helper for launching Streamlit app."""

from __future__ import annotations

import os
import subprocess
import sys


if __name__ == "__main__":
    env = os.environ.copy()
    # Avoid interactive onboarding prompt during unattended runtime.
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    try:
        raise SystemExit(
            subprocess.call(
                [sys.executable, "-m", "streamlit", "run", "streamlit_app/Home.py"],
                env=env,
            )
        )
    except KeyboardInterrupt:
        raise SystemExit(130)
