"""Entrypoint helper for launching Streamlit dashboard."""

from __future__ import annotations

import os
import subprocess
import sys


if __name__ == "__main__":
    env = os.environ.copy()
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", "streamlit_app/Home.py"],
            env=env,
        )
    )
