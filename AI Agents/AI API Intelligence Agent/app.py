"""Streamlit launcher entrypoint."""

from __future__ import annotations

import subprocess
import sys


if __name__ == "__main__":
    try:
        code = subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "streamlit_app/Home.py",
                "--server.address",
                "0.0.0.0",
                "--server.port",
                "8501",
            ]
        )
    except KeyboardInterrupt:
        code = 0
    raise SystemExit(code)
