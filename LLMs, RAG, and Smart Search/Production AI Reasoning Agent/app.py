"""Streamlit launcher for Production AI Reasoning Agent UI."""

from __future__ import annotations

import subprocess
import sys

if __name__ == "__main__":
    cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_app/Home.py"]
    raise SystemExit(subprocess.call(cmd))
