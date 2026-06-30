"""Streamlit entry helper."""

from __future__ import annotations

import subprocess

if __name__ == "__main__":
    subprocess.run(["streamlit", "run", "streamlit_app/Home.py"], check=False)
