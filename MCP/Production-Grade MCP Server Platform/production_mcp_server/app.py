from __future__ import annotations

import os

import streamlit.web.cli as stcli


def main() -> None:
    streamlit_app = os.path.join("src", "ui", "streamlit_app", "Home.py")
    stcli.main_run([streamlit_app, "--server.address", "0.0.0.0", "--server.port", "8501"], standalone_mode=False)


if __name__ == "__main__":
    main()
