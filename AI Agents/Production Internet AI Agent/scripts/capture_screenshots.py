"""Automated screenshot capture for UI/API/monitoring/report pages."""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def capture(base_url: str, api_url: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = {
        "streamlit_home": f"{base_url}",
        "streamlit_chat": f"{base_url}?page=2_Chat",
        "streamlit_search": f"{base_url}?page=3_Search_Explorer",
        "streamlit_memory": f"{base_url}?page=5_Memory",
        "streamlit_analytics": f"{base_url}?page=6_Analytics",
        "streamlit_monitoring": f"{base_url}?page=9_Monitoring",
        "fastapi_docs": f"{api_url}/docs",
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})

        for name, url in pages.items():
            page.goto(url, wait_until="networkidle")
            page.screenshot(path=str(out_dir / f"{name}.png"), full_page=True)
            print(f"saved {name}")

        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8501")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="outputs/screenshots")
    args = parser.parse_args()

    capture(args.base_url, args.api_url, Path(args.output))


if __name__ == "__main__":
    main()
