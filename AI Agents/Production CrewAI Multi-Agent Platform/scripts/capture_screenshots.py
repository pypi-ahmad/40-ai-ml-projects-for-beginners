"""Capture dashboard/API screenshots for deliverable checklist."""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    out = Path("docs/screenshots")
    out.mkdir(parents=True, exist_ok=True)

    targets = [
        ("http://127.0.0.1:8501", "dashboard.png"),
        ("http://127.0.0.1:8501/Crew_Monitor", "agent_monitor.png"),
        ("http://127.0.0.1:8501/Live_Workflow", "workflow_graph.png"),
        ("http://127.0.0.1:8501/Tasks", "task_planner.png"),
        ("http://127.0.0.1:8501/Memory", "memory_viewer.png"),
        ("http://127.0.0.1:8501/Knowledge_Base", "knowledge_base.png"),
        ("http://127.0.0.1:8501/Reports", "reports.png"),
        ("http://127.0.0.1:8501/Analytics", "analytics.png"),
        ("http://127.0.0.1:8000/docs", "fastapi_swagger.png"),
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        for url, filename in targets:
            page.goto(url, wait_until="networkidle", timeout=60_000)
            page.screenshot(path=str(out / filename), full_page=True)
        browser.close()


if __name__ == "__main__":
    main()
