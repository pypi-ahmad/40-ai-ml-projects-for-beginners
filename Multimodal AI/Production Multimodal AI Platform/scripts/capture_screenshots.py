"""Capture real screenshots for FastAPI Swagger, Streamlit, and CLI evidence."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path


def _run_background(cmd: list[str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _capture_web_screenshots(output_dir: Path) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Playwright not installed. Run `uv add playwright` and `uv run playwright install`."
        ) from exc

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        page.goto("http://127.0.0.1:8000/docs", wait_until="networkidle")
        page.screenshot(path=str(output_dir / "fastapi_swagger.png"), full_page=True)

        page.goto("http://127.0.0.1:8501", wait_until="networkidle")
        page.screenshot(path=str(output_dir / "streamlit_dashboard.png"), full_page=True)

        browser.close()


def _capture_cli_screenshot(output_dir: Path) -> None:
    command = [
        "uv",
        "run",
        "multimodal-ai",
        "doctor",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    text = "$ " + " ".join(command) + "\n\n" + (result.stdout or "") + (result.stderr or "")
    (output_dir / "cli_doctor.txt").write_text(text, encoding="utf-8")


def main() -> None:
    output_dir = Path("outputs/screenshots")
    output_dir.mkdir(parents=True, exist_ok=True)

    api_proc = _run_background(
        [
            "uv",
            "run",
            "uvicorn",
            "app:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
    )
    ui_proc = _run_background(
        [
            "uv",
            "run",
            "streamlit",
            "run",
            "src/multimodal_ai/ui/streamlit_app.py",
            "--server.address",
            "0.0.0.0",
            "--server.port",
            "8501",
        ]
    )

    try:
        time.sleep(8)
        _capture_web_screenshots(output_dir)
        _capture_cli_screenshot(output_dir)
        print(f"Screenshots saved under {output_dir}")
    finally:
        api_proc.terminate()
        ui_proc.terminate()


if __name__ == "__main__":
    main()
