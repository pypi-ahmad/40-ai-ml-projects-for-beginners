"""Background scheduler for periodic reporting."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler

from api_intel_agent.config import load_settings


class SchedulerService:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.scheduler = BackgroundScheduler(timezone="UTC")

    def start(self, generate_report: Callable[[], str]) -> None:
        if not self.settings.scheduler.enabled:
            return

        self.scheduler.add_job(
            lambda: self._run_job(generate_report),
            trigger="cron",
            minute="*/30",
            id="scheduled_report",
            replace_existing=True,
        )
        self.scheduler.start()

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

    @staticmethod
    def _run_job(generate_report: Callable[[], str]) -> str:
        path = generate_report()
        heartbeat = Path("artifacts/reports/scheduler_heartbeat.log")
        heartbeat.parent.mkdir(parents=True, exist_ok=True)
        with heartbeat.open("a") as fh:
            fh.write(f"{datetime.now(UTC).isoformat()} -> {path}\n")
        return path
