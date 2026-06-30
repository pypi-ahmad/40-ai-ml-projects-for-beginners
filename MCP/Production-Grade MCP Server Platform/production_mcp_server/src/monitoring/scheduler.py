from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

from config.settings import Settings
from memory.service import MemoryService

logger = logging.getLogger(__name__)


class JobScheduler:
    def __init__(self, settings: Settings, memory: MemoryService) -> None:
        self.settings = settings
        self.memory = memory
        self.scheduler = BackgroundScheduler(timezone=settings.scheduler.timezone)

    def _index_documents(self) -> None:
        root = Path(self.settings.plugins.directory).parents[0]
        count = 0
        for path in root.rglob("*.md"):
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            self.memory.store_semantic_memory(
                text=text,
                metadata={"type": "markdown", "path": str(path)},
                doc_id=f"doc::{path}",
            )
            count += 1
        self.memory.log_usage("periodic_index", {"indexed": count})
        logger.info("Periodic indexing complete: %s docs", count)

    def _cleanup_memory(self) -> None:
        removed = self.memory.cache_cleanup()
        self.memory.log_usage("cache_cleanup", {"removed": removed})
        logger.info("Cache cleanup complete: %s entries", removed)

    def _generate_report(self) -> None:
        now = datetime.now(UTC).isoformat()
        metrics = self.memory.recent_metrics(limit=50)
        body = f"Timestamp: {now}\n\nRecent metrics:\n{metrics[:10]}"
        path = Path(self.settings.plugins.directory).parents[0] / "reports" / "scheduled_report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        self.memory.log_usage("scheduled_report", {"path": str(path)})
        logger.info("Scheduled report generated: %s", path)

    def start(self) -> None:
        if not self.settings.scheduler.enabled:
            logger.info("Scheduler disabled")
            return

        self.scheduler.add_job(self._index_documents, "interval", minutes=self.settings.scheduler.index_every_minutes)
        self.scheduler.add_job(self._cleanup_memory, "interval", minutes=self.settings.scheduler.cleanup_every_minutes)
        self.scheduler.add_job(self._generate_report, "interval", minutes=self.settings.scheduler.report_every_minutes)
        self.scheduler.start()
        logger.info("Scheduler started")

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped")
