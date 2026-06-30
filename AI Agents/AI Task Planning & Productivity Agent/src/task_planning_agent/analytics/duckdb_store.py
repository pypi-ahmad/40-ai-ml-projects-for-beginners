"""DuckDB analytics storage and query helpers."""

from __future__ import annotations

from pathlib import Path

import duckdb


class DuckDBStore:
    """Lightweight analytical store for productivity metrics."""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_tables()

    def _init_tables(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analytics_events (
                ts TIMESTAMP,
                user_id VARCHAR,
                plan_id VARCHAR,
                metric VARCHAR,
                value DOUBLE
            )
            """
        )

    def log_metric(self, user_id: str, plan_id: str, metric: str, value: float) -> None:
        self.conn.execute(
            "INSERT INTO analytics_events VALUES(now(), ?, ?, ?, ?)",
            [user_id, plan_id, metric, value],
        )

    def fetch_weekly_summary(self, user_id: str) -> list[dict[str, float]]:
        rows = self.conn.execute(
            """
            SELECT metric, AVG(value) AS avg_value
            FROM analytics_events
            WHERE user_id = ? AND ts > now() - INTERVAL 7 DAY
            GROUP BY metric
            ORDER BY metric
            """,
            [user_id],
        ).fetchall()
        return [{"metric": row[0], "avg_value": float(row[1])} for row in rows]
