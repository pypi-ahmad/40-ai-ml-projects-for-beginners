"""Conversational analytics assistant with deterministic-first execution."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from ai_spreadsheet_analytics.analytics import AnalyticsEngine
from ai_spreadsheet_analytics.insights import InsightEngine
from ai_spreadsheet_analytics.schemas import ChatTurn, InsightPacket
from ai_spreadsheet_analytics.state_store import SQLiteStateStore


class ConversationalAnalyticsAssistant:
    """Tool-first conversational analytics."""

    def __init__(
        self,
        analytics_engine: AnalyticsEngine,
        insight_engine: InsightEngine,
        state_store: SQLiteStateStore,
        model: str,
    ) -> None:
        self.analytics_engine = analytics_engine
        self.insight_engine = insight_engine
        self.state_store = state_store
        self.model = model

    def ask(self, question: str, df: pd.DataFrame, session_id: str, role: str = "executive") -> ChatTurn:
        """Answer question with deterministic evidence then LLM narrative."""
        evidence = self._deterministic_answer(question, df)
        try:
            packet: InsightPacket = self.insight_engine.generate(
                analytics_payload={"question": question, "evidence": evidence},
                role=role,
                model=self.model,
            )
            answer = packet.summary
            latency = packet.latency_ms
        except Exception as exc:  # noqa: BLE001
            answer = (
                "LLM unavailable. Deterministic answer payload returned. "
                f"Evidence summary: {str(evidence)[:1200]}. Error: {exc}"
            )
            latency = 0.0

        self.state_store.add_chat_turn(
            session_id=session_id,
            question=question,
            answer=answer,
            evidence=evidence,
            model=self.model,
            latency_ms=latency,
        )

        return ChatTurn(
            session_id=session_id,
            question=question,
            answer=answer,
            evidence=evidence,
            model=self.model,
            latency_ms=latency,
        )

    def history(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Return persisted chat history."""
        return self.state_store.get_chat_history(session_id, limit=limit)

    def _deterministic_answer(self, question: str, df: pd.DataFrame) -> dict[str, Any]:
        q = question.lower()
        value_col = self.analytics_engine._detect_value_column(df)

        if "best product" in q or "top product" in q:
            product_col = next((c for c in df.columns if "product" in c.lower() or "item" in c.lower()), None)
            if product_col and value_col:
                work = df.copy()
                work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
                result = (
                    work.groupby(product_col, as_index=False)[value_col]
                    .sum()
                    .sort_values(value_col, ascending=False)
                    .head(10)
                )
                return {"type": "top_products", "rows": result.to_dict(orient="records")}

        if "highest revenue month" in q or "highest month" in q:
            ts = self.analytics_engine.time_series_summary(df)
            if ts.get("available"):
                monthly = pd.DataFrame(ts["monthly"])
                idx = monthly[value_col].idxmax()  # type: ignore[index]
                row = monthly.loc[idx].to_dict()
                return {"type": "peak_month", "row": row}

        if "declining" in q or "drop" in q:
            ts = self.analytics_engine.time_series_summary(df)
            if ts.get("available"):
                monthly = pd.DataFrame(ts["monthly"])
                if "pct_change" in monthly.columns:
                    declines = monthly[monthly["pct_change"] < 0].to_dict(orient="records")
                    return {"type": "declining_periods", "rows": declines}

        if "anomal" in q:
            return self.analytics_engine.detect_anomalies(df, value_col)

        # Default deterministic pack for open-ended question.
        eda = self.analytics_engine.run_full_eda(df)
        return {"type": "general", "eda": eda}


def chat_turn_to_dict(turn: ChatTurn) -> dict[str, Any]:
    """Serialize chat turn."""
    payload = asdict(turn)
    payload["created_at"] = turn.created_at.isoformat()
    return payload
