"""Recommendation engine for productivity improvements."""

from __future__ import annotations

from task_planning_agent.schemas import Recommendation, ScheduleBlock


class RecommendationAgent:
    """Generate optimization recommendations from plan artifacts."""

    def suggest(self, blocks: list[ScheduleBlock]) -> list[Recommendation]:
        if not blocks:
            return [
                Recommendation(
                    category="planning",
                    suggestion="Add at least one concrete task with deadline and estimate.",
                    impact="high",
                )
            ]

        recs: list[Recommendation] = [
            Recommendation(
                category="batching",
                suggestion="Batch admin tasks into one 45-minute block.",
                impact="medium",
            ),
            Recommendation(
                category="deep_work",
                suggestion="Reserve 90-minute focus window in morning for high-priority items.",
                impact="high",
            ),
            Recommendation(
                category="automation",
                suggestion="Automate recurring reminders for weekly report preparation.",
                impact="medium",
            ),
        ]
        high_risk = [block for block in blocks if block.risk_level.value == "high"]
        if high_risk:
            recs.append(
                Recommendation(
                    category="risk",
                    suggestion="Split risky tasks into smaller chunks and schedule earlier.",
                    impact="high",
                )
            )
        return recs
