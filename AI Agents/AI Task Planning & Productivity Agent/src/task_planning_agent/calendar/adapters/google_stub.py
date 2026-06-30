"""Google Calendar adapter contract stub."""

from __future__ import annotations

from task_planning_agent.schemas import ConnectorStatus


class GoogleCalendarAdapterStub:
    """Credential-gated stub until OAuth setup is provided."""

    def status(self) -> ConnectorStatus:
        return ConnectorStatus(
            name="google_calendar",
            enabled=False,
            healthy=False,
            capabilities=["list_events", "create_event", "find_conflicts"],
            message="Credentials not configured; running in contract-stub mode.",
        )

    def list_events(self, *_: object, **__: object) -> list[dict[str, str]]:
        return []
