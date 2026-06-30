"""External connector stubs with production contracts."""

from __future__ import annotations

from typing import Any

from task_planning_agent.schemas import ConnectorStatus
from task_planning_agent.tools.connector_contract import ExternalConnector


class BaseStubConnector(ExternalConnector):
    name = "base"
    required_env: list[str] = []

    def health_check(self) -> ConnectorStatus:
        return ConnectorStatus(
            name=self.name,
            enabled=False,
            healthy=False,
            capabilities=self.capabilities(),
            message="Running in contract-validated stub mode.",
        )

    def capabilities(self) -> list[str]:
        return ["search", "create", "update"]

    def execute(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "connector": self.name,
            "mode": "stub",
            "action": action,
            "payload": payload,
            "message": "No live credentials configured",
        }

    def dry_run(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "connector": self.name,
            "dry_run": True,
            "action": action,
            "validated": isinstance(payload, dict),
        }

    def credential_requirements(self) -> list[str]:
        return self.required_env


class GitHubIssuesConnector(BaseStubConnector):
    name = "github_issues"
    required_env = ["GITHUB_TOKEN"]


class JiraConnector(BaseStubConnector):
    name = "jira"
    required_env = ["TPA_JIRA_TOKEN"]


class NotionConnector(BaseStubConnector):
    name = "notion"
    required_env = ["TPA_NOTION_TOKEN"]


class TodoistConnector(BaseStubConnector):
    name = "todoist"
    required_env = ["TPA_TODOIST_TOKEN"]


class GoogleTasksConnector(BaseStubConnector):
    name = "google_tasks"
    required_env = ["TPA_GOOGLE_CALENDAR_REFRESH_TOKEN"]


class SlackConnector(BaseStubConnector):
    name = "slack"
    required_env = ["TPA_SLACK_BOT_TOKEN"]


class EmailSummaryConnector(BaseStubConnector):
    name = "email"
    required_env = ["TPA_EMAIL_IMAP_PASSWORD"]


class WhatsAppConnector(BaseStubConnector):
    name = "whatsapp"
    required_env = ["TPA_WHATSAPP_TOKEN"]


def default_connectors() -> list[ExternalConnector]:
    return [
        GitHubIssuesConnector(),
        JiraConnector(),
        NotionConnector(),
        TodoistConnector(),
        GoogleTasksConnector(),
        SlackConnector(),
        EmailSummaryConnector(),
        WhatsAppConnector(),
    ]
