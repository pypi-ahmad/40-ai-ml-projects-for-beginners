"""Connector contract for optional external integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from task_planning_agent.schemas import ConnectorStatus


class ExternalConnector(ABC):
    """Standardized connector interface for SaaS integrations."""

    name: str

    @abstractmethod
    def health_check(self) -> ConnectorStatus:
        """Return connector health and capabilities."""

    @abstractmethod
    def capabilities(self) -> list[str]:
        """List supported operations."""

    @abstractmethod
    def execute(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute connector action."""

    @abstractmethod
    def dry_run(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate contract behavior without external calls."""

    @abstractmethod
    def credential_requirements(self) -> list[str]:
        """Required env vars/secrets for live mode."""
