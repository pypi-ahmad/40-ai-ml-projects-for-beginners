"""Human-in-the-loop control service."""

from __future__ import annotations

from dataclasses import dataclass, field

from langgraph_platform.state.models import HITLAction


@dataclass(slots=True)
class HITLState:
    """HITL status for workflow."""

    approved: bool = False
    paused: bool = False
    rejected: bool = False
    overrides: list[str] = field(default_factory=list)


class HITLService:
    """Store and process HITL actions."""

    def __init__(self) -> None:
        self._states: dict[str, HITLState] = {}

    def _state(self, workflow_id: str) -> HITLState:
        if workflow_id not in self._states:
            self._states[workflow_id] = HITLState()
        return self._states[workflow_id]

    def apply(self, workflow_id: str, action: HITLAction, note: str | None = None) -> HITLState:
        """Apply human action to workflow status."""

        state = self._state(workflow_id)
        if action == HITLAction.APPROVE:
            state.approved = True
            state.rejected = False
        elif action == HITLAction.REJECT:
            state.rejected = True
            state.approved = False
        elif action == HITLAction.PAUSE:
            state.paused = True
        elif action == HITLAction.RESUME:
            state.paused = False
        elif action == HITLAction.OVERRIDE and note:
            state.overrides.append(note)
        elif action == HITLAction.RERUN and note:
            state.overrides.append(f"rerun:{note}")
        return state

    def get(self, workflow_id: str) -> HITLState:
        """Get HITL state for workflow."""

        return self._state(workflow_id)
