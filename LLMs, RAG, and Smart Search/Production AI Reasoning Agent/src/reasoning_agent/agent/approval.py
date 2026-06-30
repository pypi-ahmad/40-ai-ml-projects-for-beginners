"""Human approval checkpoint policy (phase-2 optional)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ApprovalRequest:
    """Approval request payload."""

    required: bool
    reason: str


class ApprovalPolicy:
    """Determine whether user approval is required for action."""

    def evaluate(self, tool_name: str, args: dict[str, object]) -> ApprovalRequest:
        """Return approval requirement for risky actions."""

        risky_tools = {"python_repl", "file_reader", "webpage_reader"}
        if tool_name in risky_tools and bool(args):
            return ApprovalRequest(required=True, reason=f"Tool {tool_name} flagged for human approval")
        return ApprovalRequest(required=False, reason="not_required")
