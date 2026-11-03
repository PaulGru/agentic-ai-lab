from typing import Literal

ApprovalMode = Literal["auto", "review", "deny"]


class ApprovalPolicy:
    """Server-side allowlist. Unconfigured known tools require review."""

    def __init__(self, modes: dict[str, ApprovalMode] | None = None):
        self.modes = dict(modes or {})
        if any(
            mode not in ("auto", "review", "deny")
            for mode in self.modes.values()
        ):
            raise ValueError("Unknown approval mode")

    def mode(self, tool_name: str, registered: set[str]) -> ApprovalMode:
        if tool_name not in registered:
            return "deny"
        return self.modes.get(tool_name, "review")
