from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    output: str
    is_error: bool = False


@dataclass
class LLMResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class HandoffRequest:
    call_id: str
    target_agent: str
    task: str


@dataclass(frozen=True)
class AgentOutcome:
    text: str | None = None
    handoff: HandoffRequest | None = None

    def __post_init__(self):
        has_text = self.text is not None
        has_handoff = self.handoff is not None

        if has_text == has_handoff:
            raise ValueError(
                "AgentOutcome must contain exactly one of "
                "'text' or 'handoff'"
            )