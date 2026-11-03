from abc import ABC, abstractmethod

from agent.models import (
    LLMResponse,
    ToolDefinition,
    ToolResult,
)


class LLMProvider(ABC):
    @abstractmethod
    def start(
        self,
        prompt: str,
        tools: list[ToolDefinition] | None = None,
        instructions: str | None = None,
    ) -> LLMResponse:
        pass

    @abstractmethod
    def continue_with_tool_results(
        self,
        results: list[ToolResult],
        tools: list[ToolDefinition] | None = None,
        instructions: str | None = None,
    ) -> LLMResponse:
        pass
    
    def export_state(self) -> dict:
        raise NotImplementedError(
            "Provider state export is not supported"
        )

    def restore_state(self, state: dict) -> None:
        raise NotImplementedError(
            "Provider state restoration is not supported"
        )
