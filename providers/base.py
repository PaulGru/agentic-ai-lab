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