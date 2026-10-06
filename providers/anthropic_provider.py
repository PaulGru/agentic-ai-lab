from anthropic import Anthropic

from agent.models import (
    LLMResponse,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from providers.base import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        max_tokens: int = 1024,
    ):
        self.client = Anthropic()
        self.model = model
        self.max_tokens = max_tokens

        self._messages = []
        self._last_assistant_content = None

    def start(
        self,
        prompt: str,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        self._messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        return self._request(tools)

    def continue_with_tool_results(
        self,
        results: list[ToolResult],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        if self._last_assistant_content is None:
            raise RuntimeError(
                "No assistant response to continue"
            )

        self._messages.append(
            {
                "role": "assistant",
                "content": self._last_assistant_content,
            }
        )

        self._messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": result.call_id,
                        "content": result.output,
                        "is_error": result.is_error,
                    }
                    for result in results
                ],
            }
        )

        return self._request(tools)

    def _request(
        self,
        tools: list[ToolDefinition] | None,
    ) -> LLMResponse:
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._messages,
        }

        if tools:
            request["tools"] = [
                self._convert_tool(tool)
                for tool in tools
            ]

        response = self.client.messages.create(**request)

        self._last_assistant_content = [
            block.model_dump()
            for block in response.content
        ]

        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        return LLMResponse(
            text="\n".join(text_parts) or None,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _convert_tool(tool: ToolDefinition) -> dict:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }