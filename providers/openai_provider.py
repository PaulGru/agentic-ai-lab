import json

from openai import OpenAI

from agent.models import (
    LLMResponse,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str):
        self.client = OpenAI()
        self.model = model
        self._previous_response_id: str | None = None

    def start(
        self,
        prompt: str,
        tools: list[ToolDefinition] | None = None,
        instructions: str | None = None,
    ) -> LLMResponse:
        request = {
            "model": self.model,
            "input": prompt,
        }

        if instructions:
            request["instructions"] = instructions

        if tools:
            request["tools"] = [
                self._convert_tool(tool)
                for tool in tools
            ]

        response = self.client.responses.create(**request)

        self._previous_response_id = response.id

        return self._convert_response(response)

    def continue_with_tool_results(
        self,
        results: list[ToolResult],
        tools: list[ToolDefinition] | None = None,
        instructions: str | None = None,
    ) -> LLMResponse:
        if self._previous_response_id is None:
            raise RuntimeError(
                "No previous response to continue"
            )

        request = {
            "model": self.model,
            "previous_response_id": self._previous_response_id,
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": result.call_id,
                    "output": result.output,
                }
                for result in results
            ],
        }

        if tools:
            request["tools"] = [
                self._convert_tool(tool)
                for tool in tools
            ]

        if instructions:
            request["instructions"] = instructions

        response = self.client.responses.create(**request)

        self._previous_response_id = response.id

        return self._convert_response(response)

    def _convert_response(self, response) -> LLMResponse:
        tool_calls = []

        for item in response.output:
            if item.type == "function_call":
                tool_calls.append(
                    ToolCall(
                        id=item.call_id,
                        name=item.name,
                        arguments=json.loads(item.arguments),
                    )
                )

        return LLMResponse(
            text=response.output_text or None,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _convert_tool(tool: ToolDefinition) -> dict:
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }