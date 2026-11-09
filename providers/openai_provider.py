
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
    def __init__(
        self,
        model: str,
        native_web_search: bool = False,
    ):
        self.client = OpenAI()
        self.model = model
        self.native_web_search = native_web_search
        self._previous_response_id: str | None = None

    def _request_tools(
        self,
        tools: list[ToolDefinition] | None,
    ) -> list[dict]:
        result = [
            self._convert_tool(tool)
            for tool in (tools or [])
        ]

        if self.native_web_search:
            result.append({
                "type": "web_search",
                "search_context_size": "low",
            })

        return result

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

        request_tools = self._request_tools(tools)

        if request_tools:
            request["tools"] = request_tools

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

        if instructions:
            request["instructions"] = instructions

        request_tools = self._request_tools(tools)

        if request_tools:
            request["tools"] = request_tools

        response = self.client.responses.create(**request)

        self._previous_response_id = response.id

        return self._convert_response(response)

    def _convert_response(self, response) -> LLMResponse:
        tool_calls = []
        sources = []
        seen_urls = set()
        web_search_used = False

        for item in response.output:
            if item.type == "web_search_call":
                web_search_used = True

            elif item.type == "function_call":
                tool_calls.append(
                    ToolCall(
                        id=item.call_id,
                        name=item.name,
                        arguments=json.loads(item.arguments),
                    )
                )

            elif item.type == "message":
                for part in item.content:
                    annotations = (
                        getattr(part, "annotations", None) or []
                    )

                    for annotation in annotations:
                        if annotation.type != "url_citation":
                            continue

                        url = annotation.url

                        if (
                            isinstance(url, str)
                            and url.startswith(("https://", "http://"))
                            and url not in seen_urls
                        ):
                            sources.append({
                                "title": annotation.title or url,
                                "url": url,
                            })
                            seen_urls.add(url)

        return LLMResponse(
            text=response.output_text or None,
            tool_calls=tool_calls,
            web_search_used=web_search_used,
            sources=sources,
        )

    @staticmethod
    def _convert_tool(tool: ToolDefinition) -> dict:
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }

    def export_state(self) -> dict:
        return {
            "previous_response_id": self._previous_response_id,
        }

    def restore_state(self, state: dict) -> None:
        self._previous_response_id = state["previous_response_id"]
