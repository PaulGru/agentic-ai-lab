
import json

from openai import OpenAI

from agent.models import (
    LLMResponse,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from providers.base import LLMProvider


def reasoning_for(model: str) -> dict[str, str] | None:
    if model == "gpt-6.2-astra":
        return {"effort": "low"}
    return None


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        native_web_search: bool = False,
        *,
        max_output_tokens: int | None = None,
        max_retries: int | None = None,
        timeout: float | None = None,
    ):
        if max_output_tokens is not None and (type(max_output_tokens) is not int or max_output_tokens < 1):
            raise ValueError("max_output_tokens must be a positive integer")
        options = {}
        if max_retries is not None:
            options["max_retries"] = max_retries
        if timeout is not None:
            options["timeout"] = timeout
        self.client = OpenAI(**options)
        self.model = model
        self.native_web_search = native_web_search
        self.max_output_tokens = max_output_tokens
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

    def _request_options(
        self,
        tools: list[ToolDefinition] | None,
        instructions: str | None,
    ) -> dict:
        request = {"model": self.model}
        if self.max_output_tokens is not None:
            request["max_output_tokens"] = self.max_output_tokens
        if instructions:
            request["instructions"] = instructions
        reasoning = reasoning_for(self.model)
        if reasoning is not None:
            request["reasoning"] = reasoning
        request_tools = self._request_tools(tools)
        if request_tools:
            request["tools"] = request_tools
        return request

    def start(
        self,
        prompt: str,
        tools: list[ToolDefinition] | None = None,
        instructions: str | None = None,
    ) -> LLMResponse:
        response = self.client.responses.create(input=prompt, **self._request_options(tools, instructions))

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

        inputs = [
            {"type": "function_call_output", "call_id": result.call_id, "output": result.output}
            for result in results
        ]
        response = self.client.responses.create(
            previous_response_id=self._previous_response_id,
            input=inputs,
            **self._request_options(tools, instructions),
        )

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
