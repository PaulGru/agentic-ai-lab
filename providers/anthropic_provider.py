from anthropic import Anthropic

import copy

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
        native_web_search: bool = False,
    ):
        self.client = Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.native_web_search = native_web_search

        self._messages = []
        self._last_assistant_content = None

    def start(
        self,
        prompt: str,
        tools: list[ToolDefinition] | None = None,
        instructions: str | None = None,
    ) -> LLMResponse:
        self._messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        self._last_assistant_content = None

        return self._request(
            tools=tools,
            instructions=instructions,
        )

    def continue_with_tool_results(
        self,
        results: list[ToolResult],
        tools: list[ToolDefinition] | None = None,
        instructions: str | None = None,
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

        return self._request(
            tools=tools,
            instructions=instructions,
        )
        
    
    def _request(
        self,
        tools: list[ToolDefinition] | None,
        instructions: str | None = None,
    ) -> LLMResponse:
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._messages,
        }

        if instructions:
            request["system"] = instructions

        request_tools = [
            self._convert_tool(tool)
            for tool in (tools or [])
        ]

        if self.native_web_search:
            request_tools.append({
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 2,
            })

        if request_tools:
            request["tools"] = request_tools

        response = self.client.messages.create(**request)

        self._last_assistant_content = [
            block.model_dump()
            for block in response.content
        ]

        text_parts = []
        tool_calls = []
        sources = []
        seen_urls = set()
        web_search_used = False

        for block in response.content:
            if block.type == "server_tool_use":
                if block.name == "web_search":
                    web_search_used = True

            elif block.type == "web_search_tool_result":
                web_search_used = True

            elif block.type == "text":
                text_parts.append(block.text)

                citations = (
                    getattr(block, "citations", None) or []
                )

                for citation in citations:
                    if citation.type != "web_search_result_location":
                        continue

                    url = citation.url

                    if (
                        isinstance(url, str)
                        and url.startswith(("https://", "http://"))
                        and url not in seen_urls
                    ):
                        sources.append({
                            "title": citation.title or url,
                            "url": url,
                        })
                        seen_urls.add(url)

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
            web_search_used=web_search_used,
            sources=sources,
        )

    @staticmethod
    def _convert_tool(tool: ToolDefinition) -> dict:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
   
    def export_state(self) -> dict:
        return {
            "messages": copy.deepcopy(
                self._messages
            ),
            "last_assistant_content": copy.deepcopy(
                self._last_assistant_content
            ),
        }

    def restore_state(self, state: dict) -> None:
        self._messages = copy.deepcopy(
            state["messages"]
        )
        self._last_assistant_content = copy.deepcopy(
            state["last_assistant_content"]
        )
