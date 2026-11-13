"""Deterministic, restartable provider used by legacy workflow demos/tests."""

from dataclasses import asdict

from agent.models import LLMResponse, ToolCall
from providers.base import LLMProvider


class FakeProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.start_calls = []
        self.continue_calls = []

    def start(self, prompt, tools=None, instructions=None) -> LLMResponse:
        self.start_calls.append({
            "prompt": prompt,
            "tools": tools,
            "instructions": instructions,
        })
        return self._next_response()

    def continue_with_tool_results(self, results, tools=None, instructions=None) -> LLMResponse:
        self.continue_calls.append({
            "results": results,
            "tools": tools,
            "instructions": instructions,
        })
        return self._next_response()

    def _next_response(self) -> LLMResponse:
        if not self.responses:
            raise RuntimeError("FakeProvider has no response left")
        return self.responses.pop(0)


class StatefulFakeProvider(FakeProvider):
    def export_state(self) -> dict:
        return {"remaining": [asdict(r) for r in self.responses]}

    def restore_state(self, state: dict) -> None:
        self.responses = [
            LLMResponse(
                text=item["text"],
                tool_calls=[ToolCall(**call) for call in item["tool_calls"]],
                web_search_used=item.get("web_search_used", False),
                sources=item.get("sources", []),
            )
            for item in state["remaining"]
        ]
