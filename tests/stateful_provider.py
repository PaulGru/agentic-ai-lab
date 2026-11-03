"""Restartable fake used by V4.5 demonstrations and unit tests."""
from dataclasses import asdict

from agent.models import LLMResponse, ToolCall
from tests.fakes import FakeProvider


class StatefulFakeProvider(FakeProvider):
    def export_state(self) -> dict:
        return {"remaining": [asdict(response) for response in self.responses]}

    def restore_state(self, state: dict) -> None:
        self.responses = [
            LLMResponse(
                text=item["text"],
                tool_calls=[ToolCall(**call) for call in item["tool_calls"]],
            )
            for item in state["remaining"]
        ]
