from agent.models import LLMResponse
from providers.base import LLMProvider


class FakeProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.start_calls = []
        self.continue_calls = []

    def start(
        self,
        prompt: str,
        tools=None,
        instructions: str | None = None,
    ) -> LLMResponse:
        self.start_calls.append(
            {
                "prompt": prompt,
                "tools": tools,
                "instructions": instructions,
            }
        )

        return self._next_response()

    def continue_with_tool_results(
        self,
        results,
        tools=None,
        instructions: str | None = None,
    ) -> LLMResponse:
        self.continue_calls.append(
            {
                "results": results,
                "tools": tools,
                "instructions": instructions,
            }
        )

        return self._next_response()

    def _next_response(self) -> LLMResponse:
        if not self.responses:
            raise RuntimeError(
                "FakeProvider has no response left"
            )

        return self.responses.pop(0)