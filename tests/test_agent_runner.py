import unittest

from agent.models import (
    LLMResponse,
    ToolCall,
)
from agent.runner import AgentRunner
from tools.calculator import CalculatorTool
from tools.registry import ToolRegistry

from tests.fakes import FakeProvider


class AgentRunnerTests(unittest.TestCase):
    def test_agent_returns_final_text(self):
        provider = FakeProvider(
            [
                LLMResponse(
                    text="Hello!"
                )
            ]
        )

        agent = AgentRunner(
            name="general",
            description="General assistant",
            instructions="Be helpful.",
            provider=provider,
            registry=ToolRegistry(),
        )

        outcome = agent.run("Hello")

        self.assertEqual(
            outcome.text,
            "Hello!",
        )

        self.assertIsNone(
            outcome.handoff
        )

    def test_agent_executes_tool_and_continues(self):
        provider = FakeProvider(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="calculate",
                            arguments={
                                "expression": "125 * 48"
                            },
                        )
                    ]
                ),
                LLMResponse(
                    text="6000"
                ),
            ]
        )

        registry = ToolRegistry()
        registry.register(
            CalculatorTool()
        )

        agent = AgentRunner(
            name="math",
            description="Math specialist",
            instructions="Use the calculator.",
            provider=provider,
            registry=registry,
        )

        outcome = agent.run(
            "Calculate 125 * 48"
        )

        self.assertEqual(
            outcome.text,
            "6000",
        )

        self.assertEqual(
            len(provider.continue_calls),
            1,
        )

        tool_result = (
            provider
            .continue_calls[0]["results"][0]
        )

        self.assertEqual(
            tool_result.output,
            "6000",
        )

        self.assertFalse(
            tool_result.is_error
        )

    def test_agent_returns_handoff_request(self):
        provider = FakeProvider(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="handoff-1",
                            name="handoff_to_agent",
                            arguments={
                                "agent": "math",
                                "task": (
                                    "Calculate 125 * 48"
                                ),
                            },
                        )
                    ]
                )
            ]
        )

        agent = AgentRunner(
            name="general",
            description="General assistant",
            instructions="Delegate calculations.",
            provider=provider,
            registry=ToolRegistry(),
            handoff_targets={
                "math": "Math specialist"
            },
        )

        outcome = agent.run(
            "Explain this calculation"
        )

        self.assertIsNotNone(
            outcome.handoff
        )

        self.assertEqual(
            outcome.handoff.target_agent,
            "math",
        )

        self.assertEqual(
            outcome.handoff.task,
            "Calculate 125 * 48",
        )


if __name__ == "__main__":
    unittest.main()