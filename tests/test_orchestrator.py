import unittest

from agent.models import (
    LLMResponse,
    ToolCall,
)
from agent.runner import AgentRunner
from orchestration.orchestrator import Orchestrator
from orchestration.router import Router
from tools.calculator import CalculatorTool
from tools.registry import ToolRegistry

from tests.fakes import FakeProvider


class OrchestratorTests(unittest.TestCase):
    def test_complete_handoff_flow(self):
        # -------------------------
        # Router
        # -------------------------

        router_provider = FakeProvider(
            [
                LLMResponse(
                    text="general"
                )
            ]
        )

        # -------------------------
        # General agent
        # -------------------------

        general_provider = FakeProvider(
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
                ),
                LLMResponse(
                    text=(
                        "Le coût total est "
                        "de 6000 €."
                    )
                ),
            ]
        )

        general_agent = AgentRunner(
            name="general",
            description=(
                "Handles general and mixed requests."
            ),
            instructions=(
                "Delegate exact calculations."
            ),
            provider=general_provider,
            registry=ToolRegistry(),
            handoff_targets={
                "math": "Math specialist"
            },
        )

        # -------------------------
        # Math agent
        # -------------------------

        math_provider = FakeProvider(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="calculate-1",
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

        math_registry = ToolRegistry()
        math_registry.register(
            CalculatorTool()
        )

        math_agent = AgentRunner(
            name="math",
            description="Math specialist",
            instructions="Use the calculator.",
            provider=math_provider,
            registry=math_registry,
        )

        # -------------------------
        # Orchestration
        # -------------------------

        router = Router(
            provider=router_provider,
            agents=[
                math_agent,
                general_agent,
            ],
        )

        orchestrator = Orchestrator(
            router=router,
        )

        answer = orchestrator.run(
            "Explain the total cost of "
            "125 units at 48 € each."
        )

        # -------------------------
        # Assertions
        # -------------------------

        self.assertEqual(
            answer,
            "Le coût total est de 6000 €.",
        )

        self.assertEqual(
            len(
                general_provider.continue_calls
            ),
            1,
        )

        returned_result = (
            general_provider
            .continue_calls[0]["results"][0]
        )

        self.assertEqual(
            returned_result.output,
            "6000",
        )
    def test_handoff_cycle_is_rejected(self):
        agent_a_provider = FakeProvider(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="a-to-b",
                            name="handoff_to_agent",
                            arguments={
                                "agent": "b",
                                "task": "Ask agent B",
                            },
                        )
                    ]
                )
            ]
        )

        agent_b_provider = FakeProvider(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="b-to-a",
                            name="handoff_to_agent",
                            arguments={
                                "agent": "a",
                                "task": "Ask agent A",
                            },
                        )
                    ]
                )
            ]
        )

        agent_a = AgentRunner(
            name="a",
            description="Agent A",
            instructions="Delegate to B.",
            provider=agent_a_provider,
            registry=ToolRegistry(),
            handoff_targets={
                "b": "Agent B"
            },
        )

        agent_b = AgentRunner(
            name="b",
            description="Agent B",
            instructions="Delegate to A.",
            provider=agent_b_provider,
            registry=ToolRegistry(),
            handoff_targets={
                "a": "Agent A"
            },
        )

        router = Router(
            provider=FakeProvider(
                [
                    LLMResponse(text="a")
                ]
            ),
            agents=[
                agent_a,
                agent_b,
            ],
        )

        orchestrator = Orchestrator(
            router=router,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Handoff cycle detected",
        ):
            orchestrator.run("Start")
            

if __name__ == "__main__":
    unittest.main()