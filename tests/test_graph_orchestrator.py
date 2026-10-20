import unittest

from agent.models import LLMResponse, ToolCall
from agent.runner import AgentRunner
from observability.tracer import Tracer
from orchestration.graph_orchestrator import (
    GraphOrchestrator,
)
from orchestration.orchestrator import Orchestrator
from orchestration.router import Router
from tests.fakes import FakeProvider
from tools.calculator import CalculatorTool
from tools.registry import ToolRegistry


class GraphOrchestratorTests(unittest.TestCase):

    def test_graph_preserves_handoff(self):
        tracer = Tracer(print_events=False)

        router_provider = FakeProvider([
            LLMResponse(text="general")
        ])

        general_provider = FakeProvider([
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="handoff-1",
                        name="handoff_to_agent",
                        arguments={
                            "agent": "math",
                            "task": "Calculate 125 * 48",
                        },
                    )
                ]
            ),
            LLMResponse(
                text="Le coût total est de 6000 €."
            ),
        ])

        math_provider = FakeProvider([
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="calculate-1",
                        name="calculate",
                        arguments={
                            "expression": "125 * 48",
                        },
                    )
                ]
            ),
            LLMResponse(text="6000"),
        ])

        math_registry = ToolRegistry()
        math_registry.register(CalculatorTool())

        math_agent = AgentRunner(
            name="math",
            description="Math specialist",
            instructions="Use the calculator.",
            provider=math_provider,
            registry=math_registry,
            tracer=tracer,
        )

        general_agent = AgentRunner(
            name="general",
            description="General specialist",
            instructions="Delegate calculations.",
            provider=general_provider,
            registry=ToolRegistry(),
            handoff_targets={
                "math": "Math specialist",
            },
            tracer=tracer,
        )

        router = Router(
            provider=router_provider,
            agents=[math_agent, general_agent],
            tracer=tracer,
        )

        orchestrator = Orchestrator(
            router=router,
            tracer=tracer,
        )

        graph_orchestrator = GraphOrchestrator(
            router=router,
            orchestrator=orchestrator,
            tracer=tracer,
        )

        answer = graph_orchestrator.run(
            "Explain the cost of 125 units at 48 €."
        )

        self.assertEqual(
            answer,
            "Le coût total est de 6000 €.",
        )

        self.assertEqual(
            len(router_provider.start_calls),
            1,
        )

        self.assertEqual(
            len(general_provider.continue_calls),
            1,
        )

        event_types = [
            event.type
            for event in tracer.events
        ]

        self.assertIn("route_selected", event_types)
        self.assertIn("handoff_started", event_types)
        self.assertIn("handoff_returned", event_types)
        self.assertIn("run_finished", event_types)


if __name__ == "__main__":
    unittest.main()