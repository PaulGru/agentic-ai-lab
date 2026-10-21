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

        graph_orchestrator = GraphOrchestrator(
            router=router,
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

        entered_nodes = [
            event.data["node"]
            for event in tracer.events
            if event.type == "graph_node_entered"
        ]

        self.assertEqual(
            entered_nodes,
            ["general", "math", "general"],
        )
    
    
    def test_graph_routes_directly_to_math(self):
        tracer = Tracer(print_events=False)

        # Router selects Math
        router_provider = FakeProvider([
            LLMResponse(text="math")
        ])

        # Math agent uses the calculator
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

        # General must not be called
        general_provider = FakeProvider([])

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
            instructions="General assistance.",
            provider=general_provider,
            registry=ToolRegistry(),
            tracer=tracer,
        )

        router = Router(
            provider=router_provider,
            agents=[math_agent, general_agent],
            tracer=tracer,
        )

        graph_orchestrator = GraphOrchestrator(
            router=router,
            tracer=tracer,
        )

        answer = graph_orchestrator.run(
            "Calculate 125 * 48"
        )

        self.assertEqual(answer, "6000")

        # Routing decision made exactly once
        self.assertEqual(
            len(router_provider.start_calls),
            1,
        )

        # Math executed, General skipped
        self.assertEqual(
            len(math_provider.start_calls),
            1,
        )

        self.assertEqual(
            len(general_provider.start_calls),
            0,
        )

        # Verify the actual graph path
        entered_nodes = [
            event.data["node"]
            for event in tracer.events
            if event.type == "graph_node_entered"
        ]

        self.assertEqual(
            entered_nodes,
            ["math"],
        )
    
    
    def test_graph_rejects_handoff_cycle(self):
        tracer = Tracer(print_events=False)

        general_provider = FakeProvider([
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="general-to-math",
                        name="handoff_to_agent",
                        arguments={
                            "agent": "math",
                            "task": "Ask Math",
                        },
                    )
                ]
            )
        ])

        math_provider = FakeProvider([
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="math-to-general",
                        name="handoff_to_agent",
                        arguments={
                            "agent": "general",
                            "task": "Ask General",
                        },
                    )
                ]
            )
        ])

        general_agent = AgentRunner(
            name="general",
            description="General agent",
            instructions="Delegate to Math.",
            provider=general_provider,
            registry=ToolRegistry(),
            handoff_targets={
                "math": "Math specialist",
            },
            tracer=tracer,
        )

        math_agent = AgentRunner(
            name="math",
            description="Math agent",
            instructions="Delegate to General.",
            provider=math_provider,
            registry=ToolRegistry(),
            handoff_targets={
                "general": "General specialist",
            },
            tracer=tracer,
        )

        router = Router(
            provider=FakeProvider([
                LLMResponse(text="general")
            ]),
            agents=[general_agent, math_agent],
            tracer=tracer,
        )

        graph_orchestrator = GraphOrchestrator(
            router=router,
            tracer=tracer,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Handoff cycle detected",
        ):
            graph_orchestrator.run("Start")

        event_types = [
            event.type
            for event in tracer.events
        ]

        self.assertIn("run_failed", event_types)
    
    
    def test_graph_checkpoint_and_resume(self):
        router_provider = FakeProvider([
            LLMResponse(text="math")
        ])

        math_provider = FakeProvider([
            LLMResponse(text="6000")
        ])

        math_agent = AgentRunner(
            name="math",
            description="Math agent",
            instructions="Calculate.",
            provider=math_provider,
            registry=ToolRegistry(),
        )

        general_agent = AgentRunner(
            name="general",
            description="General agent",
            instructions="Answer.",
            provider=FakeProvider([]),
            registry=ToolRegistry(),
        )

        router = Router(
            provider=router_provider,
            agents=[math_agent, general_agent],
        )

        graph_orchestrator = GraphOrchestrator(
            router=router,
        )

        config = {
            "configurable": {
                "thread_id": "test-checkpoint",
            }
        }

        # Stop before the Math node
        graph_orchestrator.graph.invoke(
            {"prompt": "Calculate 125 * 48"},
            config=config,
            interrupt_before=["math"],
        )

        snapshot = graph_orchestrator.graph.get_state(
            config
        )

        self.assertEqual(snapshot.next, ("math",))
        self.assertEqual(
            len(math_provider.start_calls),
            0,
        )

        # Resume the saved execution
        result = graph_orchestrator.graph.invoke(
            None,
            config=config,
        )

        self.assertEqual(result["answer"], "6000")
        self.assertEqual(
            len(router_provider.start_calls),
            1,
        )
        self.assertEqual(
            len(math_provider.start_calls),
            1,
        )

        final_snapshot = (
            graph_orchestrator.graph.get_state(config)
        )

        self.assertEqual(final_snapshot.next, ())
        self.assertTrue(
            final_snapshot.values["finished"]
        )
    
    
    def test_graph_keeps_threads_isolated(self):
        router_provider = FakeProvider([
            LLMResponse(text="math"),
            LLMResponse(text="general"),
        ])

        math_provider = FakeProvider([
            LLMResponse(text="6000"),
        ])

        general_provider = FakeProvider([
            LLMResponse(text="Bonjour !"),
        ])

        math_agent = AgentRunner(
            name="math",
            description="Math specialist",
            instructions="Calculate.",
            provider=math_provider,
            registry=ToolRegistry(),
        )

        general_agent = AgentRunner(
            name="general",
            description="General specialist",
            instructions="Answer questions.",
            provider=general_provider,
            registry=ToolRegistry(),
        )

        router = Router(
            provider=router_provider,
            agents=[math_agent, general_agent],
        )

        orchestrator = GraphOrchestrator(
            router=router,
        )

        math_answer = orchestrator.run(
            "Calculate 125 * 48",
            thread_id="thread-math",
        )

        general_answer = orchestrator.run(
            "Say hello",
            thread_id="thread-general",
        )

        self.assertEqual(math_answer, "6000")
        self.assertEqual(
            general_answer,
            "Bonjour !",
        )

        math_state = orchestrator.graph.get_state({
            "configurable": {
                "thread_id": "thread-math",
            }
        })

        general_state = orchestrator.graph.get_state({
            "configurable": {
                "thread_id": "thread-general",
            }
        })

        self.assertEqual(
            math_state.values["selected_agent"],
            "math",
        )

        self.assertEqual(
            general_state.values["selected_agent"],
            "general",
        )

        self.assertEqual(
            math_state.values["answer"],
            "6000",
        )

        self.assertEqual(
            general_state.values["answer"],
            "Bonjour !",
        )

        self.assertEqual(math_state.next, ())
        self.assertEqual(general_state.next, ())

    
    def test_checkpoint_preserves_pending_handoff(self):
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
            LLMResponse(text="6000"),
        ])

        math_agent = AgentRunner(
            name="math",
            description="Math specialist",
            instructions="Calculate.",
            provider=math_provider,
            registry=ToolRegistry(),
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
        )

        router_provider = FakeProvider([
            LLMResponse(text="general"),
        ])

        router = Router(
            provider=router_provider,
            agents=[math_agent, general_agent],
        )

        orchestrator = GraphOrchestrator(
            router=router,
        )

        config = {
            "configurable": {
                "thread_id": "pending-handoff",
            },
        }

        # Execute General and pause before Math.
        orchestrator.graph.invoke(
            {"prompt": "Explain the total cost."},
            config=config,
            interrupt_before=["math"],
        )

        snapshot = orchestrator.graph.get_state(
            config
        )

        self.assertEqual(snapshot.next, ("math",))
        self.assertEqual(
            snapshot.values["current_agent"],
            "math",
        )

        stack = snapshot.values["handoff_stack"]

        self.assertEqual(len(stack), 1)
        self.assertEqual(
            stack[0]["source_agent"],
            "general",
        )
        self.assertEqual(
            stack[0]["target_agent"],
            "math",
        )
        self.assertEqual(
            stack[0]["call_id"],
            "handoff-1",
        )

        self.assertEqual(
            len(math_provider.start_calls),
            0,
        )

        # Resume in the same process.
        final_state = orchestrator.graph.invoke(
            None,
            config=config,
        )

        self.assertEqual(
            final_state["answer"],
            "Le coût total est de 6000 €.",
        )

        self.assertEqual(
            final_state["handoff_stack"],
            [],
        )

        self.assertEqual(
            len(general_provider.continue_calls),
            1,
        )

        returned_result = (
            general_provider.continue_calls[0]
            ["results"][0]
        )

        self.assertEqual(
            returned_result.output,
            "6000",
        )

        self.assertEqual(
            len(router_provider.start_calls),
            1,
        )

        self.assertEqual(
            orchestrator.graph.get_state(config).next,
            (),
        )


if __name__ == "__main__":
    unittest.main()