
import unittest

from langgraph.types import Command

from agent.models import LLMResponse
from agent.runner import AgentRunner
from orchestration.graph_orchestrator import GraphOrchestrator
from orchestration.router import Router
from tests.fakes import FakeProvider
from tools.registry import ToolRegistry


class HumanReviewTests(unittest.TestCase):

    def test_approve_and_reject(self):
        for approved in (True, False):
            with self.subTest(approved=approved):
                router_provider = FakeProvider([
                    LLMResponse(text="math")
                ])

                math_provider = FakeProvider([
                    LLMResponse(text="6000")
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
                    instructions="Answer.",
                    provider=FakeProvider([]),
                    registry=ToolRegistry(),
                )

                router = Router(
                    provider=router_provider,
                    agents=[math_agent, general_agent],
                )

                orchestrator = GraphOrchestrator(
                    router=router,
                )

                config = {
                    "configurable": {
                        "thread_id": f"review-{approved}",
                    }
                }

                paused = orchestrator.graph.invoke(
                    {
                        "prompt": "Calculate 125 * 48",
                        "require_review": True,
                    },
                    config=config,
                )

                self.assertIn("__interrupt__", paused)

                snapshot = orchestrator.graph.get_state(
                    config
                )

                self.assertEqual(
                    snapshot.next,
                    ("human_review",),
                )

                self.assertEqual(
                    len(math_provider.start_calls),
                    1,
                )

                final = orchestrator.graph.invoke(
                    Command(resume=approved),
                    config=config,
                )

                self.assertEqual(
                    final["review_approved"],
                    approved,
                )

                self.assertEqual(
                    final["answer"],
                    "6000" if approved
                    else (
                        "Réponse refusée par "
                        "le validateur humain."
                    ),
                )

                self.assertEqual(
                    len(router_provider.start_calls),
                    1,
                )

                self.assertEqual(
                    len(math_provider.start_calls),
                    1,
                )

                self.assertEqual(
                    orchestrator.graph.get_state(
                        config
                    ).next,
                    (),
                )


if __name__ == "__main__":
    unittest.main()
