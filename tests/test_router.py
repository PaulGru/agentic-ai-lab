import unittest

from agent.models import LLMResponse
from agent.runner import AgentRunner
from orchestration.router import Router
from tools.registry import ToolRegistry

from tests.fakes import FakeProvider


def make_agent(name: str) -> AgentRunner:
    return AgentRunner(
        name=name,
        description=f"{name} agent",
        instructions=f"You are the {name} agent.",
        provider=FakeProvider(
            [LLMResponse(text="unused")]
        ),
        registry=ToolRegistry(),
    )


class RouterTests(unittest.TestCase):
    def test_router_selects_math_agent(self):
        math_agent = make_agent("math")
        general_agent = make_agent("general")

        router_provider = FakeProvider(
            [
                LLMResponse(text="math")
            ]
        )

        router = Router(
            provider=router_provider,
            agents=[
                math_agent,
                general_agent,
            ],
        )

        selected_agent = router.route(
            "Calculate 125 * 48"
        )

        self.assertEqual(
            selected_agent.name,
            "math",
        )

    def test_router_rejects_unknown_agent(self):
        math_agent = make_agent("math")

        router_provider = FakeProvider(
            [
                LLMResponse(text="unknown")
            ]
        )

        router = Router(
            provider=router_provider,
            agents=[math_agent],
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Router selected unknown agent",
        ):
            router.route("Hello")


if __name__ == "__main__":
    unittest.main()