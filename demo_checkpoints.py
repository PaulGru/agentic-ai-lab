
import sys

from langgraph.checkpoint.sqlite import SqliteSaver

from agent.models import LLMResponse
from agent.runner import AgentRunner
from orchestration.graph_orchestrator import GraphOrchestrator
from orchestration.router import Router
from tests.fakes import FakeProvider
from tools.registry import ToolRegistry


THREAD_ID = "v34-demo-math"


def build_demo(checkpointer, mode):
    # Router has a response only for the first launch.
    # During resume, calling it again would fail.
    router_provider = FakeProvider(
        [LLMResponse(text="math")]
        if mode == "pause"
        else []
    )

    math_provider = FakeProvider([
        LLMResponse(text="6000")
    ])

    math_agent = AgentRunner(
        name="math",
        description="Math specialist",
        instructions="Calculate exactly.",
        provider=math_provider,
        registry=ToolRegistry(),
    )

    general_agent = AgentRunner(
        name="general",
        description="General specialist",
        instructions="Answer general questions.",
        provider=FakeProvider([]),
        registry=ToolRegistry(),
    )

    router = Router(
        provider=router_provider,
        agents=[math_agent, general_agent],
    )

    orchestrator = GraphOrchestrator(
        router=router,
        checkpointer=checkpointer,
    )

    return orchestrator, router_provider, math_provider


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python demo_checkpoints.py pause|resume"
        )

    mode = sys.argv[1]

    if mode not in ("pause", "resume"):
        raise ValueError("Unknown mode")

    config = {
        "configurable": {
            "thread_id": THREAD_ID,
        },
        "recursion_limit": 30,
    }

    with SqliteSaver.from_conn_string(
        "checkpoints_demo.sqlite"
    ) as checkpointer:

        orchestrator, router_provider, math_provider = (
            build_demo(checkpointer, mode)
        )

        if mode == "pause":
            orchestrator.graph.invoke(
                {"prompt": "Calculate 125 * 48"},
                config=config,
                interrupt_before=["math"],
            )

            snapshot = orchestrator.graph.get_state(
                config
            )

            print("Next nodes:", snapshot.next)
            print("Saved state:", snapshot.values)
            print(
                "Router calls:",
                len(router_provider.start_calls),
            )
            print(
                "Math calls:",
                len(math_provider.start_calls),
            )

        else:
            snapshot = orchestrator.graph.get_state(
                config
            )

            if snapshot.next != ("math",):
                raise RuntimeError(
                    "No pending Math checkpoint"
                )

            result = orchestrator.graph.invoke(
                None,
                config=config,
            )

            print("Answer:", result["answer"])
            print(
                "Router calls:",
                len(router_provider.start_calls),
            )
            print(
                "Math calls:",
                len(math_provider.start_calls),
            )
            print(
                "Next nodes:",
                orchestrator.graph.get_state(config).next,
            )


if __name__ == "__main__":
    main()
