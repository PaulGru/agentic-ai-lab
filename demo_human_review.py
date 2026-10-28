
import sys

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent.models import LLMResponse
from agent.runner import AgentRunner
from orchestration.graph_orchestrator import GraphOrchestrator
from orchestration.router import Router
from tests.fakes import FakeProvider
from tools.registry import ToolRegistry


THREAD_ID = "v41-human-review"


def build_demo(checkpointer, mode):
    initial = mode == "pause"

    router_provider = FakeProvider(
        [LLMResponse(text="math")] if initial else []
    )

    math_provider = FakeProvider(
        [LLMResponse(text="6000")] if initial else []
    )

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

    return GraphOrchestrator(
        router=router,
        checkpointer=checkpointer,
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python demo_human_review.py "
            "pause|approve|reject"
        )

    mode = sys.argv[1]

    if mode not in ("pause", "approve", "reject"):
        raise ValueError("Unknown mode")

    config = {
        "configurable": {
            "thread_id": THREAD_ID,
        },
        "recursion_limit": 30,
    }

    with SqliteSaver.from_conn_string(
        "checkpoints_hitl.sqlite"
    ) as checkpointer:

        orchestrator = build_demo(
            checkpointer,
            mode,
        )

        snapshot = orchestrator.graph.get_state(
            config
        )

        if mode == "pause":
            if snapshot.interrupts:
                raise RuntimeError(
                    "A review is already pending"
                )

            result = orchestrator.graph.invoke(
                {
                    "prompt": "Calculate 125 * 48",
                    "require_review": True,
                },
                config=config,
            )

            if "__interrupt__" not in result:
                raise RuntimeError(
                    "Expected a human review interrupt"
                )

            request = result["__interrupt__"][0].value

            print("Question:", request["question"])
            print("Draft:", request["draft"])

            snapshot = orchestrator.graph.get_state(
                config
            )

            print("Next nodes:", snapshot.next)

        else:
            if not snapshot.interrupts:
                raise RuntimeError(
                    "No pending human review"
                )

            approved = mode == "approve"

            result = orchestrator.graph.invoke(
                Command(resume=approved),
                config=config,
            )

            print(
                "Review approved:",
                result["review_approved"],
            )
            print("Answer:", result["answer"])

            snapshot = orchestrator.graph.get_state(
                config
            )

            print("Next nodes:", snapshot.next)


if __name__ == "__main__":
    main()
