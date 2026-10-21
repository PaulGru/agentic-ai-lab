from dotenv import load_dotenv

from agent.runner import AgentRunner
from orchestration.router import Router
from providers.openai_provider import OpenAIProvider
from tools.calculator import CalculatorTool
from tools.registry import ToolRegistry
from observability.tracer import Tracer
from orchestration.graph_orchestrator import GraphOrchestrator
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver


load_dotenv()
tracer = Tracer()


# -------------------------
# Math agent
# -------------------------

math_provider = OpenAIProvider(
    model="gpt-6-luna",
)

math_registry = ToolRegistry()
math_registry.register(CalculatorTool())

math_agent = AgentRunner(
    name="math",
    description=(
        "Handles requests whose primary goal is an exact "
        "arithmetic or mathematical calculation."
    ),
    instructions=(
        "You are a math specialist. "
        "Use the calculate tool whenever an exact arithmetic "
        "calculation is required."
    ),
    provider=math_provider,
    registry=math_registry,
    tracer=tracer,
)


# -------------------------
# General agent
# -------------------------

general_provider = OpenAIProvider(
    model="gpt-6-luna",
)

general_registry = ToolRegistry()

general_agent = AgentRunner(
    name="general",
    description=(
        "Handles general explanations and mixed requests. "
        "It can delegate exact calculations to the math agent."
    ),
    instructions=(
        "You are a helpful general assistant. "
        "Answer clearly and concisely. "
        "If an exact arithmetic calculation is needed as part "
        "of a broader answer, delegate only that calculation "
        "to the math agent using handoff_to_agent. "
        "When the result is returned, use it to complete your "
        "final answer."
    ),
    provider=general_provider,
    registry=general_registry,
    tracer=tracer,
    handoff_targets={
        "math": (
            "Specialist for exact arithmetic "
            "and mathematical calculations."
        )
    },
)


# -------------------------
# Router
# -------------------------
from orchestration.orchestrator import Orchestrator

router_provider = OpenAIProvider(
    model="gpt-6-luna",
)

router = Router(
    provider=router_provider,
    agents=[
        math_agent,
        general_agent,
    ],
    tracer=tracer,
)


# -------------------------
# Persistent execution
# -------------------------

with SqliteSaver.from_conn_string(
    "checkpoints.sqlite"
) as checkpointer:

    graph_orchestrator = GraphOrchestrator(
        router=router,
        tracer=tracer,
        max_handoff_depth=3,
        checkpointer=checkpointer,
    )

    thread_id = str(uuid4())

    answer = graph_orchestrator.run(
        "Rédige une courte explication destinée à un débutant "
        "sur la différence entre coût unitaire et coût total. "
        "Dans ton explication, utilise l'exemple de 125 unités "
        "à 48 € chacune et donne le coût total exact.",
        thread_id=thread_id,
    )

    print()
    print("Assistant >", answer)

    # -------------------------
    # Inspect checkpoint
    # -------------------------

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    snapshot = graph_orchestrator.graph.get_state(
        config
    )

    print()
    print("Thread ID >", thread_id)
    print("Finished >", snapshot.values["finished"])
    print("Next nodes >", snapshot.next)

    history = list(
        graph_orchestrator.graph.get_state_history(
            config
        )
    )

    print("Checkpoints >", len(history))