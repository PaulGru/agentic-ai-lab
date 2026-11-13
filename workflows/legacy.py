"""V4.5/V5 deterministic workflow factory, separate from the CLI demo."""

from agent.models import LLMResponse, ToolCall
from agent.runner import AgentRunner
from orchestration.agent_tool_graph import AgentToolGraph
from orchestration.approval_policy import ApprovalPolicy
from orchestration.router import Router
from providers.replay_provider import FakeProvider, StatefulFakeProvider
from tools.calculator import CalculatorTool
from tools.create_note_once import CreateNoteOnceTool
from tools.registry import ToolRegistry

NOTE_DIRECTORY = "v45_notes"
CHECKPOINT_FILE = "checkpoints_v45.sqlite"


def build_graph(saver, thread_id: str, start: bool):
    router_provider = FakeProvider(
        [LLMResponse(text="general")] if start else []
    )
    general_provider = StatefulFakeProvider([
        LLMResponse(tool_calls=[ToolCall(
            id="handoff-1", name="handoff_to_agent",
            arguments={"agent": "math", "task": "Calculate 125 * 48"},
        )]),
        LLMResponse(tool_calls=[ToolCall(
            id="note-1", name="create_note_once",
            arguments={"note_id": thread_id, "content": "125 * 48 = 6000"},
        )]),
        LLMResponse(text="Traitement terminé. Vérifie le statut de l'outil."),
    ] if start else [])
    math_provider = StatefulFakeProvider([
        LLMResponse(tool_calls=[ToolCall(
            id="calc-1", name="calculate",
            arguments={"expression": "125 * 48"},
        )]),
        LLMResponse(text="6000"),
    ] if start else [])

    math_registry = ToolRegistry()
    math_registry.register(CalculatorTool())
    general_registry = ToolRegistry()
    general_registry.register(CreateNoteOnceTool(NOTE_DIRECTORY))

    general = AgentRunner(
        name="general", description="Creates local notes",
        instructions="Delegate arithmetic to math; save a note afterwards.",
        provider=general_provider, registry=general_registry,
        handoff_targets={"math": "Arithmetic specialist"},
    )
    math = AgentRunner(
        name="math", description="Arithmetic specialist",
        instructions="Calculate exactly using calculate.",
        provider=math_provider, registry=math_registry,
    )
    router = Router(provider=router_provider, agents=[general, math])
    graph = AgentToolGraph(
        agent=None, router=router, checkpointer=saver,
        policy=ApprovalPolicy({
            "calculate": "auto",
            "create_note_once": "review",
        }),
    ).graph
    return graph, router_provider, general_provider
