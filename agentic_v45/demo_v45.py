"""Free, deterministic V4.5 demonstration: router -> handoff -> tool -> HITL."""
import re
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent.models import LLMResponse, ToolCall
from agent.runner import AgentRunner
from orchestration.agent_tool_graph import AgentToolGraph
from orchestration.approval_policy import ApprovalPolicy
from orchestration.router import Router
from tests.fakes import FakeProvider
from tests.stateful_provider import StatefulFakeProvider
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


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python demo_v45.py pause|approve|reject|edit|replay THREAD_ID")
    mode, thread_id = sys.argv[1:]
    if mode not in {"pause", "approve", "reject", "edit", "replay"}:
        raise ValueError("Unknown mode")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", thread_id):
        raise ValueError("Invalid thread_id")

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 60,
    }
    path = Path(NOTE_DIRECTORY) / f"{thread_id}.txt"

    with SqliteSaver.from_conn_string(CHECKPOINT_FILE) as saver:
        graph, router_provider, general_provider = build_graph(
            saver, thread_id, start=mode == "pause"
        )
        snapshot = graph.get_state(config)

        if mode == "pause":
            if snapshot.values:
                raise RuntimeError("Thread already used; choose another ID")
            result = graph.invoke(
                {"prompt": "Calculate 125 * 48 and save it to a note"},
                config=config,
            )
            if "__interrupt__" not in result:
                raise RuntimeError("Expected a pending approval")
            print("Proposed action:", result["__interrupt__"][0].value)
            print("Router calls:", len(router_provider.start_calls))
        elif mode in {"approve", "reject", "edit"}:
            if not snapshot.interrupts:
                raise RuntimeError("No pending approval")
            if mode == "edit":
                original = snapshot.interrupts[0].value["action"]
                arguments = dict(original["arguments"])
                arguments["content"] = "125 * 48 = 6000 (corrigé par l'humain)"
                decision = {"decision": "edit", "arguments": arguments}
            else:
                decision = {"decision": mode}
            result = graph.invoke(Command(resume=decision), config=config)
            if "__interrupt__" in result:
                print("Another approval required:", result["__interrupt__"][0].value)
            else:
                print("Assistant >", result["answer"])
                if general_provider.continue_calls:
                    outcome = general_provider.continue_calls[-1]["results"][-1]
                    print("Tool result:", outcome.output)
                    print("Tool error:", outcome.is_error)
                print("Router calls during resume:", len(router_provider.start_calls))
        else:
            if snapshot.next or not snapshot.values:
                raise RuntimeError("Finish the workflow before replaying")
            if not path.exists():
                raise RuntimeError("No saved note to replay")
            content = path.read_text(encoding="utf-8")
            # Repeat the same desired effect: the create-once tool is idempotent.
            tool = CreateNoteOnceTool(NOTE_DIRECTORY)
            print(tool.execute({"note_id": thread_id, "content": content}))
            print("Replay kept the existing file unchanged")

        print("File exists:", path.exists())
        print("Next nodes:", graph.get_state(config).next)


if __name__ == "__main__":
    main()
