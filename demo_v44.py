import os
import re
import sys

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent.runner import AgentRunner
from orchestration.agent_tool_graph import AgentToolGraph
from orchestration.approval_policy import ApprovalPolicy
from orchestration.router import Router
from tools.calculator import CalculatorTool
from tools.note_writer import WriteNoteTool
from tools.registry import ToolRegistry


def make_provider():
    name = os.getenv("LLM_PROVIDER", "openai").lower()
    if name == "openai":
        from providers.openai_provider import OpenAIProvider
        return OpenAIProvider(os.getenv("OPENAI_MODEL", "gpt-6-luna"))
    if name == "anthropic":
        from providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(os.environ["ANTHROPIC_MODEL"])
    raise ValueError("LLM_PROVIDER must be openai or anthropic")


def make_router():
    math_tools = ToolRegistry()
    math_tools.register(CalculatorTool())
    note_tools = ToolRegistry()
    note_tools.register(WriteNoteTool("v44_notes"))

    math_agent = AgentRunner(
        name="math", description="Handles exact arithmetic calculations.",
        instructions="Use calculate for arithmetic and report its result.",
        provider=make_provider(), registry=math_tools,
    )
    general_agent = AgentRunner(
        name="general", description="Writes local notes when asked.",
        instructions=(
            "Use write_note when the user requests a note. "
            "Never claim it was saved without a successful tool result. "
            "If refused, explain that the note was not created."
        ),
        provider=make_provider(), registry=note_tools,
    )
    return Router(
        provider=make_provider(),
        agents=[math_agent, general_agent],
    )


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python demo_v44.py math|note|approve|reject|edit THREAD_ID")

    mode, thread_id = sys.argv[1:]
    if mode not in {"math", "note", "approve", "reject", "edit"}:
        raise ValueError("Invalid mode")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", thread_id):
        raise ValueError("Invalid thread_id")

    load_dotenv()
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 40,
    }
    policy = ApprovalPolicy({
        "calculate": "auto",
        "write_note": "review",
    })

    with SqliteSaver.from_conn_string("checkpoints_v44.sqlite") as saver:
        graph = AgentToolGraph(
            agent=None, checkpointer=saver,
            policy=policy, router=make_router(),
        ).graph
        snapshot = graph.get_state(config)

        if mode in {"math", "note"}:
            if snapshot.values:
                raise RuntimeError("Thread already used; choose a new THREAD_ID")
            prompt = (
                "Calcule exactement 125 * 48 en utilisant l'outil calculate."
                if mode == "math" else
                f"Utilise write_note pour créer la note {thread_id} "
                "avec le contenu 'Brouillon V4.4', puis confirme le résultat."
            )
            result = graph.invoke({"prompt": prompt}, config=config)
        else:
            if not snapshot.interrupts:
                raise RuntimeError("No pending tool approval")
            if mode == "edit":
                action = snapshot.interrupts[0].value["action"]
                if action["name"] != "write_note":
                    raise ValueError("This demo only edits write_note")
                args = dict(action["arguments"])
                args["content"] = "Contenu corrigé par un humain en V4.4"
                decision = {"decision": "edit", "arguments": args}
            else:
                decision = {"decision": "approve" if mode == "approve" else "reject"}
            result = graph.invoke(Command(resume=decision), config=config)

        if "__interrupt__" in result:
            print("À valider:", result["__interrupt__"][0].value)
        else:
            print("Agent sélectionné:", result["agent_name"])
            print("Assistant >", result["answer"])
        print("Next nodes:", graph.get_state(config).next)


if __name__ == "__main__":
    main()
