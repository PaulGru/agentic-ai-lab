
import os
import re
import sys

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent.runner import AgentRunner
from orchestration.agent_tool_graph import AgentToolGraph
from tools.note_writer import WriteNoteTool
from tools.registry import ToolRegistry


def build_agent():
    provider_name = os.getenv(
        "LLM_PROVIDER", "openai"
    ).lower()

    if provider_name == "openai":
        from providers.openai_provider import OpenAIProvider
        
        provider = OpenAIProvider(
            model=os.getenv(
                "OPENAI_MODEL", "gpt-6-luna"
            )
        )
    elif provider_name == "anthropic":
        from providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(
            model=os.environ["ANTHROPIC_MODEL"]
        )
    else:
        raise ValueError(
            "LLM_PROVIDER must be openai or anthropic"
        )

    registry = ToolRegistry()
    registry.register(
        WriteNoteTool("v43_notes")
    )

    return AgentRunner(
        name="general",
        description=(
            "Assistant capable d'écrire des notes."
        ),
        instructions=(
            "Tu dois utiliser l'outil write_note "
            "pour enregistrer la note demandée. "
            "N'affirme jamais avoir écrit un fichier "
            "avant d'avoir reçu le résultat de l'outil. "
            "Si l'outil est refusé, explique que "
            "la note n'a pas été créée."
        ),
        provider=provider,
        registry=registry,
    )


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python demo_agent_tool_graph.py "
            "pause|approve|reject THREAD_ID"
        )

    mode, thread_id = sys.argv[1:]

    if mode not in {
        "pause", "approve", "reject"
    }:
        raise ValueError("Unknown mode")

    if not re.fullmatch(
        r"[A-Za-z0-9_-]{1,64}",
        thread_id,
    ):
        raise ValueError("Invalid thread_id")

    load_dotenv()

    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": 40,
    }

    with SqliteSaver.from_conn_string(
        "checkpoints_v43.sqlite"
    ) as checkpointer:

        graph = AgentToolGraph(
            build_agent(),
            checkpointer,
        ).graph

        snapshot = graph.get_state(config)

        if mode == "pause":
            if snapshot.values:
                raise RuntimeError(
                    "Thread already used"
                )

            result = graph.invoke(
                {
                    "prompt": (
                        f"Crée une note appelée {thread_id} "
                        "avec exactement le contenu "
                        "'Bonjour depuis la V4.3', "
                        "puis confirme le résultat."
                    ),
                },
                config=config,
            )

            if "__interrupt__" not in result:
                raise RuntimeError(
                    "Le modèle n'a demandé "
                    "aucune validation"
                )

        else:
            if not snapshot.interrupts:
                raise RuntimeError(
                    "No pending tool approval"
                )

            result = graph.invoke(
                Command(
                    resume=(mode == "approve")
                ),
                config=config,
            )

        if "__interrupt__" in result:
            action = result[
                "__interrupt__"
            ][0].value["action"]

            print("À valider:", action)

        else:
            print(
                "Assistant >", result["answer"]
            )

        print(
            "Next nodes:",
            graph.get_state(config).next,
        )


if __name__ == "__main__":
    main()
