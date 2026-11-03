
import re
import sys
from dataclasses import asdict
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent.models import ToolCall
from orchestration.tool_approval_graph import (
    ToolApprovalGraph,
)
from tools.note_writer import WriteNoteTool
from tools.registry import ToolRegistry


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python demo_tool_approval.py "
            "pause|approve|reject THREAD_ID"
        )

    mode, thread_id = sys.argv[1:]

    if mode not in ("pause", "approve", "reject"):
        raise ValueError("Unknown mode")

    if not re.fullmatch(
        r"[a-zA-Z0-9_-]{1,64}",
        thread_id,
    ):
        raise ValueError("Invalid thread_id")

    registry = ToolRegistry()
    registry.register(WriteNoteTool())

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    note_path = (
        Path("v42_notes") / f"{thread_id}.txt"
    )

    with SqliteSaver.from_conn_string(
        "checkpoints_tool_approval.sqlite"
    ) as checkpointer:

        graph = ToolApprovalGraph(
            registry=registry,
            checkpointer=checkpointer,
        ).graph

        snapshot = graph.get_state(config)

        if mode == "pause":
            if snapshot.values:
                raise RuntimeError(
                    "Thread already used. "
                    "Choose a new thread_id."
                )

            # Simulated tool call from an agent
            call = ToolCall(
                id=f"call-{thread_id}",
                name="write_note",
                arguments={
                    "note_id": thread_id,
                    "content": (
                        "V4.2: Action approved by a human."
                    ),
                },
            )

            result = graph.invoke(
                {"action": asdict(call)},
                config=config,
            )

            pending = result["__interrupt__"][0]

            print(
                "Pending action:",
                pending.value["action"],
            )

        else:
            if not snapshot.interrupts:
                raise RuntimeError(
                    "No pending tool approval"
                )

            result = graph.invoke(
                Command(resume=mode == "approve"),
                config=config,
            )

            print("Status:", result["status"])
            print(
                "Tool output:",
                result.get("tool_output", "None"),
            )

        print("File exists:", note_path.exists())
        print("Next nodes:", graph.get_state(config).next)


if __name__ == "__main__":
    main()
