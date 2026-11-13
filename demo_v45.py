"""Historical CLI demo (V4.5), using the shared legacy workflow factory."""
import re
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from tools.create_note_once import CreateNoteOnceTool
from workflows.legacy import build_graph, NOTE_DIRECTORY, CHECKPOINT_FILE


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
            tool = CreateNoteOnceTool(NOTE_DIRECTORY)
            print(tool.execute({"note_id": thread_id, "content": content}))
            print("Replay kept the existing file unchanged")

        print("File exists:", path.exists())
        print("Next nodes:", graph.get_state(config).next)


if __name__ == "__main__":
    main()
