
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.models import ToolCall
from orchestration.tool_approval_graph import (
    ToolApprovalGraph,
)
from tools.note_writer import WriteNoteTool
from tools.registry import ToolRegistry


class ToolApprovalTests(unittest.TestCase):

    def test_approval_controls_execution(self):
        for approved in (True, False):
            with self.subTest(approved=approved):
                with TemporaryDirectory() as directory:
                    registry = ToolRegistry()
                    registry.register(
                        WriteNoteTool(directory)
                    )

                    graph = ToolApprovalGraph(
                        registry=registry,
                        checkpointer=InMemorySaver(),
                    ).graph

                    config = {
                        "configurable": {
                            "thread_id": "test-action",
                        }
                    }

                    call = ToolCall(
                        id="call-1",
                        name="write_note",
                        arguments={
                            "note_id": "test",
                            "content": "Hello V4.2",
                        },
                    )

                    path = (
                        Path(directory) / "test.txt"
                    )

                    paused = graph.invoke(
                        {"action": asdict(call)},
                        config=config,
                    )

                    self.assertIn(
                        "__interrupt__",
                        paused,
                    )

                    # No side effects before approval.
                    self.assertFalse(path.exists())

                    result = graph.invoke(
                        Command(resume=approved),
                        config=config,
                    )

                    self.assertEqual(
                        result["status"],
                        "executed" if approved
                        else "rejected",
                    )

                    self.assertEqual(
                        path.exists(),
                        approved,
                    )

                    self.assertEqual(
                        graph.get_state(config).next,
                        (),
                    )


if __name__ == "__main__":
    unittest.main()
