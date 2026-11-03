
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.models import LLMResponse, ToolCall
from agent.runner import AgentRunner
from orchestration.agent_tool_graph import AgentToolGraph
from tests.fakes import FakeProvider
from tools.note_writer import WriteNoteTool
from tools.registry import ToolRegistry


class RestartableFakeProvider(FakeProvider):
    def export_state(self):
        return {
            "remaining": [
                asdict(r) for r in self.responses
            ]
        }

    def restore_state(self, state):
        self.responses = [
            LLMResponse(
                text=item["text"],
                tool_calls=[
                    ToolCall(**call)
                    for call in item["tool_calls"]
                ],
            )
            for item in state["remaining"]
        ]


def build_agent(provider, directory):
    registry = ToolRegistry()
    registry.register(WriteNoteTool(directory))

    return AgentRunner(
        name="general",
        description="Writes notes",
        instructions="Use tools when needed.",
        provider=provider,
        registry=registry,
    )


class AgentToolGraphTests(unittest.TestCase):

    def test_approval_and_rejection_after_restart(self):
        for approved in (True, False):
            with self.subTest(approved=approved):
                with TemporaryDirectory() as tmp:
                    checkpointer = InMemorySaver()
                    path = Path(tmp) / "test.txt"

                    first_provider = RestartableFakeProvider([
                        LLMResponse(
                            tool_calls=[
                                ToolCall(
                                    id="call-1",
                                    name="write_note",
                                    arguments={
                                        "note_id": "test",
                                        "content": "V4.3",
                                    },
                                )
                            ]
                        ),
                        LLMResponse(
                            text="Action traitée."
                        ),
                    ])

                    graph = AgentToolGraph(
                        build_agent(
                            first_provider, tmp
                        ),
                        checkpointer,
                    ).graph

                    config = {
                        "configurable": {
                            "thread_id": f"v43-{approved}",
                        }
                    }

                    paused = graph.invoke(
                        {"prompt": "Écris une note."},
                        config=config,
                    )

                    self.assertIn(
                        "__interrupt__", paused
                    )
                    self.assertFalse(path.exists())

                    self.assertEqual(
                        len(first_provider.start_calls),
                        1,
                    )

                    # Recreate agent and provider.
                    second_provider = (
                        RestartableFakeProvider([])
                    )

                    resumed_graph = AgentToolGraph(
                        build_agent(
                            second_provider, tmp
                        ),
                        checkpointer,
                    ).graph

                    final = resumed_graph.invoke(
                        Command(resume=approved),
                        config=config,
                    )

                    self.assertEqual(
                        final["answer"],
                        "Action traitée.",
                    )

                    self.assertEqual(
                        path.exists(), approved
                    )

                    # The first LLM call was not repeated.
                    self.assertEqual(
                        len(second_provider.start_calls),
                        0,
                    )

                    self.assertEqual(
                        len(second_provider.continue_calls),
                        1,
                    )

                    result = (
                        second_provider
                        .continue_calls[0]["results"][0]
                    )

                    self.assertEqual(
                        result.is_error,
                        not approved,
                    )

                    self.assertEqual(
                        resumed_graph.get_state(
                            config
                        ).next,
                        (),
                    )


if __name__ == "__main__":
    unittest.main()
