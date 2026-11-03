import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.models import LLMResponse, ToolCall
from agent.runner import AgentRunner
from orchestration.agent_tool_graph import AgentToolGraph
from orchestration.approval_policy import ApprovalPolicy
from orchestration.router import Router
from tests.fakes import FakeProvider
from tools.calculator import CalculatorTool
from tools.note_writer import WriteNoteTool
from tools.registry import ToolRegistry


class StatefulFakeProvider(FakeProvider):
    def export_state(self):
        return {"remaining": [asdict(response) for response in self.responses]}

    def restore_state(self, state):
        self.responses = [
            LLMResponse(
                text=item["text"],
                tool_calls=[ToolCall(**c) for c in item["tool_calls"]],
            )
            for item in state["remaining"]
        ]


def agent(name, provider, tool=None):
    registry = ToolRegistry()
    if tool is not None:
        registry.register(tool)
    return AgentRunner(
        name=name, description=name + " agent",
        instructions="Use appropriate tools.",
        provider=provider, registry=registry,
    )


class V44PolicyTests(unittest.TestCase):
    def test_router_auto_calculation_and_edited_note(self):
        with TemporaryDirectory() as directory:
            router_provider = FakeProvider([
                LLMResponse(text="math"),
                LLMResponse(text="general"),
            ])
            math_provider = StatefulFakeProvider([
                LLMResponse(tool_calls=[ToolCall(
                    id="math-call", name="calculate",
                    arguments={"expression": "125 * 48"},
                )]),
                LLMResponse(text="6000"),
            ])
            general_provider = StatefulFakeProvider([
                LLMResponse(tool_calls=[ToolCall(
                    id="note-call", name="write_note",
                    arguments={"note_id": "example", "content": "Draft"},
                )]),
                LLMResponse(text="Note saved."),
            ])
            router = Router(
                provider=router_provider,
                agents=[
                    agent("math", math_provider, CalculatorTool()),
                    agent("general", general_provider, WriteNoteTool(directory)),
                ],
            )
            graph = AgentToolGraph(
                agent=None, checkpointer=InMemorySaver(),
                router=router,
                policy=ApprovalPolicy({"calculate": "auto", "write_note": "review"}),
            ).graph

            math_config = {"configurable": {"thread_id": "v44-math"}}
            result = graph.invoke({"prompt": "125 * 48"}, config=math_config)
            self.assertNotIn("__interrupt__", result)
            self.assertEqual(result["answer"], "6000")
            self.assertEqual(result["agent_name"], "math")
            self.assertEqual(len(math_provider.continue_calls), 1)

            note_config = {"configurable": {"thread_id": "v44-note"}}
            paused = graph.invoke({"prompt": "Write a note"}, config=note_config)
            self.assertIn("__interrupt__", paused)
            self.assertEqual(paused["__interrupt__"][0].value["agent"], "general")
            path = Path(directory) / "example.txt"
            self.assertFalse(path.exists())

            resumed = graph.invoke(Command(resume={
                "decision": "edit",
                "arguments": {"note_id": "example", "content": "Approved edit"},
            }), config=note_config)
            self.assertEqual(resumed["answer"], "Note saved.")
            self.assertEqual(path.read_text(encoding="utf-8"), "Approved edit")
            self.assertEqual(
                general_provider.continue_calls[0]["results"][0].call_id,
                "note-call",
            )
            self.assertEqual(len(router_provider.start_calls), 2)
            self.assertEqual(graph.get_state(note_config).next, ())

    def test_rejection_returns_error_without_writing(self):
        with TemporaryDirectory() as directory:
            provider = StatefulFakeProvider([
                LLMResponse(tool_calls=[ToolCall(
                    id="call", name="write_note",
                    arguments={"note_id": "rejected", "content": "Draft"},
                )]),
                LLMResponse(text="The note was not created."),
            ])
            graph = AgentToolGraph(
                agent("general", provider, WriteNoteTool(directory)),
                InMemorySaver(), policy=ApprovalPolicy({"write_note": "review"}),
            ).graph
            config = {"configurable": {"thread_id": "reject"}}
            graph.invoke({"prompt": "write"}, config=config)
            result = graph.invoke(Command(resume={"decision": "reject"}), config=config)
            self.assertEqual(result["answer"], "The note was not created.")
            self.assertFalse((Path(directory) / "rejected.txt").exists())
            self.assertTrue(provider.continue_calls[0]["results"][0].is_error)

    def test_unknown_tool_is_denied_even_if_marked_auto(self):
        provider = StatefulFakeProvider([
            LLMResponse(tool_calls=[ToolCall(
                id="bad-call", name="unknown_tool", arguments={},
            )]),
            LLMResponse(text="Blocked."),
        ])
        graph = AgentToolGraph(
            agent("general", provider), InMemorySaver(),
            policy=ApprovalPolicy({"unknown_tool": "auto"}),
        ).graph
        result = graph.invoke(
            {"prompt": "Try unknown tool"},
            config={"configurable": {"thread_id": "unknown"}},
        )
        self.assertNotIn("__interrupt__", result)
        self.assertEqual(result["answer"], "Blocked.")
        self.assertTrue(provider.continue_calls[0]["results"][0].is_error)

    def test_router_not_rerun_on_resume_with_new_objects(self):
        with TemporaryDirectory() as directory:
            saver = InMemorySaver()
            provider = StatefulFakeProvider([
                LLMResponse(tool_calls=[ToolCall(
                    id="c", name="write_note",
                    arguments={"note_id": "restart", "content": "V4.4"},
                )]),
                LLMResponse(text="Saved."),
            ])
            old_router = Router(
                provider=FakeProvider([LLMResponse(text="general")]),
                agents=[agent("general", provider, WriteNoteTool(directory))],
            )
            config = {"configurable": {"thread_id": "restart"}}
            first = AgentToolGraph(None, saver, router=old_router).graph
            paused = first.invoke({"prompt": "write note"}, config=config)
            self.assertIn("__interrupt__", paused)

            new_router_provider = FakeProvider([])
            new_agent_provider = StatefulFakeProvider([])
            new_router = Router(
                provider=new_router_provider,
                agents=[agent("general", new_agent_provider, WriteNoteTool(directory))],
            )
            second = AgentToolGraph(None, saver, router=new_router).graph
            final = second.invoke(Command(resume=True), config=config)
            self.assertEqual(final["answer"], "Saved.")
            self.assertEqual(len(new_router_provider.start_calls), 0)
            self.assertEqual(len(new_agent_provider.start_calls), 0)
            self.assertEqual(len(new_agent_provider.continue_calls), 1)
            self.assertTrue((Path(directory) / "restart.txt").exists())


if __name__ == "__main__":
    unittest.main()
