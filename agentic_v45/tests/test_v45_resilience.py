import unittest
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
from tests.stateful_provider import StatefulFakeProvider
from tools.calculator import CalculatorTool
from tools.create_note_once import CreateNoteOnceTool
from tools.registry import ToolRegistry


def make_agent(name, provider, tool=None, targets=None):
    registry = ToolRegistry()
    if tool is not None:
        registry.register(tool)
    return AgentRunner(
        name=name, description=name, instructions="Use tools.",
        provider=provider, registry=registry,
        handoff_targets=targets,
    )


def make_handoff_graph(saver, directory, first, policy=None):
    router_provider = FakeProvider(
        [LLMResponse(text="general")] if first else []
    )
    general_provider = StatefulFakeProvider([
        LLMResponse(tool_calls=[ToolCall(
            id="handoff-01", name="handoff_to_agent",
            arguments={"agent": "math", "task": "Calculate 125 * 48"},
        )]),
        LLMResponse(tool_calls=[ToolCall(
            id="write-01", name="create_note_once",
            arguments={"note_id": "test", "content": "Result: 6000"},
        )]),
        LLMResponse(text="Finished."),
    ] if first else [])
    math_provider = StatefulFakeProvider([
        LLMResponse(tool_calls=[ToolCall(
            id="calc-01", name="calculate",
            arguments={"expression": "125 * 48"},
        )]),
        LLMResponse(text="6000"),
    ] if first else [])
    router = Router(provider=router_provider, agents=[
        make_agent("general", general_provider,
                   CreateNoteOnceTool(directory), {"math": "math"}),
        make_agent("math", math_provider, CalculatorTool()),
    ])
    graph = AgentToolGraph(
        agent=None, router=router, checkpointer=saver,
        policy=policy if policy is not None else ApprovalPolicy({
            "calculate": "auto", "create_note_once": "review",
        }),
    ).graph
    return graph, router_provider, general_provider, math_provider


class V45ResilienceTests(unittest.TestCase):
    def test_handoff_and_restart_then_approve_or_reject(self):
        for approved in (True, False):
            with self.subTest(approved=approved), TemporaryDirectory() as tmp:
                saver = InMemorySaver()
                config = {"configurable": {"thread_id": f"v45-{approved}"}}
                path = Path(tmp) / "test.txt"
                first, router1, general1, math1 = make_handoff_graph(saver, tmp, True)
                paused = first.invoke({"prompt": "Calculate and save"}, config=config)

                self.assertIn("__interrupt__", paused)
                self.assertEqual(paused["__interrupt__"][0].value["agent"], "general")
                self.assertFalse(path.exists())
                self.assertEqual(len(router1.start_calls), 1)
                self.assertEqual(len(math1.start_calls), 1)
                self.assertEqual(len(math1.continue_calls), 1)
                self.assertEqual(len(general1.continue_calls), 1)
                handoff_result = general1.continue_calls[0]["results"][0]
                self.assertEqual(handoff_result.name, "handoff_to_agent")
                self.assertEqual(handoff_result.call_id, "handoff-01")
                self.assertEqual(handoff_result.output, "6000")

                resumed, router2, general2, math2 = make_handoff_graph(saver, tmp, False)
                done = resumed.invoke(
                    Command(resume={"decision": "approve" if approved else "reject"}),
                    config=config,
                )
                self.assertEqual(done["answer"], "Finished.")
                self.assertEqual(resumed.get_state(config).next, ())
                self.assertEqual(path.exists(), approved)
                self.assertEqual(len(router2.start_calls), 0)
                self.assertEqual(len(general2.start_calls), 0)
                self.assertEqual(len(math2.start_calls), 0)
                self.assertEqual(len(general2.continue_calls), 1)
                note_result = general2.continue_calls[0]["results"][0]
                self.assertEqual(note_result.call_id, "write-01")
                self.assertEqual(note_result.is_error, not approved)

    def test_policy_changed_while_pending_must_fail_closed(self):
        with TemporaryDirectory() as tmp:
            saver = InMemorySaver()
            config = {"configurable": {"thread_id": "drift"}}
            first, _, _, _ = make_handoff_graph(saver, tmp, True)
            self.assertIn("__interrupt__", first.invoke({"prompt": "save"}, config=config))
            new_policy = ApprovalPolicy({
                "calculate": "auto", "create_note_once": "deny",
            })
            second, _, general, _ = make_handoff_graph(saver, tmp, False, new_policy)
            done = second.invoke(Command(resume={"decision": "approve"}), config=config)
            self.assertEqual(done["answer"], "Finished.")
            self.assertFalse((Path(tmp) / "test.txt").exists())
            result = general.continue_calls[0]["results"][0]
            self.assertTrue(result.is_error)
            self.assertIn("blocked", result.output)

    def test_idempotent_note_replay_and_conflict(self):
        with TemporaryDirectory() as tmp:
            tool = CreateNoteOnceTool(tmp)
            data = {"note_id": "sample", "content": "first"}
            first = tool.execute(data)
            second = tool.execute(data)
            self.assertEqual(first, second)
            path = Path(tmp) / "sample.txt"
            self.assertEqual(path.read_text(encoding="utf-8"), "first")
            with self.assertRaisesRegex(ValueError, "different content"):
                tool.execute({"note_id": "sample", "content": "other"})
            self.assertEqual(path.read_text(encoding="utf-8"), "first")
            with self.assertRaises(ValueError):
                tool.execute({"note_id": "../escape", "content": "bad"})

    def test_replaying_execute_node_does_not_overwrite_the_note(self):
        """Simulate a crash after the tool ran but before its graph checkpoint."""
        with TemporaryDirectory() as tmp:
            agent = make_agent(
                "general", StatefulFakeProvider([]), CreateNoteOnceTool(tmp)
            )
            orchestrator = AgentToolGraph(
                agent, InMemorySaver(),
                policy=ApprovalPolicy({"create_note_once": "review"}),
            )
            call = {
                "id": "once-1", "name": "create_note_once",
                "arguments": {"note_id": "replayed", "content": "first"},
            }
            state = {
                "current_agent": "general",
                "pending_calls": [call],
                "pending_index": 0,
                "tool_results": [],
                "approved": True,
                "approval_status": "approved",
                "effective_action": call,
            }
            first_result = orchestrator._execute_tool_node(state)
            # The state update is intentionally not applied (lost checkpoint).
            second_result = orchestrator._execute_tool_node(state)
            self.assertEqual(first_result, second_result)
            self.assertFalse(first_result["tool_results"][0]["is_error"])
            self.assertEqual(
                (Path(tmp) / "replayed.txt").read_text(encoding="utf-8"),
                "first",
            )

    def test_edit_invalid_argument_type_does_not_write(self):
        with TemporaryDirectory() as tmp:
            saver = InMemorySaver()
            config = {"configurable": {"thread_id": "bad-edit"}}
            first, _, _, _ = make_handoff_graph(saver, tmp, True)
            first.invoke({"prompt": "save"}, config=config)
            with self.assertRaisesRegex(ValueError, "types must not change"):
                first.invoke(Command(resume={
                    "decision": "edit",
                    "arguments": {"note_id": "test", "content": 42},
                }), config=config)
            self.assertFalse((Path(tmp) / "test.txt").exists())

    def test_handoff_to_unregistered_target_is_rejected(self):
        provider = StatefulFakeProvider([LLMResponse(tool_calls=[
            ToolCall(id="bad", name="handoff_to_agent",
                     arguments={"agent": "other", "task": "Do it"}),
        ])])
        agent = make_agent("general", provider, targets={"math": "Math"})
        graph = AgentToolGraph(agent, InMemorySaver()).graph
        with self.assertRaisesRegex(RuntimeError, "not permitted"):
            graph.invoke(
                {"prompt": "Start"},
                config={"configurable": {"thread_id": "invalid-target"}},
            )

    def test_unknown_tool_is_denied(self):
        provider = StatefulFakeProvider([
            LLMResponse(tool_calls=[ToolCall(id="1", name="unknown_tool", arguments={})]),
            LLMResponse(text="Blocked."),
        ])
        agent = make_agent("general", provider)
        graph = AgentToolGraph(
            agent, InMemorySaver(), policy=ApprovalPolicy({"unknown_tool": "auto"})
        ).graph
        result = graph.invoke(
            {"prompt": "Try"},
            config={"configurable": {"thread_id": "unknown-v45"}},
        )
        self.assertEqual(result["answer"], "Blocked.")
        self.assertTrue(provider.continue_calls[0]["results"][0].is_error)


if __name__ == "__main__":
    unittest.main()
