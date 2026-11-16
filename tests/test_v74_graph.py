
import json
import unittest

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.models import LLMResponse
from experiments.graph import ExperimentWorkflow
from experiments.sandbox import SandboxResult
from tests.fakes import FakeProvider

from scripts.demo_v74 import (
    demo_proposal,
    pending_approval,
)


class FakeSandbox:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = []

    def run(self, code, *, approved=False, dependencies=None):
        self.calls.append((code, approved, dependencies))

        status = self.statuses.pop(0)
        return SandboxResult(
            status=status,
            exit_code=0 if status == "success" else 1,
            stdout="3.0" if status == "success" else "",
            stderr="NameError" if status == "error" else "",
            duration_seconds=0.01,
        )


def config(name):
    return {"configurable": {"thread_id": name}}


def build(saver, sandbox, revisions=()):
    answers = iter(revisions)

    def factory():
        return FakeProvider([
            LLMResponse(
                text=json.dumps({
                    "code": next(answers)
                })
            )
        ])

    return ExperimentWorkflow(
        provider_factory=factory,
        sandbox=sandbox,
    ).build(saver)


def resume(graph, cfg, decision="approve", digest=None):
    snapshot = graph.get_state(cfg)
    pending = pending_approval(snapshot)

    assert pending is not None

    return graph.invoke(
        Command(resume={
            "decision": decision,
            "sha256": digest or pending["sha256"],
        }),
        config=cfg,
        durability="sync",
    )


class V74GraphTests(unittest.TestCase):

    def test_initial_interrupt_prevents_execution(self):
        saver = InMemorySaver()
        sandbox = FakeSandbox([])
        graph = build(saver, sandbox)

        response = graph.invoke(
            {"initial_proposal": demo_proposal().model_dump()},
            config=config("initial"),
            durability="sync",
        )

        self.assertIn("__interrupt__", response)
        self.assertEqual(sandbox.calls, [])

    def test_resume_after_graph_reconstruction(self):
        saver = InMemorySaver()
        sandbox = FakeSandbox(["success"])
        cfg = config("restart")

        graph = build(saver, sandbox)
        graph.invoke(
            {"initial_proposal": demo_proposal().model_dump()},
            config=cfg,
        )

        # Simule la reconstruction du graphe après
        # fermeture du processus Python.
        restarted = build(saver, sandbox)

        self.assertIsNotNone(
            pending_approval(restarted.get_state(cfg))
        )

        resume(restarted, cfg)

        final = restarted.get_state(cfg)
        self.assertEqual(
            final.values["status"], "completed"
        )
        self.assertEqual(len(sandbox.calls), 1)

    def test_rejection_never_executes_code(self):
        saver = InMemorySaver()
        sandbox = FakeSandbox([])
        cfg = config("rejected")
        graph = build(saver, sandbox)

        graph.invoke(
            {"initial_proposal": demo_proposal().model_dump()},
            config=cfg,
        )

        resume(graph, cfg, decision="reject")

        self.assertEqual(sandbox.calls, [])
        self.assertEqual(
            graph.get_state(cfg).values["status"],
            "rejected",
        )

    def test_wrong_code_digest_is_rejected(self):
        saver = InMemorySaver()
        sandbox = FakeSandbox([])
        cfg = config("wrong-digest")
        graph = build(saver, sandbox)

        graph.invoke(
            {"initial_proposal": demo_proposal().model_dump()},
            config=cfg,
        )

        resume(
            graph,
            cfg,
            digest="invalid-sha256",
        )

        self.assertEqual(sandbox.calls, [])
        self.assertEqual(
            graph.get_state(cfg).values["status"],
            "rejected",
        )

    def test_failure_revision_requires_new_approval(self):
        saver = InMemorySaver()
        sandbox = FakeSandbox(["error", "success"])
        cfg = config("correction")

        graph = build(
            saver,
            sandbox,
            revisions=["print(3.0)"],
        )

        graph.invoke(
            {"initial_proposal": demo_proposal().model_dump()},
            config=cfg,
        )

        resume(graph, cfg)

        snapshot = graph.get_state(cfg)

        self.assertEqual(len(sandbox.calls), 1)
        self.assertEqual(
            len(snapshot.values["attempts"]), 1
        )
        self.assertIsNotNone(pending_approval(snapshot))

        resume(graph, cfg)

        final = graph.get_state(cfg)

        self.assertEqual(len(sandbox.calls), 2)
        self.assertEqual(
            final.values["status"], "completed"
        )

    def test_maximum_three_executions(self):
        saver = InMemorySaver()
        sandbox = FakeSandbox(
            ["error", "error", "error"]
        )
        cfg = config("bounded")

        graph = build(
            saver,
            sandbox,
            revisions=[
                "print(1)",
                "print(2)",
            ],
        )

        graph.invoke(
            {"initial_proposal": demo_proposal().model_dump()},
            config=cfg,
        )

        for _ in range(3):
            resume(graph, cfg)

        final = graph.get_state(cfg)

        self.assertEqual(len(sandbox.calls), 3)
        self.assertEqual(
            final.values["status"],
            "max_attempts",
        )
        self.assertEqual(final.next, ())

    def test_external_dependencies_blocked(self):
        saver = InMemorySaver()
        sandbox = FakeSandbox([])
        cfg = config("dependencies")
        graph = build(saver, sandbox)

        proposal = demo_proposal().model_copy(
            update={"dependencies": ["numpy==2.3.0"]}
        )

        graph.invoke(
            {"initial_proposal": proposal.model_dump()},
            config=cfg,
        )

        snapshot = graph.get_state(cfg)

        self.assertEqual(
            snapshot.values["status"],
            "unsupported_dependencies",
        )
        self.assertEqual(sandbox.calls, [])
        self.assertEqual(snapshot.next, ())


if __name__ == "__main__":
    unittest.main()
