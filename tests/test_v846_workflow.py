import tempfile
import unittest
from pathlib import Path

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:
    HAS_GRAPH = False
else:
    HAS_GRAPH = True
    from langgraph.types import Command
    from execution.campaign_workflow import PilotWorkflow, pending_pilot_approval

from execution.backends.local import LocalDockerBackend
from execution.job_manager import DurableJobManager
from execution.campaign import PilotProtocol, build_report
from test_v846_campaign import DeterministicTestTransport, protocol


@unittest.skipUnless(HAS_GRAPH, "langgraph not installed in validation environment")
class PilotWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.transport = DeterministicTestTransport()
        self.backend = LocalDockerBackend(execution_authorized=True, transport=self.transport)
        self.manager = DurableJobManager(self.backend, Path(self.temp.name) / "jobs.db")
        self.saver = InMemorySaver()
        self.graph = PilotWorkflow(self.manager, poll_seconds=0.001).build(self.saver)
        self.p = protocol()

    def start(self, name):
        cfg = {"configurable": {"thread_id": name}}
        result = self.graph.invoke({"protocol_document": self.p.model_dump(mode="json")}, config=cfg)
        self.assertIn("__interrupt__", result)
        return cfg

    def resume(self, cfg, decision="approve", digest=None, backend="docker-local"):
        pending = pending_pilot_approval(self.graph.get_state(cfg))
        return self.graph.invoke(Command(resume={"decision": decision,
                                                 "protocol_sha256": digest or pending["protocol_sha256"],
                                                 "backend": backend}), config=cfg)

    def test_approval_required_before_any_job(self):
        cfg = self.start("v846-before")
        self.assertEqual(pending_pilot_approval(self.graph.get_state(cfg))["kind"], "v846_campaign_approval")
        for job in self.p.jobs():
            self.assertIsNone(self.manager.get(job.job_id))
        self.assertEqual(self.transport.calls, [])

    def test_rejection_does_not_submit(self):
        cfg = self.start("v846-reject")
        self.resume(cfg, decision="reject")
        self.assertEqual(self.graph.get_state(cfg).values["status"], "rejected")
        self.assertEqual(self.transport.calls, [])

    def test_wrong_digest_or_backend_no_side_effects(self):
        for tag, digest, backend in [("digest", "a" * 64, "docker-local"),
                                     ("backend", None, "slurm-sim")]:
            cfg = self.start(f"v846-{tag}")
            self.resume(cfg, digest=digest, backend=backend)
            self.assertEqual(self.graph.get_state(cfg).values["status"], "rejected")
        self.assertEqual(self.transport.calls, [])

    def test_approved_runs_complete_then_build_report(self):
        cfg = self.start("v846-success")
        self.resume(cfg)
        snap = self.graph.get_state(cfg)
        self.assertEqual(snap.values["status"], "completed")
        self.assertEqual(snap.values["analysis"]["number_of_runs"], 2)
        self.assertEqual(len(self.transport.calls), 2)
        self.assertEqual(build_report(self.p, self.manager)["analysis"]["last_minus_first_mean"], 3.0)
        # Rebuilding the graph must not repeat any external side effect.
        rebuilt = PilotWorkflow(self.manager, poll_seconds=0.001).build(self.saver)
        self.assertEqual(rebuilt.get_state(cfg).values["status"], "completed")
        self.assertEqual(len(self.transport.calls), 2)


if __name__ == "__main__":
    unittest.main()
