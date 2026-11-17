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
    from execution.workflow import ExecutionWorkflow, pending_approval

from execution.backends.slurm_sim import SlurmSimBackend
from execution.contracts import ExperimentJob
from execution.job_manager import DurableJobManager

JOB = Path(__file__).resolve().parents[1] / "examples" / "v841_job.json"


def example():
    return ExperimentJob.model_validate_json(JOB.read_text(encoding="utf-8"))


def config(id):
    return {"configurable": {"thread_id": id}}


@unittest.skipUnless(HAS_GRAPH, "langgraph unavailable in local validation environment")
class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.backend = SlurmSimBackend(self.path / "sim.db")
        self.manager = DurableJobManager(self.backend, self.path / "jobs.db")
        self.saver = InMemorySaver()
        self.graph = ExecutionWorkflow(self.manager, backend="slurm-sim").build(self.saver)
        self.job = example()

    def start(self, name="start"):
        cfg = config(name)
        out = self.graph.invoke({"job_document": self.job.model_dump(mode="json")}, config=cfg)
        self.assertIn("__interrupt__", out)
        self.assertIsNone(self.manager.get(self.job.job_id))
        return cfg

    def resume(self, cfg, *, decision="approve", digest=None, backend="slurm-sim"):
        pending = pending_approval(self.graph.get_state(cfg))
        return self.graph.invoke(Command(resume={
            "decision": decision, "sha256": digest or pending["sha256"],
            "backend": backend,
        }), config=cfg)

    def test_interrupt_happens_before_submission(self):
        cfg = self.start()
        self.assertEqual(pending_approval(self.graph.get_state(cfg))["job_id"], self.job.job_id)

    def test_restart_then_approve_and_sync(self):
        cfg = self.start("approve")
        graph = ExecutionWorkflow(self.manager, backend="slurm-sim").build(self.saver)
        pending = pending_approval(graph.get_state(cfg))
        result = graph.invoke(Command(resume={"decision": "approve", "sha256": pending["sha256"],
                                              "backend": "slurm-sim"}), config=cfg)
        self.assertEqual(self.manager.get(self.job.job_id).phase, "submitted")
        record = self.manager.get(self.job.job_id)
        self.backend.advance(record.handle)
        self.assertEqual(self.manager.refresh(self.job.job_id).phase, "running")
        self.backend.advance(record.handle)
        self.assertEqual(self.manager.refresh(self.job.job_id).phase, "succeeded")

    def test_rejection_no_side_effect(self):
        cfg = self.start("reject")
        out = self.resume(cfg, decision="reject")
        self.assertEqual(self.graph.get_state(cfg).values["status"], "rejected")
        self.assertIsNone(self.manager.get(self.job.job_id))

    def test_wrong_digest_no_side_effect(self):
        cfg = self.start("digest")
        self.resume(cfg, digest="a" * 64)
        self.assertEqual(self.graph.get_state(cfg).values["status"], "rejected")
        self.assertIsNone(self.manager.get(self.job.job_id))

    def test_wrong_backend_no_side_effect(self):
        cfg = self.start("backend")
        self.resume(cfg, backend="docker-local")
        self.assertEqual(self.graph.get_state(cfg).values["status"], "rejected")
        self.assertIsNone(self.manager.get(self.job.job_id))

    def test_reinvoke_does_not_duplicate_job(self):
        cfg = self.start("once")
        self.resume(cfg)
        before = self.manager.get(self.job.job_id)
        duplicate = self.manager.submit(self.job)
        self.assertEqual(before.handle, duplicate.handle)


if __name__ == "__main__":
    unittest.main()
