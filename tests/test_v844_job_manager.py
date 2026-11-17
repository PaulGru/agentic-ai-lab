import tempfile
import unittest
from pathlib import Path

from execution.backends.slurm_sim import SlurmSimBackend
from execution.contracts import ExperimentJob
from execution.job_manager import DurableJobManager

JOB = Path(__file__).resolve().parents[1] / "examples" / "v841_job.json"


def example():
    return ExperimentJob.model_validate_json(JOB.read_text(encoding="utf-8"))


class FailingBackend:
    def __init__(self):
        self.calls = 0

    def submit(self, job):
        self.calls += 1
        raise TimeoutError("maybe submitted")


class DurableManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.simpath = self.path / "sim.db"
        self.jobpath = self.path / "jobs.db"
        self.backend = SlurmSimBackend(self.simpath)
        self.manager = DurableJobManager(self.backend, self.jobpath)
        self.job = example()

    def test_roundtrip_across_restarts(self):
        a = self.manager.submit(self.job)
        self.assertEqual(a.phase, "submitted")
        self.assertTrue(a.handle.simulated)
        self.backend.advance(a.handle)
        reloaded = DurableJobManager(SlurmSimBackend(self.simpath), self.jobpath)
        self.assertEqual(reloaded.refresh(self.job.job_id).phase, "running")
        self.backend.advance(a.handle)
        final = reloaded.refresh(self.job.job_id)
        self.assertEqual(final.phase, "succeeded")
        self.assertTrue(reloaded.collect(self.job.job_id).simulated)
        self.assertEqual(reloaded.collect(self.job.job_id).files, {})

    def test_duplicate_submit_never_calls_backend_twice(self):
        a = self.manager.submit(self.job)
        b = self.manager.submit(self.job)
        self.assertEqual(a.handle, b.handle)

    def test_conflict_same_id_diff_digest(self):
        self.manager.submit(self.job)
        with self.assertRaises(ValueError):
            self.manager.submit(self.job.model_copy(update={"seed": 999}))

    def test_ambiguous_timeout_is_not_resubmitted(self):
        backend = FailingBackend()
        manager = DurableJobManager(backend, self.jobpath)
        self.assertEqual(manager.submit(self.job).phase, "needs_reconciliation")
        self.assertEqual(manager.submit(self.job).phase, "needs_reconciliation")
        self.assertEqual(backend.calls, 1)

    def test_crash_after_intent_has_no_auto_retry(self):
        # Simulate interrupted process after INSERT, before backend returns.
        from execution.job_manager import _canonical
        with self.manager._connect() as db:
            db.execute("""INSERT INTO jobs (job_id, spec_sha256, job_json, backend, phase, message)
                VALUES (?, ?, ?, ?, 'submission_reserved', 'interrupted')""",
                (self.job.job_id, self.job.fingerprint(),
                 _canonical(self.job.model_dump(mode="json")),
                 type(self.backend).__name__))
        self.assertEqual(self.manager.refresh(self.job.job_id).phase, "needs_reconciliation")
        self.assertIsNone(self.manager.submit(self.job).handle)

    def test_cancel_persisted(self):
        a = self.manager.submit(self.job)
        self.assertEqual(self.manager.cancel(self.job.job_id).phase, "cancelled")
        self.assertEqual(DurableJobManager(SlurmSimBackend(self.simpath), self.jobpath).get(self.job.job_id).phase,
                         "cancelled")

    def test_invalid_artifact_digest_detected(self):
        a = self.manager.submit(self.job)
        self.backend.advance(a.handle)
        self.backend.advance(a.handle)
        self.manager.refresh(self.job.job_id)
        with self.manager._connect() as db:
            db.execute("UPDATE jobs SET artifacts_sha256=? WHERE job_id=?",
                       ("0" * 64, self.job.job_id))
        with self.assertRaises(ValueError):
            self.manager.collect(self.job.job_id)

    def test_backend_mismatch_does_not_corrupt_ledger(self):
        from execution.backends.fake import FakeBackend
        self.manager.submit(self.job)
        different = DurableJobManager(FakeBackend(), self.jobpath)
        with self.assertRaises(ValueError):
            different.refresh(self.job.job_id)
        self.assertEqual(self.manager.get(self.job.job_id).phase, "submitted")

    def test_missing_backend_state_is_not_silent_success(self):
        a = self.manager.submit(self.job)
        with self.backend._connect() as db:
            db.execute("DELETE FROM simulated_jobs WHERE remote_id=?", (a.handle.remote_id,))
        self.assertEqual(self.manager.refresh(self.job.job_id).phase, "needs_reconciliation")


if __name__ == "__main__":
    unittest.main()
