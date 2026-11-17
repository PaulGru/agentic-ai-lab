import tempfile
import unittest
from pathlib import Path

from execution.backends.slurm_sim import SlurmSimBackend, render_sbatch_preview, map_slurm_state
from execution.contracts import ExperimentJob, JobStatus


JOB = Path(__file__).resolve().parents[1] / "examples" / "v841_job.json"


def example():
    return ExperimentJob.model_validate_json(JOB.read_text(encoding="utf-8"))


class SlurmPreviewTests(unittest.TestCase):
    def test_safe_preview_without_any_execution(self):
        script = render_sbatch_preview(example())
        self.assertIn("#SBATCH --no-requeue", script)
        self.assertIn("--cpus-per-task=1", script)
        self.assertIn("result_envelope.json", script)
        self.assertNotIn("--gres=gpu", script)

    def test_no_unknown_workload(self):
        doc = example().model_dump()
        doc["workload"]["name"] = "danger.shell"
        with self.assertRaises(ValueError):
            render_sbatch_preview(ExperimentJob.model_validate(doc))

    def test_state_mapping_and_unknown(self):
        self.assertEqual(map_slurm_state("PENDING"), JobStatus.SUBMITTED)
        self.assertEqual(map_slurm_state("COMPLETED+"), JobStatus.SUCCEEDED)
        self.assertEqual(map_slurm_state("CANCELLED by 123"), JobStatus.CANCELLED)
        self.assertEqual(map_slurm_state("OUT_OF_MEMORY"), JobStatus.FAILED)
        with self.assertRaises(ValueError):
            map_slurm_state("WHATEVER")

    def test_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sim.db"
            first = SlurmSimBackend(path)
            handle = first.submit(example())
            self.assertEqual(first.status(handle).status, JobStatus.SUBMITTED)
            other = SlurmSimBackend(path)
            self.assertEqual(other.advance(handle).status, JobStatus.RUNNING)
            last = SlurmSimBackend(path)
            self.assertEqual(last.advance(handle).status, JobStatus.SUCCEEDED)
            self.assertTrue(last.collect(handle).simulated)
            self.assertEqual(last.collect(handle).files, {})

    def test_duplicate_idempotent_and_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = SlurmSimBackend(Path(tmp) / "sim.db")
            job = example()
            self.assertEqual(b.submit(job), b.submit(job))
            updated = job.model_copy(update={"seed": 13})
            with self.assertRaises(ValueError):
                b.submit(updated)

    def test_cancel_prevents_advance(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = SlurmSimBackend(Path(tmp) / "sim.db")
            handle = b.submit(example())
            self.assertEqual(b.cancel(handle).status, JobStatus.CANCELLED)
            with self.assertRaises(ValueError):
                b.advance(handle)
            with self.assertRaises(RuntimeError):
                b.collect(handle)

    def test_mismatched_handle_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = SlurmSimBackend(Path(tmp) / "sim.db")
            handle = b.submit(example())
            wrong = handle.model_copy(update={"spec_sha256": "a" * 64})
            with self.assertRaises(KeyError):
                b.status(wrong)


if __name__ == "__main__":
    unittest.main()
