"""V8.4.1 contract tests — offline, no LLM, Docker or Slurm needed."""
import json
import math
import unittest

from pydantic import ValidationError

from execution.backends.base import ExecutionBackend
from execution.backends.fake import FakeBackend
from execution.contracts import ExperimentJob, JobHandle, JobStatus
from execution.examples import sample_job


class V841ContractTests(unittest.TestCase):
    def test_sample_is_valid_and_roundtrips(self):
        job = sample_job()
        restored = ExperimentJob.model_validate_json(job.model_dump_json())
        self.assertEqual(restored, job)
        self.assertEqual(restored.fingerprint(), job.fingerprint())

    def test_backend_is_abstract(self):
        with self.assertRaises(TypeError):
            ExecutionBackend()

    def test_schema_version_rejected(self):
        data = sample_job().model_dump()
        data["schema_version"] = "8.4"
        with self.assertRaises(ValidationError):
            ExperimentJob.model_validate(data)

    def test_malformed_workload_rejected(self):
        data = sample_job().model_dump()
        data["workload"]["name"] = "python;rm -rf ."
        with self.assertRaises(ValidationError):
            ExperimentJob.model_validate(data)

    def test_artifact_path_traversal_rejected(self):
        data = sample_job().model_dump()
        data["expected_artifacts"] = ["../out.json"]
        with self.assertRaises(ValidationError):
            ExperimentJob.model_validate(data)

    def test_duplicate_artifacts_rejected(self):
        data = sample_job().model_dump()
        data["expected_artifacts"] = ["metrics.json", "metrics.json"]
        with self.assertRaises(ValidationError):
            ExperimentJob.model_validate(data)

    def test_resource_limits(self):
        data = sample_job().model_dump()
        data["resources"]["gpus"] = 100
        with self.assertRaises(ValidationError):
            ExperimentJob.model_validate(data)

    def test_non_json_payload_rejected(self):
        data = sample_job().model_dump()
        data["workload"]["parameters"] = {"bad": object()}
        with self.assertRaises(ValidationError):
            ExperimentJob.model_validate(data)

    def test_non_finite_float_rejected(self):
        data = sample_job().model_dump()
        data["workload"]["parameters"] = {"value": float("nan")}
        with self.assertRaises(ValidationError):
            ExperimentJob.model_validate(data)

    def test_submit_is_idempotent_within_backend_instance(self):
        backend = FakeBackend()
        job = sample_job()
        first = backend.submit(job)
        second = backend.submit(job.model_copy(deep=True))
        self.assertEqual(first, second)
        self.assertEqual(backend.status(first).status, JobStatus.SUBMITTED)

    def test_same_job_id_different_spec_rejected(self):
        backend = FakeBackend()
        job = sample_job()
        backend.submit(job)
        changed = job.model_copy(update={"seed": 101})
        with self.assertRaisesRegex(ValueError, "different specification"):
            backend.submit(changed)

    def test_fake_lifecycle_has_no_results(self):
        backend = FakeBackend()
        handle = backend.submit(sample_job())
        with self.assertRaises(RuntimeError):
            backend.collect(handle)
        backend.mark_running(handle)
        backend.finish(handle, success=True)
        outcome = backend.collect(handle)
        self.assertEqual(outcome.files, {})
        self.assertTrue(outcome.simulated)
        self.assertIn("SIMULATED", outcome.warning)

    def test_failed_run_not_collectible(self):
        backend = FakeBackend()
        handle = backend.submit(sample_job())
        backend.mark_running(handle)
        backend.finish(handle, success=False)
        self.assertEqual(backend.status(handle).status, JobStatus.FAILED)
        with self.assertRaises(RuntimeError):
            backend.collect(handle)

    def test_cancel_and_terminal_states(self):
        backend = FakeBackend()
        handle = backend.submit(sample_job())
        self.assertEqual(backend.cancel(handle).status, JobStatus.CANCELLED)
        with self.assertRaises(ValueError):
            backend.mark_running(handle)
        with self.assertRaises(RuntimeError):
            backend.collect(handle)

    def test_handle_tampering_rejected(self):
        backend = FakeBackend()
        handle = backend.submit(sample_job())
        fake = handle.model_copy(update={"remote_id": "fake:another"})
        with self.assertRaises(KeyError):
            backend.status(fake)

    def test_unexpected_output_rejected(self):
        backend = FakeBackend()
        handle = backend.submit(sample_job())
        backend.mark_running(handle)
        with self.assertRaises(ValueError):
            backend.finish(handle, success=True, files={"unknown.json": {}})
        self.assertEqual(backend.status(handle).status, JobStatus.RUNNING)

    def test_multiple_jobs_are_independent(self):
        backend = FakeBackend()
        a = backend.submit(sample_job())
        b = backend.submit(sample_job().model_copy(update={"job_id": "synthetic-mean-002"}))
        backend.cancel(a)
        self.assertEqual(backend.status(a).status, JobStatus.CANCELLED)
        self.assertEqual(backend.status(b).status, JobStatus.SUBMITTED)


if __name__ == "__main__":
    unittest.main()
