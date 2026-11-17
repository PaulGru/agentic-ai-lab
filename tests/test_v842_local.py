"""V8.4.2 — offline safety and lifecycle tests. No Docker or LLM used."""
import json
import threading
import time
import unittest

from execution.backends.docker_transport import DockerOutcome, docker_command
from execution.backends.local import LocalDockerBackend
from execution.contracts import ExperimentJob, JobStatus
from execution.examples import sample_job
from execution.workloads import decode_artifacts, render_source, validate_workload


def good_output(job):
    return json.dumps({
        "kind": "agentic-v842-artifacts",
        "sha256": job.fingerprint(),
        "files": {
            "metrics.json": {"mean": 3.0, "count": 5, "minimum": 1, "maximum": 5},
            "run_metadata.json": {
                "job_sha256": job.fingerprint(), "job_id": job.job_id,
                "execution": "docker_sandbox", "workload": job.workload.name,
                "seed": job.seed, "python_version": "3.12.0",
            },
        },
    })


class StubTransport:
    simulated = True

    def __init__(self, result="success"):
        self.calls = []
        self.result = result

    def run(self, job, code, cancel):
        self.calls.append((job, code))
        return DockerOutcome(
            status=self.result, exit_code=0 if self.result == "success" else 1,
            stdout=good_output(job) if self.result == "success" else "",
            stderr="", duration_seconds=0.01,
        )


class BlockingTransport(StubTransport):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()

    def run(self, job, code, cancel):
        self.calls.append((job, code))
        self.started.set()
        cancel.wait(timeout=2)
        return DockerOutcome("cancelled", 137, "", "", 0.05)


class V842LocalTests(unittest.TestCase):
    def test_workload_is_allowlisted(self):
        job = sample_job().model_copy(deep=True)
        job.workload.name = "custom.python"
        with self.assertRaisesRegex(ValueError, "Unsupported workload"):
            render_source(job)

    def test_workload_rejects_malformed_data(self):
        for values in ([], [True, 2], ["1"], [float("nan")], [float("inf")],
                       [1_000_001], list(range(101)), [10**10000]):
            with self.subTest(label=str(type(values[0])) if values else "empty"):
                job = sample_job().model_copy(deep=True)
                job.workload.parameters = {"values": values}
                with self.assertRaises((ValueError, OverflowError)):
                    validate_workload(job)

    def test_workload_rejects_unexpected_parameters_and_artifacts(self):
        job = sample_job().model_copy(deep=True)
        job.workload.parameters = {"values": [1], "code": "print('bad')"}
        with self.assertRaises(ValueError):
            render_source(job)
        job = sample_job().model_copy(deep=True)
        job.expected_artifacts = ["metrics.json"]
        with self.assertRaises(ValueError):
            render_source(job)

    def test_no_gpu_and_enforce_local_limits(self):
        for field, value in (("gpus", 1), ("cpus", 3), ("memory_mb", 1024), ("walltime_minutes", 3)):
            job = sample_job().model_copy(deep=True)
            setattr(job.resources, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                render_source(job)

    def test_generated_code_uses_constant_template_and_repr_of_data(self):
        code = render_source(sample_job())
        self.assertIn("statistics.fmean", code)
        self.assertIn("agentic-v842-artifacts", code)
        self.assertNotIn("subprocess", code)
        compile(code, "<trusted workload>", "exec")

    def test_strong_docker_isolation_and_resource_limits(self):
        cmd = docker_command(sample_job(), "agentic-v842-test")
        for arg in ("--network=none", "--read-only", "--cap-drop=ALL",
                    "--security-opt=no-new-privileges=true", "--security-opt=seccomp=builtin",
                    "--pids-limit=32", "--memory=256m", "--memory-swap=256m",
                    "--cpus=1", "--user=10001:10001", "--pull=never", "--log-driver=none"):
            self.assertIn(arg, cmd)
        self.assertFalse(any(arg.startswith("--volume") or arg.startswith("--mount") for arg in cmd))
        self.assertNotIn("--privileged", cmd)
        self.assertEqual(cmd[-1], "agentic-lab-sandbox:v7.2")

    def test_submit_requires_execution_authorization(self):
        stub = StubTransport()
        backend = LocalDockerBackend(transport=stub)
        with self.assertRaises(PermissionError):
            backend.submit(sample_job())
        self.assertEqual(stub.calls, [])

    def test_submit_collect_and_idempotency(self):
        stub = StubTransport()
        backend = LocalDockerBackend(execution_authorized=True, transport=stub)
        job = sample_job()
        handle = backend.submit(job)
        self.assertTrue(handle.simulated)  # injected test transport, never Docker
        self.assertEqual(handle, backend.submit(job.model_copy(deep=True)))
        final = backend.wait(handle, timeout=2)
        self.assertEqual(final.status, JobStatus.SUCCEEDED)
        self.assertEqual(len(stub.calls), 1)
        contents = backend.collect(handle)
        self.assertEqual(contents.files["metrics.json"]["mean"], 3.0)
        self.assertTrue(contents.simulated)
        self.assertIn("TEST DOUBLE", contents.warning)

    def test_changed_spec_rejected(self):
        stub = StubTransport()
        backend = LocalDockerBackend(execution_authorized=True, transport=stub)
        job = sample_job()
        handle = backend.submit(job)
        changed = job.model_copy(deep=True)
        changed.seed = 54
        with self.assertRaisesRegex(ValueError, "different specification"):
            backend.submit(changed)
        backend.wait(handle, timeout=2)
        self.assertEqual(len(stub.calls), 1)

    def test_handle_tamper_and_collect_before_completion_rejected(self):
        stub = BlockingTransport()
        backend = LocalDockerBackend(execution_authorized=True, transport=stub)
        handle = backend.submit(sample_job())
        self.assertTrue(stub.started.wait(timeout=2))
        with self.assertRaises(RuntimeError):
            backend.collect(handle)
        altered = handle.model_copy(update={"remote_id": "local:invalid"})
        with self.assertRaises(KeyError):
            backend.status(altered)
        backend.cancel(handle)
        backend.wait(handle, timeout=2)

    def test_cancel_running_job(self):
        stub = BlockingTransport()
        backend = LocalDockerBackend(execution_authorized=True, transport=stub)
        handle = backend.submit(sample_job())
        self.assertTrue(stub.started.wait(timeout=2))
        backend.cancel(handle)
        status = backend.wait(handle, timeout=2)
        self.assertEqual(status.status, JobStatus.CANCELLED)
        with self.assertRaises(RuntimeError):
            backend.collect(handle)

    def test_failed_runner_cannot_collect(self):
        backend = LocalDockerBackend(execution_authorized=True, transport=StubTransport("error"))
        handle = backend.submit(sample_job())
        self.assertEqual(backend.wait(handle, timeout=2).status, JobStatus.FAILED)
        with self.assertRaises(RuntimeError):
            backend.collect(handle)

    def test_output_digest_is_checked(self):
        job = sample_job()
        self.assertEqual(decode_artifacts(job, good_output(job))["metrics.json"]["mean"], 3.0)
        corrupt = json.loads(good_output(job))
        corrupt["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            decode_artifacts(job, json.dumps(corrupt))

    def test_output_paths_and_shapes_are_checked(self):
        job = sample_job()
        corrupt = json.loads(good_output(job))
        corrupt["files"]["../escape.json"] = {}
        with self.assertRaisesRegex(ValueError, "unexpected"):
            decode_artifacts(job, json.dumps(corrupt))
        corrupt = json.loads(good_output(job))
        corrupt["files"]["metrics.json"]["mean"] = "3"
        with self.assertRaisesRegex(ValueError, "metric values"):
            decode_artifacts(job, json.dumps(corrupt))

    def test_opaque_python_not_allowed_in_job_payload(self):
        job = sample_job().model_dump()
        job["workload"]["parameters"] = {"values": [1], "python_source": "print('oops')"}
        candidate = ExperimentJob.model_validate(job)
        with self.assertRaises(ValueError):
            render_source(candidate)

    def test_simulated_status_not_marked_real(self):
        backend = LocalDockerBackend(execution_authorized=True, transport=StubTransport())
        handle = backend.submit(sample_job())
        self.assertEqual(handle.backend, "docker-local")
        self.assertTrue(handle.simulated)
        backend.wait(handle, timeout=2)
        self.assertTrue(backend.collect(handle).simulated)


if __name__ == "__main__":
    unittest.main()
