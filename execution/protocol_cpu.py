"""Execute a verified expression plan through the durable local job interface."""
from __future__ import annotations

from time import perf_counter_ns

from execution.backends.base import ExecutionBackend
from execution.contracts import ExperimentJob, JobArtifacts, JobHandle, JobSnapshot, JobStatus
from execution.job_manager import DurableJobManager
from experiments.protocol_program import RestrictedProgram

CAPABILITY = "cpu.integer_exact.v1"
WORKLOAD = "protocol.integer_exact"
BACKEND = "protocol-cpu"
ARTIFACT = "predictions.json"


def predictions(job: ExperimentJob, *, clock=perf_counter_ns):
    params = job.workload.parameters
    levels = params["levels"]
    program = RestrictedProgram(params["source"], features=params["cases"][0]["inputs"], levels=levels)
    rows = []
    for seed in params["seeds"]:
        for case in params["cases"]:
            start = clock()
            try:
                value = program.predict(case["inputs"], levels, seed)
                outcome = "answered"
            except ValueError:
                value, outcome = None, "invalid_output"
            rows.append({"case_id": case["id"], "seed": seed, "value": value, "outcome": outcome,
                         "elapsed_ns": clock() - start, "model_tokens": 0})
    return rows


class ProtocolCPUBackend(ExecutionBackend):
    def __init__(self, jobs):
        self._approved = {job.job_id: job.fingerprint() for job in jobs}
        self._artifacts = {}

    def submit(self, job):
        digest = job.fingerprint()
        if job.workload.name != WORKLOAD or self._approved.get(job.job_id) != digest:
            raise PermissionError("Job differs from the authorized CPU plan")
        artifact = self._artifacts.get(job.job_id)
        if artifact is not None:
            return artifact.handle
        handle = JobHandle(backend=BACKEND, remote_id=f"local:{job.job_id}", job_id=job.job_id,
                           spec_sha256=digest, simulated=False)
        self._artifacts[job.job_id] = JobArtifacts(
            handle=handle, files={ARTIFACT: predictions(job)}, simulated=False,
            warning="Descriptive pilot with an independently supplied evaluation set",
        )
        return handle

    def collect(self, handle):
        artifact = self._artifacts.get(handle.job_id)
        if artifact is None or artifact.handle != handle:
            raise KeyError("The original CPU receipt is unavailable")
        return artifact

    def status(self, handle):
        self.collect(handle)
        return JobSnapshot(handle=handle, status=JobStatus.SUCCEEDED)

    def cancel(self, handle):
        return self.status(handle)


def run_program_plan(jobs, ledger, backend):
    manager = DurableJobManager(backend, ledger)
    for job in jobs:
        record = manager.submit(job)
        if record.phase == "submitted":
            record = manager.refresh(job.job_id)
        if record.phase != "succeeded":
            raise RuntimeError(f"Receipt unavailable for {job.job_id} ({record.phase})")
