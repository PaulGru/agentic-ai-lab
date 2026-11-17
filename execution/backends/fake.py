"""Deterministic fake backend. NEVER executes commands, LLMs or Slurm jobs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from execution.backends.base import ExecutionBackend
from execution.contracts import (
    ExperimentJob, JobArtifacts, JobHandle, JobSnapshot, JobStatus,
)


@dataclass
class _Record:
    handle: JobHandle
    job: ExperimentJob
    status: JobStatus = JobStatus.SUBMITTED
    files: dict[str, Any] = field(default_factory=dict)


class FakeBackend(ExecutionBackend):
    """In-memory test double with deterministic lifecycle and idempotent submit.

    This is not durable and is not an actual experiment executor.
    """

    def __init__(self) -> None:
        self._records: dict[str, _Record] = {}

    def submit(self, job: ExperimentJob) -> JobHandle:
        digest = job.fingerprint()
        existing = self._records.get(job.job_id)
        if existing is not None:
            if existing.handle.spec_sha256 != digest:
                raise ValueError("job_id already exists with different specification")
            return existing.handle
        handle = JobHandle(
            backend="fake", remote_id=f"fake:{job.job_id}",
            job_id=job.job_id, spec_sha256=digest, simulated=True,
        )
        self._records[job.job_id] = _Record(handle=handle, job=job)
        return handle

    def _get(self, handle: JobHandle) -> _Record:
        record = self._records.get(handle.job_id)
        if record is None or record.handle != handle:
            raise KeyError("Unknown or mismatched job handle")
        return record

    def status(self, handle: JobHandle) -> JobSnapshot:
        record = self._get(handle)
        return JobSnapshot(
            handle=handle, status=record.status,
            message="SIMULATED state; no job executed",
        )

    def cancel(self, handle: JobHandle) -> JobSnapshot:
        record = self._get(handle)
        if record.status in {JobStatus.SUBMITTED, JobStatus.RUNNING}:
            record.status = JobStatus.CANCELLED
        return self.status(handle)

    def mark_running(self, handle: JobHandle) -> JobSnapshot:
        """Test-only operation emulating an external scheduler starting a job."""
        record = self._get(handle)
        if record.status != JobStatus.SUBMITTED:
            raise ValueError("Only submitted jobs may start")
        record.status = JobStatus.RUNNING
        return self.status(handle)

    def finish(
        self, handle: JobHandle, *, success: bool,
        files: dict[str, Any] | None = None,
    ) -> JobSnapshot:
        """Test-only operation emulating an external scheduler finishing."""
        record = self._get(handle)
        if record.status != JobStatus.RUNNING:
            raise ValueError("Only running jobs may finish")
        requested = files or {}
        if set(requested) - set(record.job.expected_artifacts):
            raise ValueError("Unexpected output artifact")
        record.files = dict(requested)
        record.status = JobStatus.SUCCEEDED if success else JobStatus.FAILED
        return self.status(handle)

    def collect(self, handle: JobHandle) -> JobArtifacts:
        record = self._get(handle)
        if record.status != JobStatus.SUCCEEDED:
            raise RuntimeError("Cannot collect unless job succeeded")
        return JobArtifacts(
            handle=handle, files=dict(record.files), simulated=True,
            warning="SIMULATED OUTPUTS: not experimental or scientific results",
        )
