"""V8.4.2 local backend. In-memory orchestration around isolated Docker runs.

No checkpoint durability yet: V8.4.4 will add an external job store and
reconciliation. Never assume exactly-once execution after process restarts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Any
from uuid import uuid4

from execution.backends.base import ExecutionBackend
from execution.backends.docker_transport import DockerOutcome, DockerTransport
from execution.contracts import (
    ExperimentJob, JobArtifacts, JobHandle, JobSnapshot, JobStatus,
)
from execution.workloads import decode_artifacts, render_source


@dataclass
class _LocalRecord:
    job: ExperimentJob
    handle: JobHandle
    status: JobStatus = JobStatus.SUBMITTED
    message: str = "Queued for Docker"
    files: dict[str, Any] = field(default_factory=dict)
    cancel_event: Event = field(default_factory=Event)
    thread: Thread | None = None
    duration_seconds: float | None = None


class LocalDockerBackend(ExecutionBackend):
    """One-process local executor; allowlist and user authorization required."""

    def __init__(self, *, execution_authorized: bool = False, transport=None):
        self._authorized = execution_authorized
        self._transport = transport if transport is not None else DockerTransport()
        self._lock = Lock()
        self._records: dict[str, _LocalRecord] = {}

    def submit(self, job: ExperimentJob) -> JobHandle:
        if not self._authorized:
            raise PermissionError("Local execution requires explicit operator authorization")
        # Validate workload, constraints and produce trusted code BEFORE creating a job.
        code = render_source(job)
        digest = job.fingerprint()
        with self._lock:
            existing = self._records.get(job.job_id)
            if existing is not None:
                if existing.handle.spec_sha256 != digest:
                    raise ValueError("job_id already exists with different specification")
                return existing.handle
            handle = JobHandle(
                backend="docker-local", remote_id=f"local:{uuid4().hex}",
                job_id=job.job_id, spec_sha256=digest,
                simulated=bool(self._transport.simulated),
            )
            record = _LocalRecord(job=job, handle=handle)
            self._records[job.job_id] = record
            worker = Thread(target=self._run_job, args=(record, code), daemon=True)
            record.thread = worker
            worker.start()
            return handle

    def _run_job(self, record: _LocalRecord, code: str) -> None:
        with self._lock:
            if record.cancel_event.is_set():
                record.status = JobStatus.CANCELLED
                record.message = "Cancelled before starting"
                return
            record.status = JobStatus.RUNNING
            record.message = "Docker workload running"
        try:
            outcome: DockerOutcome = self._transport.run(record.job, code, record.cancel_event)
            with self._lock:
                record.duration_seconds = outcome.duration_seconds
                if record.cancel_event.is_set() or outcome.status == "cancelled":
                    record.status = JobStatus.CANCELLED
                    record.message = "Cancelled (Docker container stopped)"
                elif outcome.status != "success":
                    record.status = JobStatus.FAILED
                    # Avoid exposing untrusted stdout/stderr to the API or logs.
                    record.message = f"Docker run failed: {outcome.status} (exit {outcome.exit_code})"
                else:
                    try:
                        record.files = decode_artifacts(record.job, outcome.stdout)
                    except ValueError as exc:
                        record.status = JobStatus.FAILED
                        record.message = f"Invalid workload results: {exc}"
                    else:
                        record.status = JobStatus.SUCCEEDED
                        record.message = f"Completed in {outcome.duration_seconds:.3f}s"
        except Exception as exc:
            with self._lock:
                record.status = JobStatus.FAILED
                # Do not serialize exception text: it may include host paths/secrets.
                record.message = f"Local transport failed ({type(exc).__name__})"

    def _record(self, handle: JobHandle) -> _LocalRecord:
        with self._lock:
            record = self._records.get(handle.job_id)
            if record is None or record.handle != handle:
                raise KeyError("Unknown or mismatched local job handle")
            return record

    def status(self, handle: JobHandle) -> JobSnapshot:
        with self._lock:
            record = self._records.get(handle.job_id)
            if record is None or record.handle != handle:
                raise KeyError("Unknown or mismatched local job handle")
            return JobSnapshot(handle=handle, status=record.status, message=record.message)

    def cancel(self, handle: JobHandle) -> JobSnapshot:
        with self._lock:
            record = self._records.get(handle.job_id)
            if record is None or record.handle != handle:
                raise KeyError("Unknown or mismatched local job handle")
            if record.status in {JobStatus.SUBMITTED, JobStatus.RUNNING}:
                record.cancel_event.set()
                record.message = "Cancellation requested"
        return self.status(handle)

    def collect(self, handle: JobHandle) -> JobArtifacts:
        with self._lock:
            record = self._records.get(handle.job_id)
            if record is None or record.handle != handle:
                raise KeyError("Unknown or mismatched local job handle")
            if record.status != JobStatus.SUCCEEDED:
                raise RuntimeError("Cannot collect unless job succeeded")
            return JobArtifacts(
                handle=handle, files=dict(record.files), simulated=handle.simulated,
                warning=("TEST DOUBLE: output was not computed by Docker"
                         if handle.simulated else
                         "REAL LOCAL EXECUTION: synthetic workload only; not research evidence"),
            )

    def wait(self, handle: JobHandle, timeout: float | None = None) -> JobSnapshot:
        """Convenience for the standalone CLI, not part of the portable backend API."""
        record = self._record(handle)
        if record.thread is not None:
            record.thread.join(timeout)
        return self.status(handle)
