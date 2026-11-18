"""V8.4.4 — SQLite job ledger with a write-ahead submission intent.

This ledger tracks jobs and stores receipts; it does NOT provide exactly-once
external side effects. Ambiguous submission/restart becomes NEEDS_RECONCILIATION,
never an automatic resubmit. No secrets or generated code belong in the job.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from execution.backends.base import ExecutionBackend
from execution.contracts import ExperimentJob, JobArtifacts, JobHandle, JobStatus


ACTIVE_PHASES = {"submitted", "running"}
TERMINAL_PHASES = {"succeeded", "failed", "cancelled"}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    spec_sha256: str
    backend: str
    phase: str
    message: str
    handle: JobHandle | None
    artifacts: JobArtifacts | None
    artifacts_sha256: str | None

    def summary(self) -> dict:
        return {
            "job_id": self.job_id, "spec_sha256": self.spec_sha256,
            "backend": self.backend, "phase": self.phase, "message": self.message,
            "handle": self.handle.model_dump(mode="json") if self.handle else None,
            "artifacts_sha256": self.artifacts_sha256,
            "simulated": self.artifacts.simulated if self.artifacts else (
                self.handle.simulated if self.handle else None),
            "artifact_names": sorted(self.artifacts.files) if self.artifacts else [],
        }


class DurableJobManager:
    def __init__(self, backend: ExecutionBackend | None = None,
                 path: str | Path = "runs_control/jobs.sqlite3", *, read_only: bool = False):
        self.backend = backend
        self.path = Path(path)
        self.read_only = read_only
        if read_only:
            if not self.path.is_file():
                raise FileNotFoundError("Existing job ledger required for read-only access")
            with self._connect() as db:
                db.execute("SELECT job_id FROM jobs LIMIT 0")
            return
        if backend is None:
            raise ValueError("A writable job manager requires an execution backend")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                spec_sha256 TEXT NOT NULL,
                job_json TEXT NOT NULL,
                backend TEXT NOT NULL,
                phase TEXT NOT NULL,
                message TEXT NOT NULL,
                handle_json TEXT,
                artifacts_json TEXT,
                artifacts_sha256 TEXT
            )""")

    @contextmanager
    def _connect(self):
        if self.read_only:
            db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
            db.execute("PRAGMA query_only=ON")
        else:
            db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, job_id: str) -> JobRecord | None:
        with self._connect() as db:
            row = db.execute("""SELECT job_id, spec_sha256, backend, phase,
                message, handle_json, artifacts_json, artifacts_sha256
                FROM jobs WHERE job_id=?""", (job_id,)).fetchone()
        if row is None:
            return None
        handle = JobHandle.model_validate_json(row[5]) if row[5] else None
        artifacts = JobArtifacts.model_validate_json(row[6]) if row[6] else None
        if artifacts:
            actual = hashlib.sha256(_canonical(artifacts.model_dump(mode="json")).encode()).hexdigest()
            if actual != row[7]:
                raise ValueError("Persisted artifact digest mismatch")
        return JobRecord(job_id=row[0], spec_sha256=row[1], backend=row[2],
                         phase=row[3], message=row[4], handle=handle,
                         artifacts=artifacts, artifacts_sha256=row[7])

    def submit(self, job: ExperimentJob) -> JobRecord:
        self._require_write()
        digest = job.fingerprint()
        exists = False
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT spec_sha256, backend FROM jobs WHERE job_id=?", (job.job_id,)).fetchone()
            if row:
                if row[0] != digest or row[1] != type(self.backend).__name__:
                    raise ValueError("Existing job_id has a different specification/backend")
                exists = True
            else:
                db.execute("""INSERT INTO jobs (job_id, spec_sha256, job_json, backend, phase, message)
                    VALUES (?, ?, ?, ?, 'submission_reserved', 'Durable intent recorded; submission unconfirmed')""",
                    (job.job_id, digest, _canonical(job.model_dump(mode="json")), type(self.backend).__name__))
        if exists:
            # Critical: NEVER submit again, including a reserved job with no handle.
            return self.get(job.job_id)
        try:
            handle = self.backend.submit(job)
            if handle.spec_sha256 != digest or handle.job_id != job.job_id:
                raise ValueError("Backend returned a mismatched handle")
        except Exception as exc:
            # A timeout/transport exception could mean that the remote job WAS submitted.
            self._update(job.job_id, "needs_reconciliation",
                         f"Submission outcome ambiguous ({type(exc).__name__}); manual review needed")
            return self.get(job.job_id)
        self._update(job.job_id, "submitted", "Submission acknowledged",
                     handle=handle)
        return self.get(job.job_id)

    def _update(self, job_id: str, phase: str, message: str,
                *, handle: JobHandle | None = None,
                artifacts: JobArtifacts | None = None):
        self._require_write()
        updates = ["phase=?", "message=?"]
        values: list[Any] = [phase, message[:250]]
        if handle is not None:
            updates.append("handle_json=?")
            values.append(_canonical(handle.model_dump(mode="json")))
        if artifacts is not None:
            encoded = _canonical(artifacts.model_dump(mode="json"))
            updates += ["artifacts_json=?", "artifacts_sha256=?"]
            values += [encoded, hashlib.sha256(encoded.encode()).hexdigest()]
        with self._connect() as db:
            cursor = db.execute(f"UPDATE jobs SET {', '.join(updates)} WHERE job_id=?",
                                (*values, job_id))
            if cursor.rowcount != 1:
                raise KeyError("Unknown job")

    def refresh(self, job_id: str) -> JobRecord:
        self._require_write()
        record = self.get(job_id)
        if record is None:
            raise KeyError("Unknown job")
        if record.backend != type(self.backend).__name__:
            raise ValueError("Ledger backend mismatch; use the original execution backend")
        if record.phase in TERMINAL_PHASES or record.phase == "needs_reconciliation":
            return record
        if record.handle is None:
            # Crash after intent, before persisting handle: ambiguous by design.
            self._update(job_id, "needs_reconciliation", "Submission handle absent; do not resubmit")
            return self.get(job_id)
        try:
            snapshot = self.backend.status(record.handle)
            if snapshot.handle != record.handle:
                raise ValueError("Backend returned a mismatched snapshot")
            phase = snapshot.status.value
            if phase == "succeeded":
                artifacts = self.backend.collect(record.handle)
                if artifacts.handle != record.handle or artifacts.simulated != record.handle.simulated:
                    raise ValueError("Artifact provenance mismatch")
                # Genuine outputs must contain every declared artifact.
                if not artifacts.simulated:
                    with self._connect() as db:
                        row = db.execute("SELECT job_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
                    job = ExperimentJob.model_validate_json(row[0])
                    if set(artifacts.files) != set(job.expected_artifacts):
                        raise ValueError("Incomplete real experiment artifacts")
                self._update(job_id, phase, snapshot.message, artifacts=artifacts)
            else:
                self._update(job_id, phase, snapshot.message)
        except Exception as exc:
            self._update(job_id, "needs_reconciliation",
                         f"Backend outcome unavailable ({type(exc).__name__}); review required")
        return self.get(job_id)

    def cancel(self, job_id: str) -> JobRecord:
        self._require_write()
        record = self.get(job_id)
        if record is None:
            raise KeyError("Unknown job")
        if record.backend != type(self.backend).__name__:
            raise ValueError("Ledger backend mismatch; use the original execution backend")
        if record.phase in TERMINAL_PHASES or record.phase == "needs_reconciliation":
            return record
        if record.handle is None:
            self._update(job_id, "needs_reconciliation", "Cancel requires external job identification")
            return self.get(job_id)
        try:
            snapshot = self.backend.cancel(record.handle)
            if snapshot.handle != record.handle:
                raise ValueError("Mismatched cancellation handle")
            self._update(job_id, snapshot.status.value, snapshot.message)
        except Exception as exc:
            self._update(job_id, "needs_reconciliation",
                         f"Cancellation outcome ambiguous ({type(exc).__name__})")
        return self.get(job_id)

    def collect(self, job_id: str) -> JobArtifacts:
        record = self.get(job_id)
        if record is None or record.phase != "succeeded" or record.artifacts is None:
            raise RuntimeError("No validated, persisted artifacts available")
        return record.artifacts

    def _require_write(self) -> None:
        if self.read_only:
            raise PermissionError("Read-only job ledger cannot submit, refresh, cancel or update jobs")
