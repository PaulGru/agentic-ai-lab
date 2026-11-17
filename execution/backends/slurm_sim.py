"""V8.4.3 — Slurm contract adapter with a durable OFFLINE simulator.

No SSH, sbatch or cluster commands are executed. The script is a *preview*
constructed from the allowlisted synthetic workload only.
"""
from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from execution.backends.base import ExecutionBackend
from execution.contracts import ExperimentJob, JobHandle, JobSnapshot, JobStatus, JobArtifacts
from execution.workloads import render_source


SCHEDULER_STATES = {
    "PENDING": JobStatus.SUBMITTED,
    "CONFIGURING": JobStatus.SUBMITTED,
    "RUNNING": JobStatus.RUNNING,
    "COMPLETING": JobStatus.RUNNING,
    "SUSPENDED": JobStatus.RUNNING,
    "RESIZING": JobStatus.RUNNING,
    "REQUEUED": JobStatus.SUBMITTED,
    "COMPLETED": JobStatus.SUCCEEDED,
    "FAILED": JobStatus.FAILED,
    "TIMEOUT": JobStatus.FAILED,
    "OUT_OF_MEMORY": JobStatus.FAILED,
    "NODE_FAIL": JobStatus.FAILED,
    "PREEMPTED": JobStatus.FAILED,
    "CANCELLED": JobStatus.CANCELLED,
}


def map_slurm_state(raw: str) -> JobStatus:
    """Slurm sometimes returns states like 'CANCELLED by 1000' or 'COMPLETED+'."""
    if not isinstance(raw, str):
        raise ValueError("Slurm state must be text")
    key = raw.strip().upper().split()[0].rstrip("+") if raw.strip() else ""
    if key not in SCHEDULER_STATES:
        raise ValueError(f"Unknown Slurm state: {key!r}; do not assume success")
    return SCHEDULER_STATES[key]


def render_sbatch_preview(job: ExperimentJob) -> str:
    """Render a restricted *preview*. Never feed user code or a shell command to Slurm."""
    source = render_source(job)  # validates allowlisted workload & small resource limits
    r = job.resources
    return (
        "#!/bin/sh\n"
        f"#SBATCH --job-name={job.job_id}\n"
        f"#SBATCH --cpus-per-task={r.cpus}\n"
        f"#SBATCH --mem={r.memory_mb}M\n"
        f"#SBATCH --time=00:{r.walltime_minutes:02}:00\n"
        "#SBATCH --no-requeue\n"
        "set -eu\n"
        "umask 077\n"
        "# PREVIEW ONLY — no cluster submission in V8.4.3\n"
        "python -I -B - <<'AGENTIC_PYTHON' > result_envelope.json\n"
        + source
        + "\nAGENTIC_PYTHON\n"
    )


class SlurmSimBackend(ExecutionBackend):
    """Durable state simulator, deliberately NOT a real Slurm backend.

    Runs are advanced only by explicit calls to advance(); simulated artifacts
    are always empty, so they cannot be mistaken for measured metrics.
    """

    def __init__(self, path: str | Path = "runs_control/slurm_sim.sqlite3"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS simulated_jobs (
                remote_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT UNIQUE NOT NULL,
                digest TEXT NOT NULL,
                script_sha256 TEXT NOT NULL,
                slurm_state TEXT NOT NULL
            )""")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def submit(self, job: ExperimentJob) -> JobHandle:
        script = render_sbatch_preview(job)
        digest = job.fingerprint()
        script_digest = hashlib.sha256(script.encode("utf-8")).hexdigest()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("""INSERT OR IGNORE INTO simulated_jobs
                (job_id, digest, script_sha256, slurm_state)
                VALUES (?, ?, ?, 'PENDING')""", (job.job_id, digest, script_digest))
            row = db.execute("SELECT remote_id, digest, script_sha256 FROM simulated_jobs WHERE job_id=?", (job.job_id,)).fetchone()
            if row[1] != digest or row[2] != script_digest:
                raise ValueError("job_id exists with a different Slurm simulation contract")
        return JobHandle(backend="slurm-sim", remote_id=str(row[0]), job_id=job.job_id,
                         spec_sha256=digest, simulated=True)

    def _read(self, handle: JobHandle):
        if handle.backend != "slurm-sim" or not handle.simulated:
            raise ValueError("Not a Slurm simulator handle")
        with self._connect() as db:
            row = db.execute("""SELECT job_id, digest, slurm_state FROM simulated_jobs
                WHERE remote_id=?""", (handle.remote_id,)).fetchone()
        if not row or row[0] != handle.job_id or row[1] != handle.spec_sha256:
            raise KeyError("Unknown or mismatched Slurm simulation handle")
        return row[2]

    def status(self, handle: JobHandle) -> JobSnapshot:
        raw = self._read(handle)
        return JobSnapshot(handle=handle, status=map_slurm_state(raw),
                           message=f"SIMULATED Slurm state: {raw}")

    def advance(self, handle: JobHandle, *, fail: bool = False) -> JobSnapshot:
        """Test-only: PENDING -> RUNNING -> COMPLETED/FAILED."""
        raw = self._read(handle)
        if raw not in ("PENDING", "RUNNING"):
            raise ValueError("Only active simulated jobs can advance")
        next_state = "RUNNING" if raw == "PENDING" else ("FAILED" if fail else "COMPLETED")
        with self._connect() as db:
            cursor = db.execute("""UPDATE simulated_jobs SET slurm_state=?
                WHERE remote_id=? AND slurm_state=?""", (next_state, handle.remote_id, raw))
            if cursor.rowcount != 1:
                raise RuntimeError("Concurrent simulator state change")
        return self.status(handle)

    def cancel(self, handle: JobHandle) -> JobSnapshot:
        self._read(handle)
        with self._connect() as db:
            db.execute("""UPDATE simulated_jobs SET slurm_state='CANCELLED'
                WHERE remote_id=? AND slurm_state IN ('PENDING','RUNNING')""", (handle.remote_id,))
        return self.status(handle)

    def collect(self, handle: JobHandle) -> JobArtifacts:
        if self.status(handle).status != JobStatus.SUCCEEDED:
            raise RuntimeError("Cannot collect unfinished simulator job")
        return JobArtifacts(handle=handle, files={}, simulated=True,
            warning="SIMULATED SLURM: no scientific computation or measurements occurred")
