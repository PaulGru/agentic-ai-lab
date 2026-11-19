"""V8.4.6 — deterministic, provider-independent *engineering* pilot campaign.

A pilot protocol becomes allowlisted ExperimentJob instances. This is not an
LLM benchmark or evidence for a research hypothesis. No arbitrary code or
shell command is accepted from the protocol.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from execution.contracts import ExperimentJob, ResourceRequest, StrictModel, WorkloadSpec
from execution.job_manager import ACTIVE_PHASES, TERMINAL_PHASES, DurableJobManager, JobRecord
from execution.workloads import render_source


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":"))


class PilotRun(StrictModel):
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,15}$")
    values: list[int | float] = Field(min_length=1, max_length=100)
    seed: int = Field(ge=0, le=2**32 - 1)

    @field_validator("values")
    @classmethod
    def finite_values(cls, values: list[int | float]) -> list[int | float]:
        if any(type(v) not in (int, float) or not math.isfinite(v)
               or abs(v) > 1_000_000 for v in values):
            raise ValueError("values must be finite numbers with |value| <= 1000000")
        return values


class PilotProtocol(StrictModel):
    schema_version: Literal["8.4.6"] = "8.4.6"
    campaign_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
    description: str = Field(min_length=8, max_length=300)
    resources: ResourceRequest = Field(default_factory=lambda: ResourceRequest(
        cpus=1, memory_mb=256, gpus=0, walltime_minutes=2))
    runs: list[PilotRun] = Field(min_length=2, max_length=4)

    @model_validator(mode="after")
    def unique_runs(self):
        names = [run.run_id for run in self.runs]
        if len(set(names)) != len(names):
            raise ValueError("run_id must be unique within the campaign")
        # Fail during planning, rather than after HITL/partial submission.
        if self.resources.gpus != 0 or self.resources.cpus > 2 or self.resources.memory_mb > 512 or self.resources.walltime_minutes > 2:
            raise ValueError("Pilot resources exceed the V8.4.2 local Docker allowlist")
        return self

    def fingerprint(self) -> str:
        return hashlib.sha256(canonical(self.model_dump(mode="json")).encode("utf-8")).hexdigest()

    def jobs(self) -> list[ExperimentJob]:
        jobs = []
        for run in self.runs:
            job = ExperimentJob(
                schema_version="8.4.1",
                experiment_id=self.campaign_id,
                job_id=f"{self.campaign_id}-{run.run_id}",
                workload=WorkloadSpec(name="synthetic.mean", parameters={"values": run.values}),
                seed=run.seed,
                resources=self.resources.model_copy(deep=True),
                expected_artifacts=["metrics.json", "run_metadata.json"],
                source_design_sha256=self.fingerprint(),
            )
            render_source(job)  # exact existing V8.4.2 execution allowlist
            jobs.append(job)
        return jobs


class CampaignBlocked(RuntimeError):
    def __init__(self, record: JobRecord, *, timed_out: bool = False):
        self.uncertain = record.phase not in TERMINAL_PHASES
        status = f"timeout, {record.phase}" if timed_out else f"{record.phase}; {record.message}"
        super().__init__(f"{record.job_id}: {status}")


def run_campaign(protocol: PilotProtocol, manager: DurableJobManager, *, poll_seconds: float = 0.1) -> None:
    """Submit an approved campaign and persist all successful receipts before returning."""
    for job in protocol.jobs():
        record = manager.submit(job)
        if record.phase == "succeeded":
            continue
        if record.phase not in ACTIVE_PHASES:
            raise CampaignBlocked(record)
        deadline = time.monotonic() + job.resources.walltime_minutes * 60 + 15
        try:
            while time.monotonic() < deadline:
                record = manager.refresh(job.job_id)
                if record.phase not in ACTIVE_PHASES:
                    break
                time.sleep(poll_seconds)
            else:
                record = manager.cancel(job.job_id)
                if record.handle is not None:
                    manager.backend.wait(record.handle, timeout=3)
                raise CampaignBlocked(manager.refresh(job.job_id), timed_out=True)
        except KeyboardInterrupt:
            manager.cancel(job.job_id)
            if record.handle is not None:
                manager.backend.wait(record.handle, timeout=3)
            raise
        if record.phase != "succeeded":
            raise CampaignBlocked(record)


def build_report(protocol: PilotProtocol, manager: DurableJobManager) -> dict[str, Any]:
    """Reconstruct a small, deterministic analysis from persisted validated data.

    It intentionally avoids a model-based interpretation or statistical claim.
    """
    results = []
    for job, run in zip(protocol.jobs(), protocol.runs):
        record = manager.get(job.job_id)
        if record is None or record.phase != "succeeded" or record.artifacts is None:
            raise RuntimeError(f"Job {job.job_id} has no durable successful outputs")
        if (record.spec_sha256 != job.fingerprint() or record.backend != "LocalDockerBackend"
                or record.artifacts.handle != record.handle or record.handle is None):
            raise ValueError("Job specification/backend/handle provenance mismatch")
        artifacts = manager.collect(job.job_id)  # checksum checked by ledger
        if set(artifacts.files) != set(job.expected_artifacts):
            raise ValueError("Incomplete pilot output artifacts")
        metrics = artifacts.files["metrics.json"]
        metadata = artifacts.files["run_metadata.json"]
        if not isinstance(metrics, dict) or not isinstance(metadata, dict):
            raise ValueError("Malformed metrics or metadata")
        if (metadata.get("job_id") != job.job_id or
                metadata.get("job_sha256") != job.fingerprint() or
                metadata.get("execution") != "docker_sandbox" or
                metadata.get("seed") != run.seed or
                metadata.get("workload") != "synthetic.mean"):
            raise ValueError("Provenance metadata mismatch")
        if (metrics.get("count") != len(run.values)
                or metrics.get("minimum") != min(run.values)
                or metrics.get("maximum") != max(run.values)
                or type(metrics.get("mean")) not in (int, float)
                or not math.isfinite(metrics["mean"])
                or not math.isclose(metrics["mean"], statistics.fmean(run.values),
                                    rel_tol=1e-12, abs_tol=1e-9)):
            raise ValueError("Metrics do not match the approved source data")
        results.append({
            "run_id": run.run_id,
            "job_id": job.job_id,
            "job_sha256": job.fingerprint(),
            "artifacts_sha256": record.artifacts_sha256,
            "simulated": artifacts.simulated,
            "metrics": metrics,
            "run_metadata": metadata,
        })
    means = [item["metrics"]["mean"] for item in results]
    return {
        "schema_version": "8.4.6",
        "kind": "engineering_pilot_report",
        "campaign_id": protocol.campaign_id,
        "protocol_sha256": protocol.fingerprint(),
        "backend": "docker-local",
        "simulated": any(item["simulated"] for item in results),
        "scientific_evidence": False,
        "message": "Synthetic integration demonstration only: no scientific inference",
        "results": results,
        "analysis": {
            "run_means": {item["run_id"]: item["metrics"]["mean"] for item in results},
            "last_minus_first_mean": means[-1] - means[0],
            "number_of_runs": len(means),
            "method": "deterministic comparison of synthetic arithmetic means",
        },
    }


def save_report(report: dict[str, Any], path: str | Path) -> Path:
    """Exclusive report export; never silently overwrite a different report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError("Existing report differs; choose another output path")
        return path
    # Exclusive creation protects against accidental replacement in another process.
    # If interrupted before close, the report can be rebuilt using `report`.
    with path.open("x", encoding="utf-8") as dest:
        dest.write(text)
        dest.flush()
        os.fsync(dest.fileno())
    return path
