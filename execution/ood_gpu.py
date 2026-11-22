"""Approved SSH transport for the fixed four-trial OOD GPU calibration."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import shlex
import subprocess
from typing import ClassVar, Literal

from pydantic import Field, model_validator

from execution.backends.base import ExecutionBackend
from execution.campaign import canonical
from execution.contracts import ExperimentJob, JobArtifacts, JobHandle, JobSnapshot, JobStatus, ResourceRequest, StrictModel, WorkloadSpec
from execution.job_manager import DurableJobManager
from execution.ood_gpu_worker import (
    MAX_INPUT_TOKENS, MAX_NEW_TOKENS, MODEL, REVISION, TRIAL_IDS, decode_message, worker_sha256,
)
from execution.ood_replay import CASES

WORKLOAD = "ood_gpu"
WALLTIME_MINUTES = 10
SHA256 = r"^[a-f0-9]{64}$"
REMOTE_PATH = r"^/[-A-Za-z0-9_./]+$"


def _digest(document):
    return hashlib.sha256(canonical(document).encode("utf-8")).hexdigest()


class OODGPUConfig(StrictModel):
    worker_filename: ClassVar[str] = "ood_gpu_worker.py"
    ssh_host: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    python_path: str = Field(pattern=REMOTE_PATH, max_length=300)
    worker_path: str = Field(pattern=REMOTE_PATH, max_length=300)
    snapshot_path: str = Field(pattern=REMOTE_PATH, max_length=300)
    gpu_index: int = Field(ge=0, le=15)

    @model_validator(mode="after")
    def validate_paths(self):
        if any(".." in PurePosixPath(path).parts for path in (self.python_path, self.worker_path, self.snapshot_path)):
            raise ValueError("GPU paths must be explicit absolute paths")
        if PurePosixPath(self.worker_path).name != self.worker_filename:
            raise ValueError(f"The approved Python worker must be {self.worker_filename}")
        if PurePosixPath(self.snapshot_path).name != REVISION:
            raise ValueError("Snapshot path must end in the approved model revision")
        return self


class OODGPUProtocol(StrictModel):
    schema_version: Literal["9.0-ood-gpu-1"] = "9.0-ood-gpu-1"
    workload: Literal["ood_gpu"] = WORKLOAD
    campaign_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,31}$")
    requested_question: str
    measured_question: Literal["Le modèle répond-il correctement aux quatre entrées de calibration ?"] = (
        "Le modèle répond-il correctement aux quatre entrées de calibration ?")
    scope: Literal["Calibration GPU sur deux tâches appariées et un modèle, à portée descriptive."] = (
        "Calibration GPU sur deux tâches appariées et un modèle, à portée descriptive.")
    mission_sha256: str = Field(pattern=SHA256)
    literature_sha256: str = Field(pattern=SHA256)
    task_sha256: str = Field(default_factory=lambda: _digest(CASES), pattern=SHA256)
    worker_sha256: str = Field(default_factory=worker_sha256, pattern=SHA256)
    config: OODGPUConfig
    simulated: Literal[False] = False
    scientific_evidence: Literal[False] = False

    def fingerprint(self):
        return _digest(self.model_dump(mode="json"))

    def job(self):
        trials = []
        for trial_id in TRIAL_IDS:
            task, distribution = trial_id.split("-")
            trials.append({"id": trial_id, "prompt": CASES[task][distribution]})
        return ExperimentJob(
            experiment_id=self.campaign_id, job_id=f"{self.campaign_id}-gpu",
            workload=WorkloadSpec(name="ood.gpu_calibration", parameters={
                "model": MODEL, "revision": REVISION, "dtype": "bfloat16", "trials": trials,
                "max_input_tokens": MAX_INPUT_TOKENS, "max_new_tokens": MAX_NEW_TOKENS,
            }),
            seed=0, resources=ResourceRequest(cpus=2, memory_mb=16_384, gpus=1, walltime_minutes=WALLTIME_MINUTES),
            expected_artifacts=["responses.json"], source_design_sha256=self.fingerprint(),
        )


class GPUResponse(StrictModel):
    id: str
    text: str = Field(max_length=1024)
    input_tokens: int = Field(ge=1, le=MAX_INPUT_TOKENS)
    output_tokens: int = Field(ge=1, le=MAX_NEW_TOKENS)
    latency_ms: int = Field(ge=0)


class GPURuntime(StrictModel):
    gpu_index: int = Field(ge=0, le=15)
    gpu_name: str = Field(min_length=1, max_length=100)
    cuda_version: str = Field(min_length=1, max_length=100)
    torch_version: str = Field(min_length=1, max_length=100)
    transformers_version: str = Field(min_length=1, max_length=100)
    elapsed_ms: int = Field(ge=0, le=WALLTIME_MINUTES * 60_000)
    peak_vram_bytes: int = Field(gt=0)


class GPUReceipt(StrictModel):
    job_id: str
    job_sha256: str = Field(pattern=SHA256)
    worker_sha256: str = Field(pattern=SHA256)
    model: Literal["Qwen/Qwen3-4B-Instruct-2507"]
    revision: Literal["cdbee75f17c01a7cc42f958dc650907174af0554"]
    dtype: Literal["bfloat16"]
    responses: list[GPUResponse] = Field(min_length=len(TRIAL_IDS), max_length=len(TRIAL_IDS))
    runtime: GPURuntime


def validate_receipt(protocol, job, document) -> GPUReceipt:
    receipt = GPUReceipt.model_validate(document)
    if (receipt.job_id != job.job_id or receipt.job_sha256 != job.fingerprint()
            or receipt.worker_sha256 != protocol.worker_sha256 or receipt.runtime.gpu_index != protocol.config.gpu_index
            or tuple(item.id for item in receipt.responses) != TRIAL_IDS
            or sum(item.latency_ms for item in receipt.responses) > receipt.runtime.elapsed_ms):
        raise ValueError("GPU receipt differs from the approved job")
    return receipt


def ssh_command(config: OODGPUConfig, *, minutes=WALLTIME_MINUTES) -> list[str]:
    remote = shlex.join(["/usr/bin/timeout", "--kill-after=10s", f"{minutes * 60}s",
                         config.python_path, "-I", "-B", config.worker_path])
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "UpdateHostKeys=no",
            "-o", "ClearAllForwardings=yes", "-o", "ForwardAgent=no", "-o", "ConnectTimeout=10",
            "-o", "ConnectionAttempts=1", config.ssh_host, remote]


def ssh_receipt(config, request, *, minutes=WALLTIME_MINUTES, decoder=decode_message):
    completed = subprocess.run(
        ssh_command(config, minutes=minutes), input=canonical(request).encode("utf-8"), capture_output=True,
        timeout=minutes * 60 + 30, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"SSH GPU run failed with exit code {completed.returncode}")
    return decoder(completed.stdout)


class GPUReceiptBackend(ExecutionBackend):
    """Synchronous SSH backends expose only locally validated receipts."""

    def __init__(self):
        self.artifact = None

    def status(self, handle):
        self.collect(handle)
        return JobSnapshot(handle=handle, status=JobStatus.SUCCEEDED, message="GPU receipt validated")

    def cancel(self, handle):
        return self.status(handle)

    def collect(self, handle):
        if self.artifact is None or handle != self.artifact.handle:
            raise KeyError("GPU receipt unavailable. Reconcile before any restart")
        return self.artifact


class OODGPUBackend(GPUReceiptBackend):
    def __init__(self, protocol, *, execution_authorized=False):
        super().__init__()
        self.protocol = protocol
        self.authorized = execution_authorized

    def submit(self, job):
        if not self.authorized:
            raise PermissionError("GPU execution requires explicit authorization")
        if job.fingerprint() != self.protocol.job().fingerprint() or self.protocol.worker_sha256 != worker_sha256():
            raise ValueError("GPU job or worker source differs from the approved specification")
        if self.artifact is not None:
            return self.artifact.handle
        config = self.protocol.config
        raw = ssh_receipt(config, {
            "job_id": job.job_id, "job_sha256": job.fingerprint(), "model": MODEL, "revision": REVISION,
            "worker_sha256": self.protocol.worker_sha256, "snapshot": config.snapshot_path,
            "gpu_index": config.gpu_index, "trials": job.workload.parameters["trials"],
        })
        receipt = validate_receipt(self.protocol, job, raw)
        handle = JobHandle(backend="ood-gpu-ssh", remote_id=f"ssh:{job.job_id}", job_id=job.job_id,
                           spec_sha256=job.fingerprint(), simulated=False)
        self.artifact = JobArtifacts(handle=handle, files={"responses.json": receipt.model_dump(mode="json")},
                                     simulated=False, warning="Descriptive calibration only")
        return handle


def run_ood_gpu(protocol: OODGPUProtocol, ledger: Path) -> None:
    manager = DurableJobManager(OODGPUBackend(protocol, execution_authorized=True), ledger)
    job = protocol.job()
    record = manager.submit(job)
    if record.phase == "submitted":
        record = manager.refresh(job.job_id)
    if record.phase != "succeeded":
        raise RuntimeError(f"GPU job has no complete receipt ({record.phase})")
