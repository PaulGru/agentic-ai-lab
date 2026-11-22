"""Prepare a pinned local model and dispatch one authorized prompt batch."""
from typing import ClassVar

from pydantic import Field, model_validator

from analysis.contracts import HEX_SHA
from execution.contracts import JobArtifacts, JobHandle, StrictModel
from execution.ood_gpu import GPURuntime, GPUReceiptBackend, OODGPUConfig, ssh_receipt
from execution.protocol_gpu_worker import (
    MAX_INPUT_TOKENS, MAX_NEW_TOKENS, MAX_TRIALS, MODEL, PROVISION_MINUTES, REVISION,
    decode_message, manifest_sha256, model_bytes, worker_sha256,
)

CAPABILITY = "gpu.integer_prompt.v1"
WORKLOAD = "protocol.integer_prompt"
BACKEND = "protocol-gpu-ssh"
ARTIFACT = "responses.json"
MINUTES = 10


class ProtocolGPUConfig(OODGPUConfig):
    worker_filename: ClassVar[str] = "protocol_gpu_worker.py"
    allow_model_download: bool = False

    @model_validator(mode="before")
    @classmethod
    def resolve_cache(cls, raw):
        if not isinstance(raw, dict) or "model_cache" not in raw:
            return raw
        raw = dict(raw)
        cache = raw.pop("model_cache")
        if type(cache) is not str or not cache.startswith("/") or cache.rstrip("/") == "" or "snapshot_path" in raw:
            raise ValueError("Provide either an absolute model_cache or an explicit snapshot_path")
        raw["snapshot_path"] = f"{cache.rstrip('/')}/{REVISION}"
        return raw


def provisioning_scope(config):
    if not config.allow_model_download:
        return None
    return {"source": "https://huggingface.co", "manifest_sha256": manifest_sha256(), "max_bytes": model_bytes(),
            "walltime_minutes": PROVISION_MINUTES, "attempts": 1}


class Response(StrictModel):
    index: int = Field(ge=0, lt=MAX_TRIALS)
    text: str = Field(max_length=512)
    input_tokens: int = Field(ge=1, le=MAX_INPUT_TOKENS)
    output_tokens: int = Field(ge=1, le=MAX_NEW_TOKENS)
    elapsed_ns: int = Field(ge=0)


class Receipt(StrictModel):
    job_id: str
    job_sha256: str = Field(pattern=HEX_SHA)
    worker_sha256: str = Field(pattern=HEX_SHA)
    model: str
    revision: str
    responses: list[Response] = Field(min_length=1, max_length=MAX_TRIALS)
    runtime: GPURuntime


def validate_receipt(job, plan, document):
    receipt = Receipt.model_validate(document)
    count = len(job.workload.parameters["trials"])
    if (receipt.job_id != job.job_id or receipt.job_sha256 != job.fingerprint() or receipt.model != MODEL
            or receipt.revision != REVISION or receipt.worker_sha256 != plan["worker_sha256"]
            or receipt.runtime.gpu_index != plan["gpu_config"]["gpu_index"]
            or [row.index for row in receipt.responses] != list(range(count))
            or sum(row.elapsed_ns for row in receipt.responses) >= (receipt.runtime.elapsed_ms + 1) * 1_000_000):
        raise ValueError("GPU receipt differs from the approved protocol batch")
    return receipt


class ProtocolGPUBackend(GPUReceiptBackend):
    def __init__(self, plan):
        super().__init__()
        self._plan = plan
        self._config = ProtocolGPUConfig.model_validate(plan["gpu_config"])

    def _request(self, job, phase):
        if (job.workload.name != WORKLOAD or len(self._plan["jobs"]) != 1
                or self._plan["jobs"][0]["sha256"] != job.fingerprint()
                or self._plan["worker_sha256"] != worker_sha256()):
            raise ValueError("GPU batch differs from the verified program or plan")
        if phase == "provision" and (
                not self._config.allow_model_download or self._plan["model_provisioning"] != provisioning_scope(self._config)):
            raise PermissionError("Model download is outside the approved plan")
        timeout = {"preflight": 3, "provision": PROVISION_MINUTES, "measurement": MINUTES}[phase]
        return ssh_receipt(self._config, {
            "job_id": job.job_id, "job_sha256": job.fingerprint(), "worker_sha256": self._plan["worker_sha256"],
            "snapshot": self._config.snapshot_path, "gpu_index": self._config.gpu_index,
            "prompts": ([trial["prompt"] for trial in job.workload.parameters["trials"]] if phase == "measurement" else []),
            "phase": phase,
        }, minutes=timeout, decoder=decode_message)

    def _model_receipt(self, job, phase):
        raw = self._request(job, phase)
        keys = {"job_id", "job_sha256", "worker_sha256", "snapshot", "ready"}
        if phase == "provision":
            keys |= {"installed", "manifest_sha256"}
        if (not isinstance(raw, dict) or set(raw) != keys
                or type(raw["ready"]) is not bool or raw["job_id"] != job.job_id
                or raw["job_sha256"] != job.fingerprint() or raw["worker_sha256"] != self._plan["worker_sha256"]
                or raw["snapshot"] != self._config.snapshot_path):
            raise ValueError("Model preflight differs from the approved snapshot")
        if phase == "provision" and (
                not raw["ready"] or type(raw["installed"]) is not bool or raw["manifest_sha256"] != manifest_sha256()):
            raise ValueError("Model preparation receipt differs from the approved manifest")
        return raw

    def prepare_model(self, job):
        raw = self._model_receipt(job, "preflight")
        if not raw["ready"] and self._config.allow_model_download:
            raw = self._model_receipt(job, "provision")
        elif raw["ready"]:
            raw = {**raw, "installed": False, "manifest_sha256": manifest_sha256()}
        return raw

    def submit(self, job):
        if self.artifact is not None:
            if self.artifact.handle.spec_sha256 != job.fingerprint():
                raise ValueError("GPU batch differs from the original run")
            return self.artifact.handle
        raw = self._request(job, "measurement")
        receipt = validate_receipt(job, self._plan, raw)
        handle = JobHandle(backend=BACKEND, remote_id=f"ssh:{job.job_id}", job_id=job.job_id,
                           spec_sha256=job.fingerprint(), simulated=False)
        self.artifact = JobArtifacts(handle=handle, files={ARTIFACT: receipt.model_dump(mode="json")},
                                     simulated=False, warning="Descriptive prompt comparison on a pinned local model")
        return handle
