"""Transport and receipts for the bounded multi-checkpoint lookup benchmark."""
from pathlib import PurePosixPath
from typing import ClassVar, Literal

from pydantic import Field, model_validator

from execution.agentic_gpu_worker import (
    CAPABILITY, MAX_INPUT, MAX_OUTPUT, MAX_TRIALS, MINUTES, MODELS, POLICIES, PROFILES, PROVISION_MINUTES,
    SEEDS, TASKS, asset_manifest, decode_message, worker_sha256,
)
from execution.contracts import JobArtifacts, JobHandle, StrictModel
from execution.ood_gpu import GPUReceiptBackend, GPURuntime, OODGPUConfig, REMOTE_PATH, ssh_receipt

WORKLOAD = "protocol.agentic_lookup"
BACKEND = "agentic-gpu-ssh"
ARTIFACT = "agentic-responses.json"


class AgenticSuite(StrictModel):
    schema_version: Literal["gpu.agentic_lookup.v1"] = CAPABILITY
    cases: dict
    profiles: list[str]
    policies: list[str]
    checkpoints: list[str]
    seeds: list[int]


def suite():
    return AgenticSuite(cases=TASKS, profiles=list(PROFILES), policies=list(POLICIES),
                        checkpoints=list(MODELS), seeds=list(SEEDS))


class AgenticGPUConfig(OODGPUConfig):
    worker_filename: ClassVar[str] = "agentic_gpu_worker.py"
    model_cache: str = Field(pattern=REMOTE_PATH, max_length=250)
    allow_model_download: bool = False

    @model_validator(mode="after")
    def validate_cache(self):
        cache = PurePosixPath(self.model_cache)
        if ".." in cache.parts or cache == PurePosixPath("/"):
            raise ValueError("The agentic model cache must be an explicit non-root path")
        return self

    def snapshot(self, model_key):
        if model_key == "qwen4":
            return self.snapshot_path
        return str(PurePosixPath(self.model_cache) / MODELS[model_key][1])


def execution_scope(config):
    models = {key: {**asset_manifest(key), "snapshot": config.snapshot(key)} for key in MODELS}
    return {
        "gpu_config": config.model_dump(mode="json"), "models": models, "network": True, "gpu": True,
        "max_ssh_commands": len(models) * (3 if config.allow_model_download else 2),
        "max_download_bytes": sum(model["bytes"] for model in models.values()) if config.allow_model_download else 0,
        "download_minutes_per_model": PROVISION_MINUTES if config.allow_model_download else 0,
        "walltime_minutes_per_model": MINUTES,
    }


class AgenticRuntime(GPURuntime):
    elapsed_ms: int = Field(ge=0, le=MINUTES * 60_000)


class Generation(StrictModel):
    text: str = Field(max_length=1024)
    input_tokens: int = Field(ge=1, le=MAX_INPUT)
    output_tokens: int = Field(ge=1, le=MAX_OUTPUT)
    elapsed_ns: int = Field(ge=0)


class TrialOutput(StrictModel):
    id: str
    action: Generation
    answer: Generation


def validate_receipt(job, plan, raw):
    keys = {"job_id", "job_sha256", "worker_sha256", "model_key", "model", "runtime", "responses"}
    key = job.workload.parameters["model_key"]
    if (not isinstance(raw, dict) or set(raw) != keys or raw["job_id"] != job.job_id
            or raw["job_sha256"] != job.fingerprint() or raw["worker_sha256"] != plan["worker_sha256"]
            or raw["model_key"] != key or raw["model"] != asset_manifest(key)):
        raise ValueError("Agentic receipt provenance mismatch")
    runtime = AgenticRuntime.model_validate(raw["runtime"])
    if not isinstance(raw["responses"], list) or not 1 <= len(raw["responses"]) <= MAX_TRIALS:
        raise ValueError("Unexpected agentic response count")
    items = [TrialOutput.model_validate(row) for row in raw["responses"]]
    if (runtime.gpu_index != plan["gpu_config"]["gpu_index"]
            or [item.id for item in items] != [trial["id"] for trial in job.workload.parameters["trials"]]
            or sum(item.action.elapsed_ns + item.answer.elapsed_ns for item in items) >= (runtime.elapsed_ms + 1) * 1_000_000):
        raise ValueError("Agentic receipt differs from the approved trial schedule")
    return runtime, items


class AgenticGPUBackend(GPUReceiptBackend):
    def __init__(self, plan):
        super().__init__()
        self._plan = plan
        self._approved = {item["spec"]["job_id"]: item["sha256"] for item in plan["jobs"]}
        self._config = AgenticGPUConfig.model_validate(plan["gpu_config"])
        self._results = {}

    def _request(self, job, phase):
        if (job.workload.name != WORKLOAD or self._approved.get(job.job_id) != job.fingerprint()
                or self._plan["worker_sha256"] != worker_sha256()):
            raise ValueError("Agentic job differs from the authorized plan or worker bundle")
        if phase == "provision" and not self._config.allow_model_download:
            raise PermissionError("Agentic model download has no authorization")
        key = job.workload.parameters["model_key"]
        return ssh_receipt(self._config, {
            "phase": phase, "job_id": job.job_id, "job_sha256": job.fingerprint(),
            "worker_sha256": self._plan["worker_sha256"], "model_key": key,
            "snapshot": self._config.snapshot(key), "gpu_index": self._config.gpu_index,
            "trials": job.workload.parameters["trials"] if phase == "measurement" else [],
        }, minutes={"preflight": 3, "provision": PROVISION_MINUTES, "measurement": MINUTES}[phase],
            decoder=decode_message)

    def _preparation(self, job, phase):
        raw = self._request(job, phase)
        key = job.workload.parameters["model_key"]
        expected = {"job_id": job.job_id, "job_sha256": job.fingerprint(), "worker_sha256": self._plan["worker_sha256"],
                    "model_key": key, "manifest": asset_manifest(key)}
        if (not isinstance(raw, dict) or set(raw) != {*expected, "ready", "installed"}
                or any(raw[name] != value for name, value in expected.items())
                or type(raw["ready"]) is not bool or type(raw["installed"]) is not bool
                or (phase == "preflight" and raw["installed"]) or (phase == "provision" and not raw["ready"])):
            raise ValueError("Agentic model preparation receipt mismatch")
        return raw

    def prepare_models(self, jobs):
        receipts = []
        for job in jobs:
            receipt = self._preparation(job, "preflight")
            if not receipt["ready"]:
                if not self._config.allow_model_download:
                    raise FileNotFoundError(f"Verified snapshot unavailable for {receipt['model_key']}")
                receipt = self._preparation(job, "provision")
            receipts.append(receipt)
        return receipts

    def submit(self, job):
        if job.job_id not in self._results:
            raw = self._request(job, "measurement")
            validate_receipt(job, self._plan, raw)
            handle = JobHandle(backend=BACKEND, remote_id=f"ssh:{job.job_id}", job_id=job.job_id,
                               spec_sha256=job.fingerprint(), simulated=False)
            self._results[job.job_id] = JobArtifacts(
                handle=handle, files={ARTIFACT: raw}, simulated=False, warning="Descriptive agentic benchmark",
            )
        handle = self._results[job.job_id].handle
        if handle.spec_sha256 != job.fingerprint():
            raise ValueError("Job differs from the original submission")
        return handle

    def collect(self, handle):
        artifact = self._results.get(handle.job_id)
        if artifact is None or artifact.handle != handle:
            raise KeyError("The original agentic receipt is unavailable")
        return artifact
