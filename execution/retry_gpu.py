"""Compile a frozen catalog protocol and dispatch its exact GPU job over SSH."""
from __future__ import annotations

from copy import deepcopy
import re
from typing import ClassVar, Literal

from pydantic import Field

from analysis.contracts import HEX_SHA, fingerprint
from execution.contracts import ExperimentJob, JobArtifacts, JobHandle, ResourceRequest, StrictModel, WorkloadSpec
from execution.job_manager import DurableJobManager
from execution.ood_gpu import GPUReceiptBackend, GPURuntime, OODGPUConfig, ssh_receipt
from execution.retry_gpu_worker import (
    MAX_INPUT_TOKENS, MAX_NEW_TOKENS, MODEL, REVISION, decode_message, validate_batch, worker_sha256,
)
from experiments.retry_pilot import PilotCorpus, prepare_plan

WORKLOAD = "retry_gpu"
PROTOCOL_VERSION = "9.0-retry-gpu-1"
BACKEND = "retry-gpu-ssh"
MINUTES = 15
ARTIFACT = "responses.json"


class RetryGPUConfig(OODGPUConfig):
    worker_filename: ClassVar[str] = "retry_gpu_worker.py"


class Generation(StrictModel):
    text: str = Field(max_length=1024)
    input_tokens: int = Field(ge=1, le=MAX_INPUT_TOKENS)
    output_tokens: int = Field(ge=1, le=MAX_NEW_TOKENS)
    latency_ms: int = Field(ge=0)


class TrialResponse(StrictModel):
    id: str
    lookup: Generation
    answer: Generation


class RetryGPURuntime(GPURuntime):
    elapsed_ms: int = Field(ge=0, le=MINUTES * 60_000)


class RetryGPUReceipt(StrictModel):
    job_id: str
    job_sha256: str = Field(pattern=HEX_SHA)
    worker_sha256: str = Field(pattern=HEX_SHA)
    model: Literal["Qwen/Qwen3-4B-Instruct-2507"]
    revision: Literal["cdbee75f17c01a7cc42f958dc650907174af0554"]
    dtype: Literal["bfloat16"]
    responses: list[TrialResponse] = Field(min_length=32, max_length=32)
    runtime: RetryGPURuntime


def compile_job(corpus: PilotCorpus, protocol_sha256: str) -> ExperimentJob:
    schedule = prepare_plan(corpus)
    parameters = {
        "model": MODEL, "revision": REVISION, "dtype": "bfloat16",
        "max_input_tokens": MAX_INPUT_TOKENS, "max_new_tokens": MAX_NEW_TOKENS,
        "tasks": {task.task_id: {"prompt": task.prompt, "product": task.product, "field": task.field,
                                 "value": getattr(corpus.catalog[task.product], task.field)} for task in corpus.tasks},
        "trials": schedule["trials"],
    }
    validate_batch(parameters)
    return ExperimentJob(
        experiment_id=corpus.pilot_id, job_id=f"{corpus.pilot_id}-retry-gpu",
        workload=WorkloadSpec(name="catalog.retry_gpu", parameters=parameters),
        seed=corpus.order_seed, resources=ResourceRequest(cpus=2, memory_mb=16_384, gpus=1, walltime_minutes=MINUTES),
        expected_artifacts=[ARTIFACT], source_design_sha256=protocol_sha256,
    )


def action_plan(protocol, corpus: PilotCorpus) -> dict:
    """Describe precisely the one job authorized by the protocol and its target."""
    schedule = prepare_plan(corpus)
    if (protocol["schema_version"] != PROTOCOL_VERSION or protocol["workload"] != WORKLOAD
            or protocol["model"] != MODEL or protocol["revision"] != REVISION
            or not re.fullmatch(HEX_SHA, protocol["worker_sha256"])
            or protocol["corpus_sha256"] != schedule["corpus_sha256"]
            or protocol["schedule_sha256"] != schedule["plan_sha256"]):
        raise ValueError("Catalog or schedule differs from the research protocol")
    protocol_sha256 = fingerprint(protocol)
    job = compile_job(corpus, protocol_sha256)
    budget = {
        "batches": 1, "trials": schedule["planned_trials"], "generations": schedule["max_model_calls"],
        "input_tokens": schedule["max_model_calls"] * MAX_INPUT_TOKENS,
        "output_tokens": schedule["max_model_calls"] * MAX_NEW_TOKENS,
        "lookups_per_trial": schedule["max_physical_lookups_per_trial"], "walltime_minutes": MINUTES,
    }
    plan = {
        "schema_version": PROTOCOL_VERSION, "backend": BACKEND, "simulated": False, "network": True, "gpu": True,
        "measured_question": protocol["measured_question"],
        "protocol_sha256": protocol_sha256, "corpus_sha256": schedule["corpus_sha256"],
        "schedule_sha256": schedule["plan_sha256"], "worker_sha256": protocol["worker_sha256"],
        "gpu_config": RetryGPUConfig.model_validate(protocol["config"]).model_dump(mode="json"),
        "budget": budget, "jobs": [{"spec": job.model_dump(mode="json"), "sha256": job.fingerprint()}],
    }
    return {**plan, "plan_sha256": fingerprint(plan)}


def validate_receipt(plan, job, document):
    receipt = RetryGPUReceipt.model_validate(document)
    if (receipt.job_id != job.job_id or receipt.job_sha256 != job.fingerprint()
            or receipt.worker_sha256 != plan["worker_sha256"] or receipt.runtime.gpu_index != plan["gpu_config"]["gpu_index"]
            or [row.id for row in receipt.responses] != [row["trial_id"] for row in job.workload.parameters["trials"]]
            or sum(row.lookup.latency_ms + row.answer.latency_ms for row in receipt.responses) > receipt.runtime.elapsed_ms):
        raise ValueError("GPU receipt differs from the catalog action plan")
    return receipt


class RetryGPUBackend(GPUReceiptBackend):
    def __init__(self, plan, *, approved_plan_sha256):
        payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
        if plan["plan_sha256"] != approved_plan_sha256 or fingerprint(payload) != approved_plan_sha256:
            raise PermissionError("GPU execution requires approval of the exact catalog action plan")
        if plan["schema_version"] != PROTOCOL_VERSION or plan["backend"] != BACKEND or len(plan["jobs"]) != 1:
            raise ValueError("Unsupported GPU action plan")
        super().__init__()
        self._plan = deepcopy(plan)
        self.config = RetryGPUConfig.model_validate(plan["gpu_config"])

    def submit(self, job):
        spec = self._plan["jobs"][0]
        if (job.workload.name != "catalog.retry_gpu"
                or job.model_dump(mode="json") != spec["spec"] or job.fingerprint() != spec["sha256"]
                or self._plan["worker_sha256"] != worker_sha256()):
            raise ValueError("GPU job or worker differs from the approved action plan")
        validate_batch(job.workload.parameters)
        if self.artifact is not None:
            return self.artifact.handle
        raw = ssh_receipt(self.config, {
            "job_id": job.job_id, "job_sha256": job.fingerprint(), "worker_sha256": self._plan["worker_sha256"],
            "snapshot": self.config.snapshot_path, "gpu_index": self.config.gpu_index,
            "parameters": job.workload.parameters,
        }, minutes=MINUTES, decoder=decode_message)
        receipt = validate_receipt(self._plan, job, raw)
        handle = JobHandle(backend=BACKEND, remote_id=f"ssh:{job.job_id}", job_id=job.job_id,
                           spec_sha256=job.fingerprint(), simulated=False)
        self.artifact = JobArtifacts(handle=handle, files={ARTIFACT: receipt.model_dump(mode="json")},
                                     simulated=False, warning="Descriptive pilot on one model revision")
        return handle


def run_retry_gpu(plan, ledger, *, approved_plan_sha256):
    backend = RetryGPUBackend(plan, approved_plan_sha256=approved_plan_sha256)
    job = ExperimentJob.model_validate(plan["jobs"][0]["spec"])
    manager = DurableJobManager(backend, ledger)
    record = manager.submit(job)
    if record.phase == "submitted":
        record = manager.refresh(job.job_id)
    if record.phase != "succeeded":
        raise RuntimeError(f"Catalog GPU job has no complete receipt ({record.phase})")
