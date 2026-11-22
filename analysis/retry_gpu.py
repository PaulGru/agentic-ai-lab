"""Recompute the descriptive paired contrasts from a verified GPU receipt."""
from typing import Literal

from pydantic import Field

from analysis.contracts import HEX_SHA, fingerprint
from analysis.retry_pilot import PairedRetryAnalysis, TrialResult, paired_comparison
from execution.contracts import ExperimentJob
from execution.job_manager import DurableJobManager
from execution.retry_gpu import ARTIFACT, BACKEND, RetryGPUBackend, RetryGPURuntime, validate_receipt
from execution.retry_gpu_worker import parse_object, tool_result, worker_sha256
from experiments.retry_pilot import PilotCorpus

SOURCE_KIND = "verified_retry_gpu"


class RetryGPUAnalysis(PairedRetryAnalysis):
    source_kind: Literal["verified_retry_gpu"] = SOURCE_KIND
    protocol_sha256: str = Field(pattern=HEX_SHA)
    action_plan_sha256: str = Field(pattern=HEX_SHA)
    corpus_sha256: str = Field(pattern=HEX_SHA)
    job_sha256: str = Field(pattern=HEX_SHA)
    artifacts_sha256: str = Field(pattern=HEX_SHA)
    requested_question: str
    measured_question: str
    scope: str
    estimand: str
    revision: str
    runtime: RetryGPURuntime
    simulated: Literal[False] = False


def analyze_retry_gpu(protocol, corpus: PilotCorpus, plan, manager: DurableJobManager) -> RetryGPUAnalysis:
    if not manager.read_only:
        raise PermissionError("Catalog GPU analysis requires a read-only job ledger")
    if plan["worker_sha256"] != worker_sha256():
        raise ValueError("Catalog GPU scoring requires the source approved for this run")
    job = ExperimentJob.model_validate(plan["jobs"][0]["spec"])
    record = manager.get(job.job_id)
    if record is None or record.phase != "succeeded" or record.handle is None or record.artifacts is None:
        raise RuntimeError("Catalog GPU job has no complete receipt")
    handle = record.handle
    if (record.backend != RetryGPUBackend.__name__ or handle.backend != BACKEND
            or record.spec_sha256 != job.fingerprint() or handle.spec_sha256 != record.spec_sha256
            or handle.job_id != job.job_id or record.artifacts.handle != handle
            or handle.simulated or record.artifacts.simulated or set(record.artifacts.files) != {ARTIFACT}):
        raise ValueError("Catalog GPU receipt provenance mismatch")
    receipt = validate_receipt(plan, job, record.artifacts.files[ARTIFACT])
    parameters = job.workload.parameters
    results = []
    for trial, response in zip(parameters["trials"], receipt.responses, strict=True):
        task = parameters["tasks"][trial["task_id"]]
        _, attempts = tool_result(task, trial, response.lookup.text)
        answer = parse_object(response.answer.text)
        value = answer.get("answer")
        if set(answer) != {"answer"} or (value is not None and not isinstance(value, str)):
            score = "invalid_response"
        elif value is None:
            score = "abstention"
        else:
            score = "correct" if value == task["value"] else "incorrect"
        results.append(TrialResult(
            **trial, score=score, lookup_attempts=attempts, generations=2,
            input_tokens=response.lookup.input_tokens + response.answer.input_tokens,
            output_tokens=response.lookup.output_tokens + response.answer.output_tokens,
            event_sha256=fingerprint(response.model_dump(mode="json")),
        ))
    return RetryGPUAnalysis(
        protocol_sha256=fingerprint(protocol), action_plan_sha256=plan["plan_sha256"],
        corpus_sha256=plan["corpus_sha256"], job_sha256=job.fingerprint(), artifacts_sha256=record.artifacts_sha256,
        requested_question=protocol["requested_question"], measured_question=protocol["measured_question"],
        scope=protocol["scope"], procedure=protocol["procedure"], estimand=protocol["estimand"],
        model=receipt.model, revision=receipt.revision, runtime=receipt.runtime,
        task_count=len(corpus.tasks), trial_count=len(results), generation_count=sum(row.generations for row in results),
        trials=results, comparison=paired_comparison(corpus, results), limitations=protocol["limitations"],
    )
