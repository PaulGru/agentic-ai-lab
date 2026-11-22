"""Recompute paired OOD rehearsal scores from the original simulated job ledger."""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from analysis.contracts import HEX_SHA, fingerprint
from execution.contracts import StrictModel
from execution.job_manager import DurableJobManager
from execution.ood_gpu import GPURuntime, OODGPUBackend, OODGPUProtocol, validate_receipt
from execution.ood_replay import CASES, PROFILES, REPLIES, OODReplayBackend, OODReplayProtocol, recorded_response

SOURCE_KIND = "verified_ood_replay"
GPU_SOURCE_KIND = "verified_ood_gpu"


class IncompleteOODReceipt(RuntimeError):
    """A planned replay job has no durable successful response."""


class OODTrial(StrictModel):
    job_id: str
    job_sha256: str = Field(pattern=HEX_SHA)
    artifacts_sha256: str = Field(pattern=HEX_SHA)
    task_id: str
    model: str
    distribution: Literal["reference", "ood"]
    correct: bool


class OODArm(StrictModel):
    correct: int
    total: int
    accuracy_percent: float
    trial_ids: list[str]


class OODAnalysis(StrictModel):
    protocol_sha256: str = Field(pattern=HEX_SHA)
    arms: dict[str, OODArm]
    degradation_points: dict[str, float]
    scientific_evidence: Literal[False] = False
    ready_for_scientific_inference: Literal[False] = False

    def fingerprint(self):
        return fingerprint(self.model_dump(mode="json"))


class OODReplayAnalysis(OODAnalysis):
    source_kind: Literal["verified_ood_replay"] = SOURCE_KIND
    trials: list[OODTrial]
    simulated: Literal[True] = True


class OODGPUTrial(OODTrial):
    trial_id: str


class OODGPUAnalysis(OODAnalysis):
    source_kind: Literal["verified_ood_gpu"] = GPU_SOURCE_KIND
    trials: list[OODGPUTrial]
    runtime: GPURuntime
    model_revision: str
    reference_calibration_passed: bool
    simulated: Literal[False] = False


def _score_arms(trials, models):
    arms = {}
    for model in models:
        for profile in PROFILES:
            rows = [trial for trial in trials if trial.model == model and trial.distribution == profile]
            correct = sum(trial.correct for trial in rows)
            arms[f"{model}-{profile}"] = OODArm(
                correct=correct, total=len(rows), accuracy_percent=100.0 * correct / len(rows),
                trial_ids=[row.trial_id if isinstance(row, OODGPUTrial) else row.job_id for row in rows],
            )
    degradation = {
        model: arms[f"{model}-reference"].accuracy_percent - arms[f"{model}-ood"].accuracy_percent
        for model in models
    }
    return arms, degradation


def analyze_ood_replay(protocol: OODReplayProtocol, manager: DurableJobManager) -> OODReplayAnalysis:
    if not manager.read_only:
        raise PermissionError("OOD analysis requires the original read-only job ledger")
    trials = []
    for job in protocol.jobs():
        record = manager.get(job.job_id)
        if record is None or record.phase != "succeeded" or record.handle is None or record.artifacts is None:
            raise IncompleteOODReceipt(f"Incomplete OOD replay receipt for {job.job_id}")
        handle = record.handle
        if (record.backend != OODReplayBackend.__name__ or handle.backend != "fake"
                or record.spec_sha256 != job.fingerprint() or handle.spec_sha256 != record.spec_sha256
                or handle.job_id != job.job_id or record.artifacts.handle != handle
                or not handle.simulated or not record.artifacts.simulated):
            raise ValueError("OOD replay receipt provenance mismatch")
        if record.artifacts.files != recorded_response(job):
            raise ValueError("OOD replay response differs from the fixed fixture")
        parameters = job.workload.parameters
        task = parameters["task_id"]
        answer = record.artifacts.files["response.json"]["answer"]
        trials.append(OODTrial(
            job_id=job.job_id, job_sha256=record.spec_sha256, artifacts_sha256=record.artifacts_sha256,
            task_id=task, model=parameters["model"], distribution=parameters["distribution"],
            correct=answer == CASES[task]["expected"],
        ))
    arms, degradation = _score_arms(trials, REPLIES)
    return OODReplayAnalysis(protocol_sha256=protocol.fingerprint(), trials=trials, arms=arms,
                             degradation_points=degradation)


def analyze_ood_gpu(protocol: OODGPUProtocol, manager: DurableJobManager) -> OODGPUAnalysis:
    if not manager.read_only:
        raise PermissionError("OOD GPU analysis requires the original read-only job ledger")
    job = protocol.job()
    record = manager.get(job.job_id)
    if record is None or record.phase != "succeeded" or record.handle is None or record.artifacts is None:
        raise IncompleteOODReceipt("GPU calibration has no complete receipt")
    handle = record.handle
    if (record.backend != OODGPUBackend.__name__ or handle.backend != "ood-gpu-ssh"
            or record.spec_sha256 != job.fingerprint() or handle.spec_sha256 != record.spec_sha256
            or handle.job_id != job.job_id or record.artifacts.handle != handle
            or handle.simulated or record.artifacts.simulated
            or set(record.artifacts.files) != set(job.expected_artifacts)):
        raise ValueError("GPU receipt provenance mismatch")
    receipt = validate_receipt(protocol, job, record.artifacts.files["responses.json"])
    model_key = "qwen3-4b"
    trials = []
    for response in receipt.responses:
        task, profile = response.id.split("-")
        trials.append(OODGPUTrial(
            trial_id=response.id, job_id=job.job_id, job_sha256=record.spec_sha256,
            artifacts_sha256=record.artifacts_sha256, task_id=task, model=model_key, distribution=profile,
            correct=response.text.strip() == CASES[task]["expected"],
        ))
    arms, degradation = _score_arms(trials, (model_key,))
    reference = arms[f"{model_key}-reference"]
    return OODGPUAnalysis(protocol_sha256=protocol.fingerprint(), trials=trials, arms=arms,
                          degradation_points=degradation, runtime=receipt.runtime, model_revision=receipt.revision,
                          reference_calibration_passed=reference.correct == reference.total)
