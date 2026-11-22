"""An allowlisted replay of fixed answers for the offline OOD rehearsal."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from execution.backends.fake import FakeBackend
from execution.campaign import canonical
from execution.contracts import ExperimentJob, JobHandle, JobStatus, ResourceRequest, StrictModel, WorkloadSpec
from execution.job_manager import DurableJobManager

WORKLOAD = "ood_replay"
PROFILES = ("reference", "ood")
CASES = {
    "stock": {
        "reference": "Fiche K7. Stock : 7. Quelle est la quantité en stock ?",
        "ood": "Inventaire [quantité=7] pour K7. Donne la quantité disponible.",
        "expected": "7",
    },
    "zone": {
        "reference": "Fiche K9. Zone : C-2. Dans quelle zone se trouve K9 ?",
        "ood": "K9 est localisé à l’emplacement C-2. Indique le code de zone.",
        "expected": "C-2",
    },
}
REPLIES = {
    "simule-a": {"reference": {"stock": "7", "zone": "C-2"}, "ood": {"stock": "7", "zone": "B-1"}},
    "simule-b": {"reference": {"stock": "5", "zone": "C-2"}, "ood": {"stock": "7", "zone": "B-1"}},
}


def fixture_sha256():
    return hashlib.sha256(canonical({"cases": CASES, "replies": REPLIES}).encode("utf-8")).hexdigest()


class OODReplayProtocol(StrictModel):
    schema_version: Literal["9.0-ood-replay-1"] = "9.0-ood-replay-1"
    workload: Literal["ood_replay"] = WORKLOAD
    campaign_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,31}$")
    requested_question: str
    measured_question: Literal["Le workflow reconstruit-il les écarts entre les conditions de référence et OOD simulées ?"] = (
        "Le workflow reconstruit-il les écarts entre les conditions de référence et OOD simulées ?")
    scope: Literal["Deux tâches appariées, deux formulations et deux modèles fictifs."] = (
        "Deux tâches appariées, deux formulations et deux modèles fictifs.")
    mission_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    literature_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    fixture_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    simulated: Literal[True] = True
    scientific_evidence: Literal[False] = False

    @model_validator(mode="after")
    def validate_fixture(self):
        if self.fixture_sha256 != fixture_sha256():
            raise ValueError("OOD replay fixture differs from the approved version")
        return self

    def fingerprint(self):
        return hashlib.sha256(canonical(self.model_dump(mode="json")).encode("utf-8")).hexdigest()

    def jobs(self):
        digest = self.fingerprint()
        return [
            ExperimentJob(
                experiment_id=self.campaign_id,
                job_id=f"{self.campaign_id}-{model}-{profile}-{task}",
                workload=WorkloadSpec(name="ood.replay", parameters={
                    "model": model, "distribution": profile, "task_id": task, "prompt": case[profile],
                }),
                seed=0, resources=ResourceRequest(cpus=1, memory_mb=128, gpus=0, walltime_minutes=1),
                expected_artifacts=["response.json"], source_design_sha256=digest,
            )
            for model in REPLIES for profile in PROFILES for task, case in CASES.items()
        ]


def recorded_response(job):
    parameters = job.workload.parameters
    answer = REPLIES[parameters["model"]][parameters["distribution"]][parameters["task_id"]]
    return {"response.json": {"answer": answer}}


class OODReplayBackend(FakeBackend):
    def __init__(self, protocol: OODReplayProtocol):
        super().__init__()
        self._allowed = {job.job_id: job.fingerprint() for job in protocol.jobs()}

    def submit(self, job: ExperimentJob) -> JobHandle:
        if self._allowed.get(job.job_id) != job.fingerprint():
            raise ValueError("Job differs from the allowlisted OOD replay")
        handle = super().submit(job)
        if self.status(handle).status == JobStatus.SUBMITTED:
            self.mark_running(handle)
            self.finish(handle, success=True, files=recorded_response(job))
        return handle


def run_ood_replay(protocol: OODReplayProtocol, ledger: Path) -> None:
    backend = OODReplayBackend(protocol)
    manager = DurableJobManager(backend, ledger)
    for job in protocol.jobs():
        record = manager.submit(job)
        if record.phase == "submitted":
            record = manager.refresh(job.job_id)
        if record.phase != "succeeded":
            raise RuntimeError("OOD replay job has no successful durable receipt")
