"""V8.5.1 typed data contract and conservative quality assessment.

One record is one **aggregate from an executed job**, not one independently
replicated scientific observation. Data exported here are an engineering demo.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from execution.contracts import StrictModel

HEX_SHA = r"^[a-f0-9]{64}$"
ID = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      allow_nan=False, separators=(",", ":"))


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class MetricObservation(StrictModel):
    """A single aggregate metric, NOT independent task-level observations."""

    observation_id: str = Field(pattern=ID)
    run_id: str = Field(pattern=ID)
    job_id: str = Field(pattern=ID)
    job_sha256: str = Field(pattern=HEX_SHA)
    artifacts_sha256: str = Field(pattern=HEX_SHA)
    metric_name: Literal["synthetic.mean"] = "synthetic.mean"
    value: float = Field(allow_inf_nan=False)
    input_value_count: int = Field(ge=1, le=100)
    simulated: bool

    @field_validator("value")
    @classmethod
    def finite_metric(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Metric value must be finite")
        return value


class AnalysisDataset(StrictModel):
    schema_version: Literal["8.5.1"] = "8.5.1"
    kind: Literal["normalized_aggregate_measurements"] = "normalized_aggregate_measurements"
    campaign_id: str = Field(pattern=ID)
    protocol_sha256: str = Field(pattern=HEX_SHA)
    source_report_sha256: str = Field(pattern=HEX_SHA)
    provenance: Literal["verified_v846_sqlite_ledger"] = "verified_v846_sqlite_ledger"
    source_backend: Literal["docker-local"] = "docker-local"
    source_report_kind: Literal["engineering_pilot_report"] = "engineering_pilot_report"
    unit_of_observation: Literal["one_job_level_aggregate"] = "one_job_level_aggregate"
    scientific_evidence: Literal[False] = False
    observations: list[MetricObservation] = Field(min_length=2, max_length=4)

    @model_validator(mode="after")
    def unique_observations(self):
        for field in ("run_id", "job_id", "observation_id"):
            values = [getattr(o, field) for o in self.observations]
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate {field} in normalized dataset")
        return self

    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


class QualityAssessment(StrictModel):
    schema_version: Literal["8.5.1"] = "8.5.1"
    kind: Literal["engineering_data_quality"] = "engineering_data_quality"
    campaign_id: str = Field(pattern=ID)
    dataset_sha256: str = Field(pattern=HEX_SHA)
    observation_count: int = Field(ge=2, le=4)
    verified_job_count: int = Field(ge=2, le=4)
    simulated_observation_count: int = Field(ge=0, le=4)
    integrity_checks_passed: Literal[True] = True
    complete_for_pilot: Literal[True] = True
    ready_for_descriptive_processing: Literal[True] = True
    ready_for_scientific_inference: Literal[False] = False
    blockers: list[str] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)


def verify_export(dataset: AnalysisDataset, quality: QualityAssessment) -> None:
    """Validate consistency of saved exports (NOT re-verification against SQLite)."""
    if dataset.campaign_id != quality.campaign_id:
        raise ValueError("Dataset and assessment campaign IDs differ")
    if dataset.fingerprint() != quality.dataset_sha256:
        raise ValueError("Dataset and assessment SHA-256 differ")
    if len(dataset.observations) != quality.observation_count:
        raise ValueError("Observation count mismatch")
    if len({x.job_id for x in dataset.observations}) != quality.verified_job_count:
        raise ValueError("Verified job count mismatch")
    if sum(x.simulated for x in dataset.observations) != quality.simulated_observation_count:
        raise ValueError("Simulated observation count mismatch")
    if dataset.scientific_evidence is not False or quality.ready_for_scientific_inference is not False:
        raise ValueError("Pilot data may not be used for scientific inference")
