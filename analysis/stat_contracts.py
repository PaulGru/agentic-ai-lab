"""V8.5.2–V8.5.4: explicit, typed contracts for statistical training data.

The replicated sample format is a *pedagogical fixture*; it must not be
mistaken for verified independent scientific measurements. The V8.5.1 pilot
is always treated as aggregated job outputs, never replicated observations.
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from analysis.contracts import ID, HEX_SHA, fingerprint
from execution.contracts import StrictModel


class PairedObservation(StrictModel):
    pair_id: str = Field(pattern=ID)
    baseline: float = Field(allow_inf_nan=False, ge=-1_000_000, le=1_000_000)
    variant: float = Field(allow_inf_nan=False, ge=-1_000_000, le=1_000_000)


class PedagogicalPairedFixture(StrictModel):
    schema_version: Literal["8.5.3"] = "8.5.3"
    kind: Literal["pedagogical_paired_fixture"] = "pedagogical_paired_fixture"
    dataset_id: str = Field(pattern=ID)
    metric_name: str = Field(pattern=r"^[a-z][a-z0-9_.-]{1,63}$")
    unit: str = Field(min_length=1, max_length=24)
    pairs: list[PairedObservation] = Field(min_length=6, max_length=16)
    sign_flip_symmetry_assumed: Literal[True] = True
    scientific_evidence: Literal[False] = False

    @model_validator(mode="after")
    def distinct_pairs(self):
        identifiers = [pair.pair_id for pair in self.pairs]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("pair_id must be unique; no pseudo-replication")
        return self

    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


class NumericSummary(StrictModel):
    n: int = Field(ge=1, le=100)
    mean: float = Field(allow_inf_nan=False)
    median: float = Field(allow_inf_nan=False)
    minimum: float = Field(allow_inf_nan=False)
    maximum: float = Field(allow_inf_nan=False)
    sample_std_dev: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class DescriptiveResult(StrictModel):
    kind: Literal["job_aggregates", "paired_fixture"]
    source_sha256: str = Field(pattern=HEX_SHA)
    observation_unit: str
    n_independent_pairs: int | None = Field(default=None, ge=1, le=16)
    groups: dict[str, NumericSummary]
    mean_difference_variant_minus_baseline: float | None = Field(default=None, allow_inf_nan=False)
    cautions: list[str] = Field(default_factory=list)


class ValidityReport(StrictModel):
    kind: Literal["validity_gate"] = "validity_gate"
    source_kind: Literal["verified_pilot_aggregates", "pedagogical_paired_fixture"]
    source_sha256: str = Field(pattern=HEX_SHA)
    ready_for_descriptive: Literal[True] = True
    ready_for_pedagogical_inference: bool
    ready_for_scientific_inference: Literal[False] = False
    blockers: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)


class InferenceResult(StrictModel):
    kind: Literal["pedagogical_exact_sign_flip"] = "pedagogical_exact_sign_flip"
    source_sha256: str = Field(pattern=HEX_SHA)
    n_pairs: int = Field(ge=8, le=16)
    mean_difference: float = Field(allow_inf_nan=False)
    exact_two_sided_p: float = Field(ge=0, le=1, allow_inf_nan=False)
    sign_flips_enumerated: int = Field(ge=256, le=65_536)
    bootstrap_percentile_ci_95: list[float] = Field(min_length=2, max_length=2)
    bootstrap_resamples: int = Field(ge=1000, le=5000)
    bootstrap_seed: int
    paired_standardized_effect_dz: float = Field(allow_inf_nan=False)
    scientific_evidence: Literal[False] = False
    caveats: list[str] = Field(min_length=1)

    @field_validator("bootstrap_percentile_ci_95")
    @classmethod
    def valid_ci(cls, value: list[float]):
        if len(value) != 2 or not all(math.isfinite(x) for x in value) or value[0] > value[1]:
            raise ValueError("Finite ordered confidence limits required")
        return value


class AnalysisReport(StrictModel):
    schema_version: Literal["8.5"] = "8.5"
    kind: Literal["deterministic_statistical_analysis"] = "deterministic_statistical_analysis"
    source_kind: Literal["verified_pilot_aggregates", "pedagogical_paired_fixture"]
    source_id: str = Field(pattern=ID)
    source_sha256: str = Field(pattern=HEX_SHA)
    descriptive: DescriptiveResult
    validity: ValidityReport
    inference: InferenceResult | None = None
    scientific_evidence: Literal[False] = False
    conclusion_policy: Literal["no_scientific_claims"] = "no_scientific_claims"

    @model_validator(mode="after")
    def enforce_consistency(self):
        if (self.descriptive.source_sha256 != self.source_sha256
                or self.validity.source_sha256 != self.source_sha256
                or self.validity.source_kind != self.source_kind):
            raise ValueError("Analysis report source digest/kind mismatch")
        if self.source_kind == "verified_pilot_aggregates":
            if self.inference is not None or self.validity.ready_for_pedagogical_inference:
                raise ValueError("Aggregate pilot measurements cannot be inferred over")
            if self.descriptive.kind != "job_aggregates":
                raise ValueError("Wrong descriptive kind for verified pilot")
        else:
            if self.descriptive.kind != "paired_fixture":
                raise ValueError("Wrong descriptive kind for paired fixture")
            if (self.inference is None) != (not self.validity.ready_for_pedagogical_inference):
                raise ValueError("Inference must match validity gate")
        if self.inference and self.inference.source_sha256 != self.source_sha256:
            raise ValueError("Inference source digest mismatch")
        return self

    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json"))
