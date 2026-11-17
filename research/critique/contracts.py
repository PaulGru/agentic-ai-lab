"""Explicit V8.6 boundaries: evidence, opinions, revisions and authorizations.

Schemas check structure, not authenticity or scientific truth. Public operations
must also rederive evidence and validate cross-document references.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from analysis.contracts import HEX_SHA, ID, canonical_json, fingerprint
from analysis.stat_contracts import AnalysisReport
from execution.contracts import StrictModel
from research.experimental_design import ExperimentalDesign

SourceKind = Literal["verified_pilot_aggregates", "pedagogical_paired_fixture"]
POINTER = r"^/(?:[A-Za-z0-9_~./-]){1,300}$"


class Fingerprinted(StrictModel):
    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


class Claim(StrictModel):
    claim_id: str = Field(pattern=ID)
    kind: Literal["observation", "hypothesis", "scientific_conclusion"]
    statement: str = Field(min_length=1, max_length=1000)
    evidence_refs: list[str] = Field(default_factory=list, max_length=8)
    claimed_value: float | None = Field(default=None, allow_inf_nan=False)


class DesignContext(Fingerprinted):
    mission_document: dict[str, Any]
    literature_document: dict[str, Any]
    design_document: dict[str, Any]
    matrix_document: dict[str, Any]


class EvidenceBundle(Fingerprinted):
    schema_version: Literal["8.6"] = "8.6"
    source_kind: SourceKind
    source_document: dict[str, Any]
    analysis: AnalysisReport
    dataset_document: dict[str, Any] | None = None
    quality_document: dict[str, Any] | None = None
    context: DesignContext | None = None
    claims: list[Claim] = Field(default_factory=list, max_length=30)
    design_execution_link: Literal["unavailable", "not_applicable"]
    scientific_evidence: Literal[False] = False

    @model_validator(mode="after")
    def bounded_and_consistent(self):
        if self.source_kind != self.analysis.source_kind:
            raise ValueError("Evidence source kind differs from analysis")
        pilot = self.source_kind == "verified_pilot_aggregates"
        if pilot != (self.dataset_document is not None and self.quality_document is not None):
            raise ValueError("Pilot evidence requires dataset and quality documents")
        if not pilot and (self.dataset_document is not None or self.quality_document is not None):
            raise ValueError("Fixture evidence may not impersonate a ledger dataset")
        expected = "unavailable" if self.context else "not_applicable"
        if self.design_execution_link != expected:
            raise ValueError("No generic V8.3 design-to-execution binding exists in V8.6")
        ids = [claim.claim_id for claim in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate claim identifiers")
        if len(canonical_json(self.model_dump(mode="json")).encode("utf-8")) > 524_288:
            raise ValueError("Evidence bundle exceeds 512 KiB")
        return self


class CritiqueIssue(Fingerprinted):
    issue_id: str = Field(pattern=ID)
    origin: Literal["deterministic", "llm"]
    category: Literal["scope", "provenance", "methodology", "statistics", "bibliography", "claims"]
    severity: Literal["info", "minor", "major", "critical"]
    basis: Literal["verified_check", "potential_concern"]
    finding: str = Field(min_length=8, max_length=1400)
    recommendation: str = Field(min_length=8, max_length=1400)
    evidence_refs: list[str] = Field(min_length=1, max_length=12)
    requires_human_review: bool

    @model_validator(mode="after")
    def opinion_is_not_verified(self):
        if self.origin == "llm" and (self.basis != "potential_concern" or not self.requires_human_review):
            raise ValueError("LLM opinions must remain unverified concerns for human review")
        return self


class ClaimAssessment(StrictModel):
    claim_id: str = Field(pattern=ID)
    status: Literal["supported_numeric", "hypothesis_only", "unsupported", "outside_scientific_scope"]
    reason: str


class ReviewReport(Fingerprinted):
    schema_version: Literal["8.6"] = "8.6"
    kind: Literal["scientific_critic_report"] = "scientific_critic_report"
    bundle_sha256: str = Field(pattern=HEX_SHA)
    analysis_sha256: str = Field(pattern=HEX_SHA)
    observation_refs: list[str]
    hypothesis_refs: list[str]
    issues: list[CritiqueIssue]
    claim_assessments: list[ClaimAssessment]
    conclusion: Literal["Engineering or pedagogical review only; no scientific claims justified"] = (
        "Engineering or pedagogical review only; no scientific claims justified")
    scientific_evidence: Literal[False] = False
    ready_for_scientific_inference: Literal[False] = False

    @model_validator(mode="after")
    def distinct_issues(self):
        ids = [issue.issue_id for issue in self.issues]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate critique issue identifiers")
        return self


class SuggestedIssue(StrictModel):
    category: Literal["scope", "provenance", "methodology", "statistics", "bibliography", "claims"]
    severity: Literal["info", "minor", "major"]
    finding: str = Field(min_length=8, max_length=1400)
    recommendation: str = Field(min_length=8, max_length=1400)
    evidence_refs: list[str] = Field(min_length=1, max_length=8)


class CriticSuggestions(StrictModel):
    issues: list[SuggestedIssue] = Field(max_length=8)


class ChangeJustification(StrictModel):
    issue_ids: list[str] = Field(min_length=1, max_length=12)
    rationale: str = Field(min_length=10, max_length=1400)


class RevisionDraft(StrictModel):
    candidate_design: dict[str, Any]
    justifications: dict[str, ChangeJustification] = Field(min_length=1, max_length=60)


class ProtocolChange(StrictModel):
    path: str = Field(pattern=POINTER)
    before: Any
    after: Any
    justification: ChangeJustification


class RevisionProposal(Fingerprinted):
    schema_version: Literal["8.6"] = "8.6"
    kind: Literal["prospective_protocol_revision"] = "prospective_protocol_revision"
    review_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
    cycle: int = Field(ge=1, le=2)
    bundle_sha256: str = Field(pattern=HEX_SHA)
    report_sha256: str = Field(pattern=HEX_SHA)
    parent_design_sha256: str = Field(pattern=HEX_SHA)
    previous_proposal_sha256: str | None = Field(default=None, pattern=HEX_SHA)
    candidate_design: ExperimentalDesign
    candidate_design_sha256: str = Field(pattern=HEX_SHA)
    changes: list[ProtocolChange] = Field(min_length=1, max_length=60)
    basis: Literal["design_review_only_unlinked_results"] = "design_review_only_unlinked_results"
    reviewed_after_results: Literal[True] = True
    execution_status: Literal["not_executed"] = "not_executed"


class LLMSettings(StrictModel):
    provider: Literal["openai", "anthropic"]
    model: str = Field(min_length=1, max_length=140)
    max_output_tokens: int = Field(default=4000, ge=256, le=6000)
    timeout_seconds: int = Field(default=60, ge=5, le=120)


class ReviewInput(Fingerprinted):
    review_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
    source_kind: SourceKind
    source_path: str = Field(min_length=1, max_length=1000)
    jobs_db: str | None = Field(default=None, max_length=1000)
    context_path: str | None = Field(default=None, max_length=1000)
    claims: list[Claim] = Field(default_factory=list, max_length=30)
    max_cycles: int = Field(default=1, ge=1, le=2)
    llm: LLMSettings | None = None
    assist_critique: bool = False
    assist_revision: bool = False

    @model_validator(mode="after")
    def explicit_llm_and_ledger(self):
        if (self.assist_critique or self.assist_revision) and self.llm is None:
            raise ValueError("LLM assistance requires explicit provider/model settings")
        if (self.source_kind == "verified_pilot_aggregates") != (self.jobs_db is not None):
            raise ValueError("Only pilot inputs require an existing jobs database")
        if self.assist_revision and self.context_path is None:
            raise ValueError("Protocol revision requires an approved design context")
        return self


class LLMRequest(Fingerprinted):
    review_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
    role: Literal["critique", "revision"]
    cycle: int = Field(ge=1, le=2)
    bundle_sha256: str = Field(pattern=HEX_SHA)
    settings: LLMSettings
    prompt: str = Field(min_length=1, max_length=131_072)
    instructions: str = Field(min_length=1, max_length=24_000)
    prompt_version: Literal["v86-1"] = "v86-1"

    @model_validator(mode="after")
    def bounded_request(self):
        if self.role == "critique" and self.cycle != 1:
            raise ValueError("Only one critic call is allowed per review")
        if len((self.prompt + self.instructions).encode("utf-8")) > 131_072:
            raise ValueError("LLM input exceeds 128 KiB")
        return self

    @property
    def call_id(self) -> str:
        return f"{self.review_id}-{self.role}-{self.cycle}"


class HumanDecision(Fingerprinted):
    decision: Literal["approve", "reject", "request_changes", "skip"]
    target_sha256: str = Field(pattern=HEX_SHA)
    actor: str = Field(min_length=1, max_length=100)
    comment: str = Field(default="", max_length=2000)


class RevisionSubmission(StrictModel):
    decision: Literal["propose", "skip"]
    target_sha256: str = Field(pattern=HEX_SHA)
    actor: str = Field(min_length=1, max_length=100)
    draft: RevisionDraft | None = None

    @model_validator(mode="after")
    def explicit_draft(self):
        if (self.decision == "propose") != (self.draft is not None):
            raise ValueError("A proposal submission requires exactly one revision draft")
        return self
