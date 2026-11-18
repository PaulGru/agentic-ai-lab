"""Prospective changes to V8.3 protocols; never edit measurements or execute jobs."""
from __future__ import annotations

from execution.job_manager import DurableJobManager
from research.critique.contracts import EvidenceBundle, ProtocolChange, RevisionDraft, RevisionProposal, ReviewReport
from research.critique.evidence import validate_context
from research.critique.rules import verify_review
from research.experimental_design import ExperimentalDesign, validate_approved_literature, validate_design_against_inputs
from research.literature import digest_json

EDITABLE = {
    "hypothesis_tests", "factors", "baselines", "task_suite", "metrics", "controlled_variables",
    "perturbation_and_recovery_procedure", "randomization_and_counterbalancing", "analysis_plan",
    "resource_limits", "literature_links", "feasibility_checks", "limitations",
}


def design_diff(before, after, path="") -> list[tuple[str, object, object]]:
    """Leaf differences for equal-key dictionaries; lists and changed-key maps are atomic."""
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict) and before.keys() == after.keys():
        result = []
        for key in sorted(before):
            escaped = key.replace("~", "~0").replace("/", "~1")
            result.extend(design_diff(before[key], after[key], f"{path}/{escaped}"))
        return result
    return [(path, before, after)]


def build_revision(bundle: EvidenceBundle, report: ReviewReport, draft: RevisionDraft, *,
                   review_id: str, cycle: int = 1, previous_proposal_sha256: str | None = None,
                   manager: DurableJobManager | None = None) -> RevisionProposal:
    verify_review(bundle, report, manager)
    if bundle.context is None:
        raise ValueError("Revision requires a verified V8.3 design context")
    parent = validate_context(bundle.context)
    mission, _, literature = validate_approved_literature(
        bundle.context.mission_document, bundle.context.literature_document)
    candidate = ExperimentalDesign.model_validate(draft.candidate_design)
    if candidate.model_dump(mode="json") != draft.candidate_design:
        raise ValueError("Candidate must use canonical V8.3 field types")
    validate_design_against_inputs(candidate, mission, literature)
    changes = design_diff(parent.model_dump(mode="json"), candidate.model_dump(mode="json"))
    if not changes:
        raise ValueError("Revision proposes no change")
    if set(draft.justifications) != {p for p, _, _ in changes}:
        raise ValueError("Every changed path requires exactly one justification")
    known = {issue.issue_id for issue in report.issues}
    result = []
    for path, old, new in changes:
        if path.split("/")[1] not in EDITABLE:
            raise ValueError("Changing the objective/title or scope requires a new mission/design process")
        justification = draft.justifications[path]
        if len(justification.issue_ids) != len(set(justification.issue_ids)) or not set(justification.issue_ids) <= known:
            raise ValueError("Revision refers to unknown or duplicate critique issues")
        result.append(ProtocolChange(path=path, before=old, after=new, justification=justification))
    if (cycle == 1) != (previous_proposal_sha256 is None):
        raise ValueError("Revision cycle must preserve the previous proposal link")
    return RevisionProposal(
        review_id=review_id, cycle=cycle, bundle_sha256=bundle.fingerprint(), report_sha256=report.fingerprint(),
        parent_design_sha256=bundle.context.design_document["design_sha256"],
        previous_proposal_sha256=previous_proposal_sha256,
        candidate_design=candidate, candidate_design_sha256=digest_json(candidate.model_dump(mode="json")),
        changes=result)


def verify_revision(bundle: EvidenceBundle, report: ReviewReport, proposal: RevisionProposal,
                    manager: DurableJobManager | None = None) -> None:
    draft = RevisionDraft(candidate_design=proposal.candidate_design.model_dump(mode="json"),
                          justifications={change.path: change.justification for change in proposal.changes})
    actual = build_revision(bundle, report, draft, review_id=proposal.review_id, cycle=proposal.cycle,
                            previous_proposal_sha256=proposal.previous_proposal_sha256, manager=manager)
    if actual != proposal:
        raise ValueError("Revision differs from validated parent, diff or provenance")


def revision_matrix(proposal: RevisionProposal) -> dict:
    design = proposal.candidate_design
    return {"schema_version": "8.6", "execution_status": "not_executed",
            "candidate_design_sha256": proposal.candidate_design_sha256,
            "parent_design_sha256": proposal.parent_design_sha256,
            "reviewed_after_results": True, "cells": design.factor_matrix(),
            "tasks_per_cell": design.resource_limits.tasks_per_cell, "seeds": design.resource_limits.seeds}
