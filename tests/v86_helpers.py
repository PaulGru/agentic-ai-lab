"""Fictional inputs only; no private brief, network, Docker or API credentials."""
from pathlib import Path

from analysis.stat_contracts import PedagogicalPairedFixture
from research.critique.contracts import DesignContext
from research.critique.evidence import build_evidence
from research.literature import digest_json
from tests.test_v83_design import approved_inputs, valid_design

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "v85_paired_fixture.json"


def fixture_document():
    return PedagogicalPairedFixture.model_validate_json(FIXTURE.read_text(encoding="utf-8")).model_dump(mode="json")


def design_context():
    mission, literature = approved_inputs()
    design = valid_design()
    digest = digest_json(design)
    from research.experimental_design import ExperimentalDesign
    parsed = ExperimentalDesign.model_validate(design)
    return DesignContext(
        mission_document=mission, literature_document=literature,
        design_document={
            "schema_version": "8.3", "status": "approved", "execution_status": "not_executed",
            "mission_sha256": mission["mission_sha256"], "literature_sha256": digest_json(literature),
            "design_sha256": digest, "design": design,
            "planned_cells": parsed.design_cells, "planned_evaluations": parsed.total_evaluations,
            "planned_worst_case_llm_calls": parsed.worst_case_llm_calls,
        },
        matrix_document={
            "schema_version": "8.3", "design_sha256": digest, "execution_status": "not_executed",
            "cells": parsed.factor_matrix(), "tasks_per_cell": parsed.resource_limits.tasks_per_cell,
            "seeds": parsed.resource_limits.seeds,
        })


def sample_bundle(context=True, claims=None):
    return build_evidence("pedagogical_paired_fixture", fixture_document(),
                          context=design_context() if context else None, claims=claims)


def revision_draft(bundle):
    from research.critique.contracts import RevisionDraft
    from copy import deepcopy
    candidate = deepcopy(bundle.context.design_document["design"])
    candidate["analysis_plan"]["power_limitation"] = (
        "Pilote exploratoire : l'indépendance des unités doit être justifiée avant toute étude de puissance.")
    return RevisionDraft.model_validate({"candidate_design": candidate, "justifications": {
        "/analysis_plan/power_limitation": {"issue_ids": ["assumed_independence"],
                                            "rationale": "Expliciter les hypothèses encore non vérifiées."}}})
