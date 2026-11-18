"""Rebuild evidence through V8.5, never trust a standalone analysis export."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from analysis.ingest import ingest_verified_pilot
from analysis.pipeline import build_fixture_analysis, build_verified_pilot_analysis
from analysis.retry_pilot import RetryPilotAnalysis, SOURCE_KIND, analyze_retry_pilot
from analysis.stat_contracts import AnalysisReport, PedagogicalPairedFixture
from execution.campaign import PilotProtocol
from execution.job_manager import DurableJobManager
from experiments.retry_pilot import PilotCorpus
from experiments.retry_pilot_runner import PilotJournal
from research.critique.contracts import Claim, DesignContext, EvidenceBundle, ReviewInput
from research.experimental_design import (
    ExperimentalDesign, validate_approved_literature, validate_design_against_inputs,
)
from research.literature import digest_json


def read_document(path: str | Path) -> dict[str, Any]:
    """Bounded local JSON, including PowerShell UTF-8 BOM; no URL/UNC fetches."""
    text_path = str(path)
    if text_path.startswith(("\\\\", "//")) or "://" in text_path:
        raise ValueError("Only explicit local file paths are supported")
    source = Path(path)
    with source.open("rb") as src:
        content = src.read(262_145)
    if len(content) > 262_144:
        raise ValueError("Input JSON exceeds 256 KiB")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result

    def invalid_number(_):
        raise ValueError("Non-finite JSON numbers are forbidden")

    document = json.loads(content.decode("utf-8-sig"), object_pairs_hook=pairs,
                          parse_constant=invalid_number)
    if not isinstance(document, dict):
        raise ValueError("Expected a JSON object")
    return document


def validate_context(context: DesignContext) -> ExperimentalDesign:
    """Check V8.1–V8.3 content lineage, without importing the legacy canary."""
    mission, _, literature = validate_approved_literature(
        context.mission_document, context.literature_document)
    doc = context.design_document
    if (doc.get("schema_version") != "8.3" or doc.get("status") != "approved"
            or doc.get("execution_status") != "not_executed"):
        raise ValueError("An approved, unexecuted V8.3 design export is required")
    design = ExperimentalDesign.model_validate(doc.get("design"))
    if design.model_dump(mode="json") != doc["design"] or digest_json(doc["design"]) != doc.get("design_sha256"):
        raise ValueError("Parent design content/digest mismatch")
    if (doc.get("mission_sha256") != context.mission_document["mission_sha256"]
            or doc.get("literature_sha256") != digest_json(context.literature_document)):
        raise ValueError("Parent design input lineage mismatch")
    if (doc.get("planned_cells") != design.design_cells
            or doc.get("planned_evaluations") != design.total_evaluations
            or doc.get("planned_worst_case_llm_calls") != design.worst_case_llm_calls):
        raise ValueError("Parent design planning counts mismatch")
    expected_matrix = {
        "schema_version": "8.3", "design_sha256": doc["design_sha256"],
        "execution_status": "not_executed", "cells": design.factor_matrix(),
        "tasks_per_cell": design.resource_limits.tasks_per_cell,
        "seeds": design.resource_limits.seeds,
    }
    if context.matrix_document != expected_matrix:
        raise ValueError("Parent design matrix mismatch")
    validate_design_against_inputs(design, mission, literature)
    return design


EvidenceLedger = DurableJobManager | PilotJournal


def build_evidence(source_kind: str, source_document: dict, *,
                   manager: EvidenceLedger | None = None,
                   context: DesignContext | None = None,
                   claims: list[Claim] | None = None,
                   expected_analysis: AnalysisReport | RetryPilotAnalysis | None = None) -> EvidenceBundle:
    if context is not None:
        context = DesignContext.model_validate(context.model_dump(mode="json"))
        validate_context(context)
    dataset = quality = None
    if source_kind == "pedagogical_paired_fixture":
        fixture = PedagogicalPairedFixture.model_validate(source_document)
        source = fixture.model_dump(mode="json")
        report = build_fixture_analysis(fixture)
    elif source_kind == "verified_pilot_aggregates":
        if not isinstance(manager, DurableJobManager) or not manager.read_only:
            raise PermissionError("Critic requires an existing read-only SQLite job ledger")
        protocol = PilotProtocol.model_validate(source_document)
        source = protocol.model_dump(mode="json")
        normalized, assessment = ingest_verified_pilot(protocol, manager)
        dataset = normalized.model_dump(mode="json")
        quality = assessment.model_dump(mode="json")
        report = build_verified_pilot_analysis(protocol, manager)
    elif source_kind == SOURCE_KIND:
        if not isinstance(manager, PilotJournal):
            raise PermissionError("V9 critique requires its original retry journal")
        corpus = PilotCorpus.model_validate(source_document)
        source = corpus.model_dump(mode="json")
        report = analyze_retry_pilot(corpus, manager)
    else:
        raise ValueError("Unsupported evidence source kind")
    if expected_analysis is not None and expected_analysis != report:
        raise ValueError("Analysis does not match recomputed source results")
    return EvidenceBundle(
        schema_version="9.0" if source_kind == SOURCE_KIND else "8.6",
        source_kind=source_kind, source_document=source, analysis=report,
        dataset_document=dataset, quality_document=quality, context=context,
        claims=claims or [], design_execution_link="unavailable" if context else "not_applicable")


def verify_evidence(bundle: EvidenceBundle, manager: EvidenceLedger | None = None) -> None:
    actual = build_evidence(bundle.source_kind, bundle.source_document,
                            manager=manager, context=bundle.context, claims=bundle.claims)
    if actual != bundle:
        raise ValueError("Evidence differs from recomputed source/provenance")


def load_input(request: ReviewInput) -> EvidenceBundle:
    """Read current source files on start AND after every sensitive interruption."""
    context = (DesignContext.model_validate(read_document(request.context_path))
               if request.context_path else None)
    return build_evidence(request.source_kind, read_document(request.source_path),
                          manager=open_evidence_ledger(request), context=context, claims=request.claims)


def open_evidence_ledger(request: ReviewInput) -> EvidenceLedger | None:
    if request.jobs_db is None:
        return None
    if request.jobs_db.startswith(("\\\\", "//")) or "://" in request.jobs_db:
        raise ValueError("Only a local existing SQLite ledger is supported")
    if request.source_kind == SOURCE_KIND:
        return PilotJournal(Path(request.jobs_db))
    return DurableJobManager(path=request.jobs_db, read_only=True)


def resolve_ref(bundle: EvidenceBundle, ref: str) -> Any:
    """Resolve an RFC-6901-style reference within this bundle; never open files."""
    if not isinstance(ref, str) or not ref.startswith("/") or len(ref) > 301:
        raise ValueError("Invalid evidence reference")
    current: Any = bundle.model_dump(mode="json")
    try:
        for encoded in ref[1:].split("/"):
            # Reject malformed escapes rather than silently interpreting another key.
            import re
            if re.search(r"~(?![01])", encoded):
                raise ValueError("Invalid pointer escape")
            key = encoded.replace("~1", "/").replace("~0", "~")
            if isinstance(current, list):
                if not key.isascii() or not key.isdigit() or (len(key) > 1 and key.startswith("0")):
                    raise ValueError("Invalid list reference")
                current = current[int(key)]
            elif isinstance(current, dict):
                current = current[key]
            else:
                raise ValueError("Evidence reference traverses a scalar")
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Evidence reference not found in verified bundle") from error
    return current
