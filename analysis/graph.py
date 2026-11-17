"""V8.5.5 LangGraph analysis DAG: read-only, deterministic, no HITL required.

The graph accepts either a verified ledger-backed V8.4.6 protocol or a
pedagogical fixture. Supplying the ledger manager is a requirement for the
first case; the graph will NEVER interpret raw JSON as verified provenance.
"""
from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from analysis.contracts import AnalysisDataset, QualityAssessment
from analysis.descriptive import describe_paired_fixture, describe_verified_pilot
from analysis.inference import infer_paired_fixture
from analysis.ingest import ingest_verified_pilot
from analysis.pipeline import build_fixture_analysis, build_verified_pilot_analysis
from analysis.stat_contracts import (
    AnalysisReport, DescriptiveResult, PedagogicalPairedFixture, ValidityReport,
)
from analysis.validity import assess_paired_fixture, assess_verified_pilot
from execution.campaign import PilotProtocol
from execution.job_manager import DurableJobManager


class AnalysisState(TypedDict, total=False):
    source_kind: Literal["verified_pilot_aggregates", "pedagogical_paired_fixture"]
    protocol_document: dict[str, Any]
    fixture_document: dict[str, Any]
    dataset_document: dict[str, Any]
    quality_document: dict[str, Any]
    validity_document: dict[str, Any]
    descriptive_document: dict[str, Any]
    inference_document: dict[str, Any] | None
    report_document: dict[str, Any]
    status: str


class AnalysisWorkflow:
    def __init__(self, *, manager: DurableJobManager | None = None):
        self.manager = manager

    def load_node(self, state: AnalysisState):
        kind = state.get("source_kind")
        if kind == "verified_pilot_aggregates":
            if self.manager is None:
                raise ValueError("Verified pilot requires a SQLite ledger manager")
            protocol = PilotProtocol.model_validate(state["protocol_document"])
            dataset, quality = ingest_verified_pilot(protocol, self.manager)
            return {
                "protocol_document": protocol.model_dump(mode="json"),
                "dataset_document": dataset.model_dump(mode="json"),
                "quality_document": quality.model_dump(mode="json"),
                "status": "verified_source_loaded",
            }
        if kind == "pedagogical_paired_fixture":
            fixture = PedagogicalPairedFixture.model_validate(state["fixture_document"])
            return {"fixture_document": fixture.model_dump(mode="json"),
                    "status": "synthetic_fixture_loaded"}
        raise ValueError(f"Unsupported analysis source_kind: {kind}")

    @staticmethod
    def _validated_inputs(state: AnalysisState):
        if state["source_kind"] == "verified_pilot_aggregates":
            dataset = AnalysisDataset.model_validate(state["dataset_document"])
            quality = QualityAssessment.model_validate(state["quality_document"])
            return dataset, quality
        return PedagogicalPairedFixture.model_validate(state["fixture_document"]), None

    def validity_node(self, state: AnalysisState):
        source, quality = self._validated_inputs(state)
        gate = (assess_verified_pilot(source, quality) if quality is not None
                else assess_paired_fixture(source))
        return {"validity_document": gate.model_dump(mode="json"), "status": "validity_checked"}

    def descriptive_node(self, state: AnalysisState):
        source, quality = self._validated_inputs(state)
        summary = (describe_verified_pilot(source, quality) if quality is not None
                   else describe_paired_fixture(source))
        return {"descriptive_document": summary.model_dump(mode="json"), "status": "described"}

    def inference_node(self, state: AnalysisState):
        source, quality = self._validated_inputs(state)
        gate = ValidityReport.model_validate(state["validity_document"])
        if quality is not None or not gate.ready_for_pedagogical_inference:
            return {"inference_document": None, "status": "inference_blocked"}
        result = infer_paired_fixture(source, validity=gate)
        return {"inference_document": result.model_dump(mode="json"), "status": "pedagogical_inference_done"}

    def report_node(self, state: AnalysisState):
        # Re-derive everything from source instead of trusting mutable state JSON.
        # For a ledger-backed protocol, this RECHECKS V8.4.6 artifact checksums.
        if state["source_kind"] == "verified_pilot_aggregates":
            if self.manager is None:
                raise ValueError("Missing verified ledger")
            protocol = PilotProtocol.model_validate(state["protocol_document"])
            actual = build_verified_pilot_analysis(protocol, self.manager)
        else:
            actual = build_fixture_analysis(
                PedagogicalPairedFixture.model_validate(state["fixture_document"]))
        supplied_desc = DescriptiveResult.model_validate(state["descriptive_document"])
        supplied_gate = ValidityReport.model_validate(state["validity_document"])
        if supplied_desc != actual.descriptive or supplied_gate != actual.validity:
            raise ValueError("Analysis node result/source mismatch")
        if state.get("inference_document") != (
            actual.inference.model_dump(mode="json") if actual.inference else None
        ):
            raise ValueError("Analysis inference result/source mismatch")
        return {"report_document": actual.model_dump(mode="json"), "status": "completed"}

    def build(self, saver=None):
        builder = StateGraph(AnalysisState)
        for name, handler in (
            ("load", self.load_node),
            ("validity", self.validity_node),
            ("describe", self.descriptive_node),
            ("infer", self.inference_node),
            ("report", self.report_node),
        ):
            builder.add_node(name, handler)
        builder.add_edge(START, "load")
        builder.add_edge("load", "validity")
        builder.add_edge("validity", "describe")
        builder.add_edge("describe", "infer")
        builder.add_edge("infer", "report")
        builder.add_edge("report", END)
        return builder.compile(checkpointer=saver)
