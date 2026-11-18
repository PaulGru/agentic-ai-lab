"""V8.9: resume a bounded research workflow using V8.3–V8.8 contracts."""
from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Literal

from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt
from pydantic import Field
from typing_extensions import TypedDict

from analysis.contracts import fingerprint
from analysis.pipeline import export_report
from execution.backends.docker_transport import DockerOutcome
from execution.backends.local import LocalDockerBackend
from execution.campaign import PilotProtocol, PilotRun, save_report
from execution.campaign_workflow import PilotWorkflow
from execution.contracts import StrictModel
from execution.job_manager import DurableJobManager
from execution.workloads import validate_workload
from research.critique.contracts import DesignContext, Fingerprinted, HumanDecision, LLMRequest, LLMSettings, ReviewInput
from research.critique.evidence import read_document, validate_context
from research.critique.graph import ReviewState, ReviewWorkflow
from research.critique.reporting import export_review
from research.critique.store import ReviewStore
from research.manuscript import (
    build_manuscript, export_manuscript, make_editorial_request,
    assist_manuscript, record_manuscript, reconstruct_manuscript, verified_manuscript,
)
from research.memory.contracts import MemoryContext, SourceOrigin


class PilotBindings(StrictModel):
    cells: list[PilotRun] = Field(min_length=2, max_length=4)


class WorkflowInput(Fingerprinted):
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
    context_path: str = Field(min_length=1, max_length=1000)
    bindings: PilotBindings
    critic_settings: LLMSettings | None = None
    assist_revision: bool = False
    writer_settings: LLMSettings | None = None


def make_plan(request: WorkflowInput):
    context = DesignContext.model_validate(read_document(request.context_path))
    validate_context(context)
    cells = context.matrix_document["cells"]
    by_cell = {entry.run_id: entry for entry in request.bindings.cells}
    if len(by_cell) != len(request.bindings.cells) or set(by_cell) != {cell["cell_id"] for cell in cells}:
        raise ValueError("Provide one explicit synthetic workload binding for every approved design cell")
    protocol = PilotProtocol(campaign_id=request.run_id,
                             description="Synthetic exercise bound to design cell identifiers",
                             runs=[by_cell[cell["cell_id"]] for cell in cells])
    jobs = protocol.jobs()
    return context, protocol, {
        "schema_version": "8.9", "context_sha256": context.fingerprint(),
        "design_sha256": context.design_document["design_sha256"],
        "input_sha256": request.fingerprint(),
        "mode": "in_process_simulation", "scientific_evidence": False,
        "protocol_sha256": protocol.fingerprint(),
        "cells": [{**cell, "job": job.model_dump(mode="json"), "job_sha256": job.fingerprint()}
                  for cell, job in zip(cells, jobs, strict=True)],
    }


class ArithmeticSimulation:
    """Emit the existing transport envelope with explicitly simulated receipts."""
    simulated = True

    def run(self, job, code, cancel):
        values = validate_workload(job)["values"]
        if cancel.is_set():
            return DockerOutcome(status="cancelled", exit_code=None, stdout="", stderr="", duration_seconds=0)
        digest = job.fingerprint()
        result = {
            "kind": "agentic-v842-artifacts", "sha256": digest,
            "files": {
                "metrics.json": {"count": len(values), "mean": statistics.fmean(values),
                                 "minimum": min(values), "maximum": max(values)},
                "run_metadata.json": {"job_id": job.job_id, "job_sha256": digest, "seed": job.seed,
                                      "workload": job.workload.name, "execution": "docker_sandbox",
                                      "python_version": "simulation"},
            },
        }
        return DockerOutcome(status="success", exit_code=0, stdout=json.dumps(result), stderr="", duration_seconds=0)


class WorkflowState(ReviewState, total=False):
    workflow_input: dict
    workflow_plan_sha256: str
    workflow_status: str
    writer_request: dict
    writer_authorization: dict
    manuscript_sha256: str
    assisted_manuscript: bool


def config_for(run_id):
    return {"configurable": {"thread_id": f"v89-research:{run_id}"}, "recursion_limit": 64}


def pending_decision(snapshot):
    return next((item.value for task in snapshot.tasks for item in task.interrupts), None)


class ResearchWorkflow:
    def __init__(self, root, memory, *, provider_factory=None, transport=None, read_only=False):
        self.root = Path(root).resolve()
        self.memory = memory
        self.store = ReviewStore(self.root / "review.sqlite3", read_only=read_only)
        self.transport = transport
        self.provider_options = {"provider_factory": provider_factory} if provider_factory is not None else {}

    def _request(self, state):
        return WorkflowInput.model_validate(state["workflow_input"])

    def _plan(self, state):
        request = self._request(state)
        context, protocol, plan = make_plan(request)
        plan["memory_binding"] = self.memory.binding.model_dump(mode="json")
        return request, context, protocol, plan

    def _checked(self, state):
        request, context, protocol, plan = self._plan(state)
        if (self.store.require("workflow_plan", request.run_id) != plan
                or state["workflow_plan_sha256"] != fingerprint(plan)):
            raise ValueError("Workflow plan or verified sources changed")
        return request, context, protocol, plan

    def plan_node(self, state):
        request, _, _, plan = self._plan(state)
        if request.assist_revision and request.critic_settings is None:
            raise ValueError("Revision assistance requires explicit critic model settings")
        digest = self.store.put("workflow_plan", request.run_id, request.run_id, plan)
        return {"workflow_plan_sha256": digest, "workflow_status": "awaiting_execution_approval"}

    def execution_approval_node(self, state):
        _, _, _, plan = self._checked(state)
        decision = HumanDecision.model_validate(interrupt({
            "kind": "v89_execution_approval", "target_sha256": state["workflow_plan_sha256"],
            "plan": plan,
        }))
        request, _, _, _ = self._checked(state)
        self._record_decision(request.run_id, "execution", state["workflow_plan_sha256"], decision)
        return {"workflow_status": "execution_approved" if decision.decision == "approve" else "execution_rejected"}

    def _record_decision(self, run_id, role, digest, decision):
        if decision.decision not in {"approve", "reject"} or decision.target_sha256 != digest:
            raise PermissionError("Decision does not target this exact action")
        self.store.put("decision", f"{run_id}-v89-{role}", run_id,
                       {"kind": f"v89_{role}_decision", **decision.model_dump(mode="json")})

    def execute_node(self, state):
        request, _, protocol, _ = self._checked(state)
        decision = self.store.require("decision", f"{request.run_id}-v89-execution")
        if (decision["kind"] != "v89_execution_decision" or decision["decision"] != "approve"
                or decision["target_sha256"] != state["workflow_plan_sha256"]):
            raise PermissionError("Exact workflow execution approval required")
        transport = self.transport if self.transport is not None else ArithmeticSimulation()
        if not transport.simulated:
            raise PermissionError("V8.9 only supports the simulated pilot transport")
        manager = DurableJobManager(LocalDockerBackend(execution_authorized=True, transport=transport),
                                    self.root / "jobs.sqlite3")
        for job in protocol.jobs():
            record = manager.get(job.job_id)
            if record and record.handle is not None and not record.handle.simulated:
                raise ValueError("Existing job is outside the approved simulation mode")
        save_report(protocol.model_dump(mode="json"), self.root / "pilot_protocol.json")
        # Reuse the existing bounded executor, after the workflow's own exact HITL.
        result = PilotWorkflow(manager).execute_node({
            "protocol_document": protocol.model_dump(mode="json"), "protocol_sha256": protocol.fingerprint(),
            "approval_sha256": protocol.fingerprint(), "approved": True, "backend": "docker-local",
        })
        if result["status"] != "jobs_succeeded":
            return {"workflow_status": "execution_blocked"}
        review = self._review_input(request)
        return {"input_document": review.model_dump(mode="json"), "workflow_status": "reviewing"}

    def _review_input(self, request):
        return ReviewInput(review_id=request.run_id, source_kind="verified_pilot_aggregates",
                             source_path=str(self.root / "pilot_protocol.json"), jobs_db=str(self.root / "jobs.sqlite3"),
                             context_path=request.context_path, llm=request.critic_settings,
                             assist_critique=request.critic_settings is not None, assist_revision=request.assist_revision,
                             max_cycles=2)

    def seed_memory_node(self, state):
        request, _, _, _ = self._checked(state)
        self.memory.memory.ingest(self.memory.binding.project_id, SourceOrigin(kind="context", path=request.context_path))
        return {}

    def writer_node(self, state):
        request, _, protocol, _ = self._checked(state)
        if (self.store.require("session", request.run_id) != self._review_input(request).model_dump(mode="json")
                or self.store.require("bundle", request.run_id)["source_document"] != protocol.model_dump(mode="json")):
            raise ValueError("Review does not match the approved jobs")
        manuscript = build_manuscript(self.store, request.run_id)
        if request.writer_settings:
            editorial = make_editorial_request(self.store, manuscript, request.writer_settings)
            return {"writer_request": editorial.model_dump(mode="json"), "workflow_status": "awaiting_writer_api"}
        return {"workflow_status": "awaiting_manuscript", "manuscript_sha256": manuscript.fingerprint(),
                "assisted_manuscript": False}

    def _writer_request(self, state):
        request, _, _, _ = self._checked(state)
        manuscript = build_manuscript(self.store, request.run_id)
        editorial = LLMRequest.model_validate(state["writer_request"])
        if request.writer_settings is None or editorial != make_editorial_request(self.store, manuscript, request.writer_settings):
            raise PermissionError("Writer request changed")
        context = self.store.get("memory_context", request.run_id)
        if context is not None:
            self.memory.verify(MemoryContext.model_validate(context))
        return manuscript, editorial

    def writer_approval_node(self, state):
        manuscript, request = self._writer_request(state)
        decision = HumanDecision.model_validate(interrupt({
            "kind": "v89_writer_approval", "target_sha256": request.fingerprint(),
            "request": request.model_dump(mode="json"),
        }))
        manuscript, request = self._writer_request(state)
        self._record_decision(request.review_id, "writer", request.fingerprint(), decision)
        return {"writer_authorization": decision.model_dump(mode="json"), "assisted_manuscript": False,
                "manuscript_sha256": manuscript.fingerprint(),
                "workflow_status": "writer_authorized" if decision.decision == "approve" else "awaiting_manuscript"}

    def writer_call_node(self, state):
        manuscript, request = self._writer_request(state)
        decision = HumanDecision.model_validate(state["writer_authorization"])
        record = self.store.require("decision", f"{request.review_id}-v89-writer")
        if record != {"kind": "v89_writer_decision", **decision.model_dump(mode="json")}:
            raise PermissionError("Writer authorization differs from the durable decision")
        manuscript = assist_manuscript(self.store, manuscript, request, decision, **self.provider_options)
        return {"workflow_status": "awaiting_manuscript", "manuscript_sha256": manuscript.fingerprint(),
                "assisted_manuscript": True}

    def _manuscript(self, state):
        request, _, _, _ = self._checked(state)
        manuscript = reconstruct_manuscript(self.store, request.run_id, assisted=state["assisted_manuscript"])
        if manuscript.fingerprint() != state["manuscript_sha256"]:
            raise ValueError("Manuscript changed after preparation")
        return manuscript

    def manuscript_approval_node(self, state):
        manuscript = self._manuscript(state)
        decision = HumanDecision.model_validate(interrupt({
            "kind": "v89_manuscript_approval", "target_sha256": manuscript.fingerprint(),
            "manuscript": manuscript.model_dump(mode="json"),
        }))
        manuscript = self._manuscript(state)
        record_manuscript(self.store, manuscript, decision)
        return {"workflow_status": "manuscript_approved" if decision.decision == "approve" else "manuscript_rejected"}

    def export_node(self, state):
        request, _, protocol, plan = self._checked(state)
        report = self.root / "output"
        manuscript = verified_manuscript(self.store, request.run_id, state["manuscript_sha256"])
        export_manuscript(self.store, request.run_id, manuscript.fingerprint(), report / "paper")
        export_review(self.store, request.run_id, report / "review")
        save_report(plan, report / "plan.json")
        from analysis.stat_contracts import AnalysisReport
        bundle = self.store.require("bundle", request.run_id)
        export_report(AnalysisReport.model_validate(bundle["analysis"]), report / "analysis")
        ledger = DurableJobManager(path=self.root / "jobs.sqlite3", read_only=True)
        context = MemoryContext.model_validate(self.store.require("memory_context", request.run_id))
        calls = self.store.call_summaries(request.run_id)
        jobs = [ledger.get(job.job_id) for job in protocol.jobs()]
        save_report({
            "schema_version": "8.9", "mode": "in_process_simulation", "scientific_evidence": False,
            "jobs": [{"job_id": job.job_id, "phase": job.phase, "simulated": job.artifacts.simulated} for job in jobs],
            "analysis_sha256": manuscript.analysis_sha256, "manuscript_sha256": manuscript.fingerprint(),
            "memory_context_sha256": context.fingerprint(), "memory_hits": len(context.hits),
            "memory_indexed_items": state["memory_indexed_items"],
            "claims_with_references": len(manuscript.statements), "citations": len(manuscript.bibliography),
            "llm_calls": calls, "llm_calls_count": len(calls), "actual_llm_cost": None if calls else 0,
            "human_decisions": [{"artifact_id": a["artifact_id"], "sha256": a["sha256"],
                                 "kind": a["document"]["kind"], "decision": a["document"]["decision"]}
                                for a in self.store.artifacts(request.run_id) if a["kind"] == "decision"],
        }, report / "evaluation.json")
        return {"workflow_status": "completed"}

    def build(self, saver):
        graph = StateGraph(WorkflowState)
        graph.add_node("plan", self.plan_node)
        graph.add_node("execution_approval", self.execution_approval_node)
        graph.add_node("execute", self.execute_node)
        graph.add_node("seed_memory", self.seed_memory_node)
        graph.add_node("critic", ReviewWorkflow(self.store, memory=self.memory, **self.provider_options).build(None))
        graph.add_node("writer", self.writer_node)
        graph.add_node("writer_approval", self.writer_approval_node)
        graph.add_node("writer_call", self.writer_call_node)
        graph.add_node("manuscript_approval", self.manuscript_approval_node)
        graph.add_node("export", self.export_node)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "execution_approval")
        graph.add_conditional_edges("execution_approval", lambda s: "execute" if s["workflow_status"] == "execution_approved" else END)
        graph.add_conditional_edges("execute", lambda s: "seed_memory" if s["workflow_status"] == "reviewing" else END)
        graph.add_edge("seed_memory", "critic")
        graph.add_edge("critic", "writer")
        graph.add_conditional_edges("writer", lambda s: "writer_approval" if s["workflow_status"] == "awaiting_writer_api" else "manuscript_approval")
        graph.add_conditional_edges("writer_approval", lambda s: "writer_call" if s["workflow_status"] == "writer_authorized" else "manuscript_approval")
        graph.add_edge("writer_call", "manuscript_approval")
        graph.add_conditional_edges("manuscript_approval", lambda s: "export" if s["workflow_status"] == "manuscript_approved" else END)
        graph.add_edge("export", END)
        return graph.compile(checkpointer=saver)
