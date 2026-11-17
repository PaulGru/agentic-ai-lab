"""V8.4.5 — LangGraph HITL workflow around a durable execution ledger.

No user-provided Python/shell, no LLMs, no HPC network. The workflow commits
an intent before any side effect; uncertain submission outcomes require manual
reconciliation instead of automatic retries.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from typing_extensions import TypedDict

from execution.contracts import ExperimentJob
from execution.job_manager import DurableJobManager
from execution.backends.slurm_sim import render_sbatch_preview
from execution.workloads import render_source


class ExecutionState(TypedDict, total=False):
    job_document: dict[str, Any]
    job_spec_sha256: str
    backend: str
    approved: bool
    approval_sha256: str | None
    job_id: str
    status: str
    message: str


class ExecutionWorkflow:
    def __init__(self, manager: DurableJobManager, *, backend: str):
        if backend not in {"slurm-sim", "docker-local"}:
            raise ValueError("Only offline Slurm simulation and local Docker are supported")
        self.manager = manager
        self.backend = backend

    def validate_node(self, state: ExecutionState):
        job = ExperimentJob.model_validate(state["job_document"])
        if self.backend == "slurm-sim":
            render_sbatch_preview(job)
        else:
            render_source(job)
        return {
            "job_document": job.model_dump(mode="json"),
            "job_spec_sha256": job.fingerprint(),
            "job_id": job.job_id,
            "backend": self.backend,
            "approved": False,
            "approval_sha256": None,
            "status": "awaiting_approval",
            "message": "Review job specification before submission",
        }

    @staticmethod
    def approval_node(state: ExecutionState):
        # No external effect before interrupt.
        job = ExperimentJob.model_validate(state["job_document"])
        digest = state["job_spec_sha256"]
        response = interrupt({
            "kind": "v845_job_approval",
            "job_id": job.job_id,
            "backend": state["backend"],
            "workload": job.workload.name,
            "resources": job.resources.model_dump(mode="json"),
            "sha256": digest,
            "simulated": state["backend"] == "slurm-sim",
        })
        ok = (
            isinstance(response, dict)
            and response.get("decision") == "approve"
            and response.get("sha256") == digest
            and response.get("backend") == state["backend"]
        )
        return {
            "approved": ok, "approval_sha256": digest if ok else None,
            "status": "approved" if ok else "rejected",
            "message": "Execution approved" if ok else "Execution rejected",
        }

    @staticmethod
    def after_approval(state: ExecutionState):
        return "submit" if state.get("approved") is True else END

    def submit_node(self, state: ExecutionState):
        job = ExperimentJob.model_validate(state["job_document"])
        if (
            state.get("approved") is not True
            or state.get("approval_sha256") != job.fingerprint()
            or state.get("job_spec_sha256") != job.fingerprint()
            or state.get("backend") != self.backend
        ):
            raise PermissionError("Submission has no approval for the exact job/backend")
        record = self.manager.submit(job)
        return {"status": record.phase, "message": record.message}

    def observe_node(self, state: ExecutionState):
        record = self.manager.refresh(state["job_id"])
        return {"status": record.phase, "message": record.message}

    def build(self, saver):
        builder = StateGraph(ExecutionState)
        builder.add_node("validate", self.validate_node)
        builder.add_node("approval", self.approval_node)
        builder.add_node("submit", self.submit_node)
        builder.add_node("observe", self.observe_node)
        builder.add_edge(START, "validate")
        builder.add_edge("validate", "approval")
        builder.add_conditional_edges(
            "approval", self.after_approval, {"submit": "submit", END: END},
        )
        builder.add_edge("submit", "observe")
        builder.add_edge("observe", END)
        return builder.compile(checkpointer=saver)


def pending_approval(snapshot):
    for task in snapshot.tasks:
        for item in task.interrupts:
            value = item.value
            if isinstance(value, dict) and value.get("kind") == "v845_job_approval":
                return value
    return None
