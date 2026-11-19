"""V8.4.6 LangGraph: batch HITL -> real local jobs -> durable report.

All Docker work is completed within the *resume* process because the V8.4.2
LocalDockerBackend holds its workers in memory. A process crash is ambiguous;
the V8.4.4 ledger deliberately blocks automatic re-submission.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict

from execution.campaign import CampaignBlocked, PilotProtocol, build_report, run_campaign
from execution.job_manager import DurableJobManager


class PilotState(TypedDict, total=False):
    protocol_document: dict[str, Any]
    protocol_sha256: str
    backend: str
    approved: bool
    approval_sha256: str | None
    status: str
    message: str
    analysis: dict[str, Any]
    simulated: bool


class PilotWorkflow:
    def __init__(self, manager: DurableJobManager, *, poll_seconds: float = 0.1):
        if type(manager.backend).__name__ != "LocalDockerBackend":
            raise ValueError("V8.4.6 requires LocalDockerBackend, not SlurmSimBackend")
        self.manager = manager
        self.poll_seconds = poll_seconds

    @staticmethod
    def validate_node(state: PilotState):
        protocol = PilotProtocol.model_validate(state["protocol_document"])
        jobs = protocol.jobs()  # validates workload/resource feasibility
        return {
            "protocol_document": protocol.model_dump(mode="json"),
            "protocol_sha256": protocol.fingerprint(),
            "backend": "docker-local",
            "approved": False,
            "approval_sha256": None,
            "status": "awaiting_approval",
            "message": f"{len(jobs)} synthetic jobs ready for operator review",
        }

    @staticmethod
    def approval_node(state: PilotState):
        protocol = PilotProtocol.model_validate(state["protocol_document"])
        digest = protocol.fingerprint()
        response = interrupt({
            "kind": "v846_campaign_approval",
            "campaign_id": protocol.campaign_id,
            "protocol_sha256": digest,
            "backend": "docker-local",
            "simulated": False,
            "jobs": [{"job_id": j.job_id, "sha256": j.fingerprint(),
                      "workload": j.workload.name, "resources": j.resources.model_dump(mode="json")}
                     for j in protocol.jobs()],
        })
        approved = (isinstance(response, dict) and response.get("decision") == "approve"
                    and response.get("protocol_sha256") == digest
                    and response.get("backend") == "docker-local")
        return {
            "approved": approved,
            "approval_sha256": digest if approved else None,
            "status": "approved" if approved else "rejected",
            "message": "Campaign authorized" if approved else "Campaign rejected",
        }

    @staticmethod
    def after_approval(state: PilotState):
        return "execute" if state.get("approved") is True else END

    def execute_node(self, state: PilotState):
        protocol = PilotProtocol.model_validate(state["protocol_document"])
        digest = protocol.fingerprint()
        if (state.get("approved") is not True or state.get("approval_sha256") != digest
                or state.get("protocol_sha256") != digest or state.get("backend") != "docker-local"):
            raise PermissionError("Missing exact campaign/backend approval")
        try:
            run_campaign(protocol, self.manager, poll_seconds=self.poll_seconds)
        except CampaignBlocked as error:
            return {"status": "blocked", "message": str(error)}
        return {"status": "jobs_succeeded", "message": "All jobs finished and artifacts persisted"}

    @staticmethod
    def after_execute(state: PilotState):
        return "analyze" if state.get("status") == "jobs_succeeded" else END

    def analyze_node(self, state: PilotState):
        protocol = PilotProtocol.model_validate(state["protocol_document"])
        report = build_report(protocol, self.manager)
        return {"status": "completed", "message": "Pilot campaign validated and analyzed",
                "analysis": report["analysis"], "simulated": report["simulated"]}

    def build(self, saver):
        builder = StateGraph(PilotState)
        builder.add_node("validate", self.validate_node)
        builder.add_node("approval", self.approval_node)
        builder.add_node("execute", self.execute_node)
        builder.add_node("analyze", self.analyze_node)
        builder.add_edge(START, "validate")
        builder.add_edge("validate", "approval")
        builder.add_conditional_edges("approval", self.after_approval, {"execute": "execute", END: END})
        builder.add_conditional_edges("execute", self.after_execute, {"analyze": "analyze", END: END})
        builder.add_edge("analyze", END)
        return builder.compile(checkpointer=saver)


def pending_pilot_approval(snapshot):
    for task in snapshot.tasks:
        for item in task.interrupts:
            value = item.value
            if isinstance(value, dict) and value.get("kind") == "v846_campaign_approval":
                return value
    return None
