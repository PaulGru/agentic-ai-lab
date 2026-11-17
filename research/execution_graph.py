"""V8.4 — HITL checkpoint graph for ONE bounded engineering canary."""

from pathlib import Path
from typing import Any, Callable

from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict

from research.canary_runtime import execute_canary
from research.execution_contract import plan_sha256


class ExecutionState(TypedDict, total=False):
    thread: str
    plan: dict[str, Any]
    digest: str
    approved: bool
    status: str
    receipt_path: str
    report: dict[str, Any]


class ExecutionWorkflow:
    def __init__(self, receipts_root: Path, model_factory: Callable | None = None):
        self.receipts_root = receipts_root
        self.model_factory = model_factory

    @staticmethod
    def prepare_node(state: ExecutionState):
        if state["plan"].get("mode") != "engineering_canary_not_scientific_results":
            raise ValueError("V8.4 supports only a canary, not full experimental execution")
        return {
            "digest": plan_sha256(state["plan"]),
            "status": "awaiting_approval",
            "approved": False,
        }

    @staticmethod
    def approval_node(state: ExecutionState):
        decision = interrupt({
            "kind": "v84_canary_approval",
            "thread": state["thread"],
            "plan_sha256": state["digest"],
            "mode": state["plan"]["mode"],
            "cell": state["plan"]["cell"],
            "treatment": state["plan"]["treatment"],
            "evaluations": len(state["plan"]["task_seed_pairs"]),
            "worst_case_calls": state["plan"]["worst_case_calls"],
            "conservative_estimated_usd": state["plan"]["conservative_estimated_usd"],
        })
        allowed = (
            isinstance(decision, dict)
            and decision.get("decision") == "approve"
            and decision.get("plan_sha256") == state["digest"]
            and plan_sha256(state["plan"]) == state["digest"]
        )
        return {
            "approved": allowed,
            "status": "approved" if allowed else "rejected",
        }

    @staticmethod
    def after_approval(state: ExecutionState):
        return "execute" if state["approved"] is True else END

    def execute_node(self, state: ExecutionState):
        if state.get("approved") is not True or plan_sha256(state["plan"]) != state["digest"]:
            raise PermissionError("No valid approval for this execution plan")
        report_path = self.receipts_root / state["thread"] / "v84_execution.json"
        report = execute_canary(state["plan"], report_path, self.model_factory)
        return {
            "status": report["status"],
            "receipt_path": str(report_path),
            "report": report,
            "approved": False,
        }

    def build(self, saver):
        builder = StateGraph(ExecutionState)
        builder.add_node("prepare", self.prepare_node)
        builder.add_node("approve", self.approval_node)
        builder.add_node("execute", self.execute_node)
        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "approve")
        builder.add_conditional_edges("approve", self.after_approval, {"execute": "execute", END: END})
        builder.add_edge("execute", END)
        return builder.compile(checkpointer=saver)


def build_execution_graph(saver, receipts_root: Path, model_factory=None):
    return ExecutionWorkflow(receipts_root, model_factory).build(saver)


def pending_execution_approval(snapshot):
    for task in snapshot.tasks:
        for item in task.interrupts:
            payload = item.value
            if isinstance(payload, dict) and payload.get("kind") == "v84_canary_approval":
                return payload
    return None
