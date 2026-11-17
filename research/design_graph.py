"""V8.3 — Persistent draft -> HITL approval, no experimental execution."""

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict

from providers.base import LLMProvider
from providers.anthropic_provider import AnthropicProvider
from research.design_planner import ExperimentalDesignPlanner
from research.experimental_design import (
    ExperimentalDesign, validate_approved_literature,
    validate_design_against_inputs,
)
from research.factory import create_provider
from research.literature import digest_json


class DesignState(TypedDict, total=False):
    mission_document: dict[str, Any]
    literature_document: dict[str, Any]
    mission_sha256: str
    literature_input_sha256: str
    design: dict[str, Any]
    design_sha256: str
    status: str


class ExperimentalDesignWorkflow:
    def __init__(self, provider_factory: Callable[[], LLMProvider]):
        self.provider_factory = provider_factory

    def draft_node(self, state: DesignState) -> dict:
        mission, _, review = validate_approved_literature(
            state["mission_document"], state["literature_document"]
        )
        proposal = ExperimentalDesignPlanner(self.provider_factory()).propose(mission, review)
        design = proposal.model_dump()
        return {
            "mission_sha256": state["mission_document"]["mission_sha256"],
            "literature_input_sha256": digest_json(state["literature_document"]),
            "design": design,
            "design_sha256": digest_json(design),
            "status": "awaiting_approval",
        }

    @staticmethod
    def approval_node(state: DesignState) -> dict:
        # No writes or sandbox calls before the interruption.
        decision = interrupt({
            "kind": "experimental_design_approval",
            "mission_sha256": state["mission_sha256"],
            "literature_input_sha256": state["literature_input_sha256"],
            "design_sha256": state["design_sha256"],
            "design": state["design"],
        })
        mission, _, review = validate_approved_literature(
            state["mission_document"], state["literature_document"]
        )
        design = ExperimentalDesign.model_validate(state["design"])
        validate_design_against_inputs(design, mission, review)
        accepted = (
            isinstance(decision, dict)
            and decision.get("decision") == "approve"
            and decision.get("mission_sha256") == state["mission_sha256"]
            and decision.get("literature_input_sha256") == state["literature_input_sha256"]
            and decision.get("design_sha256") == state["design_sha256"]
            and state["mission_document"]["mission_sha256"] == state["mission_sha256"]
            and digest_json(state["literature_document"]) == state["literature_input_sha256"]
            and digest_json(state["design"]) == state["design_sha256"]
            and design.model_dump() == state["design"]
        )
        return {"status": "approved" if accepted else "rejected"}

    def build(self, saver):
        builder = StateGraph(DesignState)
        builder.add_node("draft", self.draft_node)
        builder.add_node("approval", self.approval_node)
        builder.add_edge(START, "draft")
        builder.add_edge("draft", "approval")
        builder.add_edge("approval", END)
        return builder.compile(checkpointer=saver)


def _create_design_provider() -> LLMProvider:
    """Use the shared multi-provider factory; V8.3 JSON needs a larger Anthropic output limit."""
    provider = create_provider(native_web_search=False)
    if isinstance(provider, AnthropicProvider):
        provider.max_tokens = max(provider.max_tokens, 6000)
    return provider


def build_design_graph(saver, provider_factory=None):
    return ExperimentalDesignWorkflow(
        provider_factory or _create_design_provider
    ).build(saver)


def pending_design_approval(snapshot):
    for task in snapshot.tasks:
        for item in task.interrupts:
            payload = item.value
            if isinstance(payload, dict) and payload.get("kind") == "experimental_design_approval":
                return payload
    return None
