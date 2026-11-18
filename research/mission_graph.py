"""V8.1 — LangGraph de proposition et d'approbation d'une mission."""

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict

from providers.base import LLMProvider
from research.factory import create_provider
from research.mission import mission_digest, parse_brief, sha256_text
from research.mission_planner import MissionPlanner
from research.memory.integration import approval_memory, checkpoint_memory, memory_decision_matches, verify_checkpoint_memory


class MissionState(TypedDict, total=False):
    brief_text: str
    brief_sections: dict[str, str]
    brief_sha256: str
    mission: dict[str, Any]
    mission_sha256: str
    status: str
    memory_context: dict[str, Any]
    memory_binding: dict[str, Any]


class MissionWorkflow:
    def __init__(self, provider_factory: Callable[[], LLMProvider], *, memory=None):
        self.provider_factory = provider_factory
        self.memory = memory

    def plan_node(self, state: MissionState):
        sections = parse_brief(state["brief_text"])
        planner = MissionPlanner(self.provider_factory(), memory=self.memory)
        mission = planner.propose(sections)
        payload = mission.model_dump()
        return {
            "brief_sections": sections,
            "brief_sha256": sha256_text(state["brief_text"]),
            "mission": payload,
            "mission_sha256": mission_digest(payload),
            "status": "awaiting_approval",
            **checkpoint_memory(self.memory, planner.last_memory_context),
        }

    def approval_node(self, state: MissionState):
        verify_checkpoint_memory(self.memory, state)
        # Aucun effet externe avant l'interruption.
        decision = interrupt({
            "kind": "research_mission_approval",
            "brief_sha256": state["brief_sha256"],
            "mission_sha256": state["mission_sha256"],
            "mission": state["mission"],
            **approval_memory(state),
        })
        verify_checkpoint_memory(self.memory, state)
        accepted = (
            isinstance(decision, dict)
            and decision.get("decision") == "approve"
            and decision.get("brief_sha256") == state["brief_sha256"]
            and decision.get("mission_sha256") == state["mission_sha256"]
            and mission_digest(state["mission"]) == state["mission_sha256"]
            and memory_decision_matches(state, decision)
        )
        return {"status": "approved" if accepted else "rejected"}

    def build(self, saver):
        builder = StateGraph(MissionState)
        builder.add_node("plan", self.plan_node)
        builder.add_node("approval", self.approval_node)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "approval")
        builder.add_edge("approval", END)
        return builder.compile(checkpointer=saver)


def build_mission_graph(saver, provider_factory=None, *, memory=None):
    if provider_factory is None:
        provider_factory = lambda: create_provider(native_web_search=False)
    return MissionWorkflow(provider_factory, memory=memory).build(saver)


def pending_approval(snapshot):
    for task in snapshot.tasks:
        for item in task.interrupts:
            payload = item.value
            if (isinstance(payload, dict)
                    and payload.get("kind") == "research_mission_approval"):
                return payload
    return None
