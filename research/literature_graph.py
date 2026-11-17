"""V8.2 — Persistent LangGraph: discovery -> analysis -> human approval."""

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict

from providers.base import LLMProvider
from research.factory import create_provider
from research.literature import (
    LiteratureCatalog, LiteratureReview, digest_json,
    validate_approved_mission, validate_evidence,
)
from research.literature_analyst import LiteratureAnalyst
from research.literature_search import OpenAlexClient, retrieve_catalog
from research.memory.integration import approval_memory, checkpoint_memory, memory_decision_matches, verify_checkpoint_memory


class LiteratureState(TypedDict, total=False):
    mission_document: dict[str, Any]
    catalog: dict[str, Any]
    catalog_sha256: str
    review: dict[str, Any]
    review_sha256: str
    status: str
    memory_context: dict[str, Any]
    memory_binding: dict[str, Any]


class LiteratureWorkflow:
    def __init__(
        self,
        client: OpenAlexClient,
        provider_factory: Callable[[], LLMProvider],
        *,
        max_queries: int = 4,
        per_query: int = 5,
        search_queries: list[str] | None = None,
        memory=None,
    ):
        self.client = client
        self.provider_factory = provider_factory
        self.max_queries = max_queries
        self.per_query = per_query
        self.search_queries = search_queries
        self.memory = memory

    def discover_node(self, state: LiteratureState):
        mission = validate_approved_mission(state["mission_document"])
        catalog = retrieve_catalog(
            mission, self.client, max_queries=self.max_queries,
            per_query=self.per_query, queries_override=self.search_queries,
        )
        raw = catalog.model_dump()
        return {
            "catalog": raw,
            "catalog_sha256": digest_json(raw),
            "status": "cataloged",
        }

    def analyze_node(self, state: LiteratureState):
        mission = validate_approved_mission(state["mission_document"])
        catalog = LiteratureCatalog.model_validate(state["catalog"])
        analyst = LiteratureAnalyst(self.provider_factory(), memory=self.memory)
        review = analyst.analyze(mission, catalog)
        raw = review.model_dump()
        return {
            "review": raw,
            "review_sha256": digest_json(raw),
            "status": "awaiting_approval",
            **checkpoint_memory(self.memory, analyst.last_memory_context),
        }

    def approval_node(self, state: LiteratureState):
        verify_checkpoint_memory(self.memory, state)
        # No external side effects before interrupt; resuming never re-fetches or re-prompts.
        decision = interrupt({
            "kind": "literature_review_approval",
            "mission_sha256": state["mission_document"]["mission_sha256"],
            "catalog_sha256": state["catalog_sha256"],
            "review_sha256": state["review_sha256"],
            "paper_count": len(state["catalog"]["papers"]),
            "review": state["review"],
            **approval_memory(state),
        })
        verify_checkpoint_memory(self.memory, state)
        catalog = LiteratureCatalog.model_validate(state["catalog"])
        review = LiteratureReview.model_validate(state["review"])
        validate_evidence(review, catalog)
        validate_approved_mission(state["mission_document"])
        accepted = (
            isinstance(decision, dict)
            and decision.get("decision") == "approve"
            and decision.get("mission_sha256") == state["mission_document"]["mission_sha256"]
            and decision.get("catalog_sha256") == state["catalog_sha256"]
            and decision.get("review_sha256") == state["review_sha256"]
            and digest_json(state["catalog"]) == state["catalog_sha256"]
            and digest_json(state["review"]) == state["review_sha256"]
            and memory_decision_matches(state, decision)
        )
        return {"status": "approved" if accepted else "rejected"}

    def build(self, saver):
        builder = StateGraph(LiteratureState)
        builder.add_node("discover", self.discover_node)
        builder.add_node("analyze", self.analyze_node)
        builder.add_node("approval", self.approval_node)
        builder.add_edge(START, "discover")
        builder.add_edge("discover", "analyze")
        builder.add_edge("analyze", "approval")
        builder.add_edge("approval", END)
        return builder.compile(checkpointer=saver)


def build_literature_graph(
    saver, *, client=None, provider_factory=None, max_queries=4,
    per_query=5, search_queries: list[str] | None = None, memory=None,
):
    if client is None:
        client = OpenAlexClient()
    if provider_factory is None:
        provider_factory = lambda: create_provider(native_web_search=False)
    return LiteratureWorkflow(
        client, provider_factory, max_queries=max_queries,
        per_query=per_query, search_queries=search_queries, memory=memory,
    ).build(saver)


def pending_literature_approval(snapshot):
    for task in snapshot.tasks:
        for item in task.interrupts:
            payload = item.value
            if isinstance(payload, dict) and payload.get("kind") == "literature_review_approval":
                return payload
    return None
