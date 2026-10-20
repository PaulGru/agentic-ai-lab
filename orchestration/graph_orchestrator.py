
from typing_extensions import TypedDict, NotRequired

from langgraph.graph import StateGraph, START, END

from orchestration.router import Router
from orchestration.orchestrator import Orchestrator
from observability.tracer import Tracer


# -------------------------
# Shared graph state
# -------------------------

class GraphState(TypedDict):
    prompt: str
    selected_agent: NotRequired[str]
    answer: NotRequired[str]


# -------------------------
# Graph orchestrator
# -------------------------

class GraphOrchestrator:
    def __init__(
        self,
        router: Router,
        orchestrator: Orchestrator,
        tracer: Tracer | None = None,
    ):
        self.router = router
        self.orchestrator = orchestrator
        self.tracer = tracer

        builder = StateGraph(GraphState)

        # Define nodes
        builder.add_node(
            "route",
            self._route_node,
        )

        builder.add_node(
            "math",
            self._math_node,
        )

        builder.add_node(
            "general",
            self._general_node,
        )

        # Define graph transitions
        builder.add_edge(START, "route")

        builder.add_conditional_edges(
            "route",
            self._select_next_node,
            {
                "math": "math",
                "general": "general",
            },
        )

        builder.add_edge("math", END)
        builder.add_edge("general", END)

        self.graph = builder.compile()

    # -------------------------
    # Routing node
    # -------------------------

    def _route_node(
        self,
        state: GraphState,
    ) -> dict[str, str]:
        agent = self.router.route(
            state["prompt"]
        )

        return {
            "selected_agent": agent.name,
        }

    # -------------------------
    # Conditional routing
    # -------------------------

    def _select_next_node(
        self,
        state: GraphState,
    ) -> str:
        selected_agent = state["selected_agent"]

        if selected_agent not in ("math", "general"):
            raise RuntimeError(
                f"Unsupported graph agent: "
                f"'{selected_agent}'"
            )

        return selected_agent

    # -------------------------
    # Agent nodes
    # -------------------------

    def _math_node(
        self,
        state: GraphState,
    ) -> dict[str, str]:
        self._trace(
            "graph_node_entered",
            node="math",
        )

        answer = self.orchestrator.run_selected(
            prompt=state["prompt"],
            agent_name="math",
        )

        return {
            "answer": answer,
        }

    def _general_node(
        self,
        state: GraphState,
    ) -> dict[str, str]:
        self._trace(
            "graph_node_entered",
            node="general",
        )

        answer = self.orchestrator.run_selected(
            prompt=state["prompt"],
            agent_name="general",
        )

        return {
            "answer": answer,
        }

    # -------------------------
    # Public execution
    # -------------------------

    def run(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if self.tracer is not None:
            self.tracer.start_run()
            self.tracer.record(
                "run_started",
                prompt=prompt,
            )

        try:
            final_state = self.graph.invoke(
                {
                    "prompt": prompt,
                }
            )

            self._trace("run_finished")

            return final_state["answer"]

        except Exception as error:
            self._trace(
                "run_failed",
                error_type=type(error).__name__,
                error=str(error),
            )

            raise

    def _trace(
        self,
        event_type: str,
        **data,
    ) -> None:
        if self.tracer is not None:
            self.tracer.record(
                event_type,
                **data,
            )
