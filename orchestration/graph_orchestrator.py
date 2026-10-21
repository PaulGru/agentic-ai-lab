
from typing_extensions import TypedDict, NotRequired

from langgraph.graph import StateGraph, START, END

from agent.models import HandoffRequest
from orchestration.router import Router
from observability.tracer import Tracer


# -------------------------
# Handoff information
# -------------------------

class HandoffFrame(TypedDict):
    source_agent: str
    target_agent: str
    task: str
    call_id: str


# -------------------------
# Shared graph state
# -------------------------

class GraphState(TypedDict):
    prompt: str

    selected_agent: NotRequired[str]
    current_agent: NotRequired[str]
    current_prompt: NotRequired[str]

    handoff_stack: NotRequired[list[HandoffFrame]]
    return_result: NotRequired[str | None]

    answer: NotRequired[str]
    finished: NotRequired[bool]


# -------------------------
# Graph orchestrator
# -------------------------

class GraphOrchestrator:
    def __init__(
        self,
        router: Router,
        tracer: Tracer | None = None,
        max_handoff_depth: int = 3,
    ):
        if max_handoff_depth <= 0:
            raise ValueError(
                "max_handoff_depth must be greater than 0"
            )

        self.router = router
        self.tracer = tracer
        self.max_handoff_depth = max_handoff_depth

        builder = StateGraph(GraphState)

        # Nodes
        builder.add_node("route", self._route_node)
        builder.add_node("math", self._math_node)
        builder.add_node("general", self._general_node)

        # Initial routing
        builder.add_edge(START, "route")

        builder.add_conditional_edges(
            "route",
            self._next_node,
            {
                "math": "math",
                "general": "general",
            },
        )

        # Agent transitions, including handoffs
        paths = {
            "math": "math",
            "general": "general",
            "end": END,
        }

        builder.add_conditional_edges(
            "math",
            self._next_node,
            paths,
        )

        builder.add_conditional_edges(
            "general",
            self._next_node,
            paths,
        )

        self.graph = builder.compile()

    # -------------------------
    # Initial routing
    # -------------------------

    def _route_node(
        self,
        state: GraphState,
    ) -> dict:
        agent = self.router.route(
            state["prompt"]
        )

        return {
            "selected_agent": agent.name,
            "current_agent": agent.name,
            "current_prompt": state["prompt"],
            "handoff_stack": [],
            "return_result": None,
            "finished": False,
        }

    # -------------------------
    # Graph navigation
    # -------------------------

    def _next_node(
        self,
        state: GraphState,
    ) -> str:
        if state.get("finished", False):
            return "end"

        agent_name = state["current_agent"]

        if agent_name not in ("math", "general"):
            raise RuntimeError(
                f"Unsupported graph agent: '{agent_name}'"
            )

        return agent_name

    # -------------------------
    # Agent nodes
    # -------------------------

    def _math_node(
        self,
        state: GraphState,
    ) -> dict:
        return self._execute_agent(
            state,
            agent_name="math",
        )

    def _general_node(
        self,
        state: GraphState,
    ) -> dict:
        return self._execute_agent(
            state,
            agent_name="general",
        )

    # -------------------------
    # Agent execution
    # -------------------------

    def _execute_agent(
        self,
        state: GraphState,
        agent_name: str,
    ) -> dict:
        if state["current_agent"] != agent_name:
            raise RuntimeError(
                "Graph agent mismatch"
            )

        self._trace(
            "graph_node_entered",
            node=agent_name,
        )

        agent = self.router.agents[agent_name]
        stack = list(state["handoff_stack"])

        returned_result = state.get("return_result")

        # Resume an agent after a delegated task
        if returned_result is not None:
            if (
                not stack
                or stack[-1]["source_agent"] != agent_name
            ):
                raise RuntimeError(
                    "Invalid handoff return state"
                )

            frame = stack.pop()

            request = HandoffRequest(
                call_id=frame["call_id"],
                target_agent=frame["target_agent"],
                task=frame["task"],
            )

            outcome = agent.resume_handoff(
                request=request,
                result=returned_result,
            )

        # First execution of an agent
        else:
            outcome = agent.run(
                state["current_prompt"]
            )

        # Common state updates
        updates = {
            "handoff_stack": stack,
            "return_result": None,
        }

        # -------------------------
        # Case 1: Handoff requested
        # -------------------------

        if outcome.handoff is not None:
            request = outcome.handoff
            target = request.target_agent

            if target not in self.router.agents:
                raise RuntimeError(
                    f"Unknown handoff target: '{target}'"
                )

            if target not in ("math", "general"):
                raise RuntimeError(
                    f"Unsupported graph agent: '{target}'"
                )

            if len(stack) >= self.max_handoff_depth:
                raise RuntimeError(
                    "Maximum handoff depth reached"
                )

            active_agents = {
                frame["source_agent"]
                for frame in stack
            }
            active_agents.add(agent_name)

            if target in active_agents:
                raise RuntimeError(
                    f"Handoff cycle detected for "
                    f"agent '{target}'"
                )

            frame = HandoffFrame(
                source_agent=agent_name,
                target_agent=target,
                task=request.task,
                call_id=request.call_id,
            )

            stack.append(frame)

            self._trace(
                "handoff_started",
                source=agent_name,
                target=target,
                depth=len(stack),
            )

            return {
                **updates,
                "handoff_stack": stack,
                "current_agent": target,
                "current_prompt": request.task,
                "finished": False,
            }

        # -------------------------
        # Case 2: Agent returned text
        # -------------------------

        if outcome.text is None:
            raise RuntimeError(
                "Agent produced no final text"
            )

        # Return the result to the delegating agent
        if stack:
            frame = stack[-1]

            if frame["target_agent"] != agent_name:
                raise RuntimeError(
                    "Invalid handoff stack"
                )

            self._trace(
                "handoff_returned",
                source=agent_name,
                target=frame["source_agent"],
                depth=len(stack),
            )

            return {
                **updates,
                "current_agent": frame["source_agent"],
                "return_result": outcome.text,
                "finished": False,
            }

        # No pending handoff: final answer
        return {
            **updates,
            "answer": outcome.text,
            "finished": True,
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
                {"prompt": prompt},
                config={"recursion_limit": 30},
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
