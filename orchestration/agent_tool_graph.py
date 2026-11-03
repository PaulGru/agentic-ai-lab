
from dataclasses import asdict
from typing import Any

from typing_extensions import TypedDict, NotRequired

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent.models import ToolCall, ToolResult
from agent.runner import AgentRunner, HANDOFF_TOOL_NAME


class AgentToolState(TypedDict):
    prompt: str

    provider_state: NotRequired[dict[str, Any]]

    pending_calls: NotRequired[list[dict[str, Any]]]
    pending_index: NotRequired[int]

    tool_results: NotRequired[list[dict[str, Any]]]
    approved: NotRequired[bool]

    llm_steps: NotRequired[int]
    answer: NotRequired[str]


class AgentToolGraph:
    def __init__(
        self,
        agent: AgentRunner,
        checkpointer: BaseCheckpointSaver,
    ):
        self.agent = agent

        builder = StateGraph(AgentToolState)

        builder.add_node("llm", self._llm_node)
        builder.add_node("approval", self._approval_node)
        builder.add_node(
            "execute_tool",
            self._execute_tool_node,
        )

        builder.add_edge(START, "llm")

        builder.add_conditional_edges(
            "llm",
            self._after_llm,
            {
                "approval": "approval",
                "end": END,
            },
        )

        builder.add_edge("approval", "execute_tool")

        builder.add_conditional_edges(
            "execute_tool",
            self._after_tool,
            {
                "approval": "approval",
                "llm": "llm",
            },
        )

        self.graph = builder.compile(
            checkpointer=checkpointer,
        )

    # -------------------------
    # Call or continue the LLM
    # -------------------------

    def _llm_node(
        self,
        state: AgentToolState,
    ) -> dict:
        steps = state.get("llm_steps", 0)

        if steps >= self.agent.max_iterations:
            raise RuntimeError(
                "Maximum number of agent iterations reached"
            )

        if state.get("pending_calls"):
            if (
                state["pending_index"]
                != len(state["pending_calls"])
            ):
                raise RuntimeError(
                    "Incomplete tool results"
                )

            self.agent.provider.restore_state(
                state["provider_state"]
            )

            results = [
                ToolResult(**item)
                for item in state["tool_results"]
            ]

            response = self.agent.continue_after_tools(
                results
            )

        else:
            response = self.agent.propose(
                state["prompt"]
            )

        provider_state = (
            self.agent.provider.export_state()
        )

        if response.tool_calls:
            if any(
                call.name == HANDOFF_TOOL_NAME
                for call in response.tool_calls
            ):
                raise RuntimeError(
                    "Handoffs are not supported "
                    "by AgentToolGraph yet"
                )

            return {
                "provider_state": provider_state,
                "pending_calls": [
                    asdict(call)
                    for call in response.tool_calls
                ],
                "pending_index": 0,
                "tool_results": [],
                "llm_steps": steps + 1,
                "answer": "",
            }

        if not response.text:
            raise RuntimeError(
                "LLM returned neither text nor tool calls"
            )

        return {
            "provider_state": provider_state,
            "pending_calls": [],
            "pending_index": 0,
            "tool_results": [],
            "llm_steps": steps + 1,
            "answer": response.text,
        }

    @staticmethod
    def _after_llm(
        state: AgentToolState,
    ) -> str:
        return (
            "approval"
            if state.get("pending_calls")
            else "end"
        )

    # -------------------------
    # Human approval
    # -------------------------

    @staticmethod
    def _approval_node(
        state: AgentToolState,
    ) -> dict:
        action = state["pending_calls"][
            state["pending_index"]
        ]

        decision = interrupt({
            "type": "tool_approval",
            "action": action,
        })

        if type(decision) is not bool:
            raise ValueError(
                "Approval must be a boolean"
            )

        return {"approved": decision}

    # -------------------------
    # Execute or reject tool
    # -------------------------

    def _execute_tool_node(
        self,
        state: AgentToolState,
    ) -> dict:
        call = ToolCall(
            **state["pending_calls"][
                state["pending_index"]
            ]
        )

        if state["approved"] is True:
            result = self.agent.registry.execute(
                call
            )
        else:
            result = ToolResult(
                call_id=call.id,
                name=call.name,
                output=(
                    "Action refused by the human reviewer."
                ),
                is_error=True,
            )

        return {
            "tool_results": [
                *state["tool_results"],
                asdict(result),
            ],
            "pending_index": (
                state["pending_index"] + 1
            ),
        }

    @staticmethod
    def _after_tool(
        state: AgentToolState,
    ) -> str:
        if (
            state["pending_index"]
            < len(state["pending_calls"])
        ):
            return "approval"

        return "llm"
