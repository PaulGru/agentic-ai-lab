
from typing import Any

from typing_extensions import TypedDict, NotRequired

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent.models import ToolCall
from tools.registry import ToolRegistry


class ActionState(TypedDict):
    action: dict[str, Any]
    approved: NotRequired[bool]
    status: NotRequired[str]
    tool_output: NotRequired[str]


class ToolApprovalGraph:
    def __init__(
        self,
        registry: ToolRegistry,
        checkpointer: BaseCheckpointSaver,
    ):
        self.registry = registry

        builder = StateGraph(ActionState)

        builder.add_node(
            "approval",
            self._approval_node,
        )

        builder.add_node(
            "execute_tool",
            self._execute_tool_node,
        )

        builder.add_edge(START, "approval")

        builder.add_conditional_edges(
            "approval",
            self._after_approval,
            {
                "execute_tool": "execute_tool",
                "end": END,
            },
        )

        builder.add_edge("execute_tool", END)

        self.graph = builder.compile(
            checkpointer=checkpointer,
        )

    def _approval_node(
        self,
        state: ActionState,
    ) -> dict:
        decision = interrupt({
            "type": "tool_approval",
            "action": state["action"],
        })

        if type(decision) is not bool:
            raise ValueError(
                "Approval must be a boolean"
            )

        return {
            "approved": decision,
            "status": (
                "approved" if decision
                else "rejected"
            ),
        }

    def _after_approval(
        self,
        state: ActionState,
    ) -> str:
        if state["approved"] is True:
            return "execute_tool"

        return "end"

    def _execute_tool_node(
        self,
        state: ActionState,
    ) -> dict:
        if state.get("approved") is not True:
            raise PermissionError(
                "Tool execution not approved"
            )

        call = ToolCall(**state["action"])

        result = self.registry.execute(call)

        return {
            "tool_output": result.output,
            "status": (
                "error" if result.is_error
                else "executed"
            ),
        }
