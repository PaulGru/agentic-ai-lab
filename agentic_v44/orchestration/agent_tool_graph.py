from dataclasses import asdict
from typing import Any

from typing_extensions import TypedDict, NotRequired

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent.models import ToolCall, ToolResult
from agent.runner import AgentRunner, HANDOFF_TOOL_NAME
from orchestration.router import Router
from orchestration.approval_policy import ApprovalPolicy


class AgentToolState(TypedDict):
    prompt: str
    agent_name: NotRequired[str]
    provider_state: NotRequired[dict[str, Any]]
    pending_calls: NotRequired[list[dict[str, Any]]]
    pending_index: NotRequired[int]
    tool_results: NotRequired[list[dict[str, Any]]]
    approved: NotRequired[bool]
    effective_action: NotRequired[dict[str, Any]]
    approval_status: NotRequired[str]
    llm_steps: NotRequired[int]
    answer: NotRequired[str]


class AgentToolGraph:
    def __init__(
        self,
        agent: AgentRunner | None,
        checkpointer: BaseCheckpointSaver,
        policy: ApprovalPolicy | None = None,
        router: Router | None = None,
    ):
        if (agent is None) == (router is None):
            raise ValueError("Provide exactly one of agent or router")

        self.agent = agent
        self.router = router
        self.policy = policy if policy is not None else ApprovalPolicy()

        builder = StateGraph(AgentToolState)
        builder.add_node("route", self._route_node)
        builder.add_node("llm", self._llm_node)
        builder.add_node("policy", self._policy_node)
        builder.add_node("approval", self._approval_node)
        builder.add_node("execute_tool", self._execute_tool_node)

        builder.add_edge(START, "route")
        builder.add_edge("route", "llm")
        builder.add_conditional_edges(
            "llm", self._after_llm, {"policy": "policy", "end": END}
        )
        builder.add_conditional_edges(
            "policy",
            self._after_policy,
            {"approval": "approval", "execute_tool": "execute_tool"},
        )
        builder.add_edge("approval", "execute_tool")
        builder.add_conditional_edges(
            "execute_tool",
            self._after_tool,
            {"policy": "policy", "llm": "llm"},
        )

        self.graph = builder.compile(checkpointer=checkpointer)

    def _route_node(self, state: AgentToolState) -> dict:
        agent = (
            self.router.route(state["prompt"])
            if self.router is not None
            else self.agent
        )
        return {"agent_name": agent.name}

    def _selected_agent(self, state: AgentToolState) -> AgentRunner:
        if self.router is not None:
            return self.router.agents[state["agent_name"]]
        return self.agent

    def _llm_node(self, state: AgentToolState) -> dict:
        agent = self._selected_agent(state)
        steps = state.get("llm_steps", 0)
        if steps >= agent.max_iterations:
            raise RuntimeError("Maximum number of agent iterations reached")

        if state.get("pending_calls"):
            if state["pending_index"] != len(state["pending_calls"]):
                raise RuntimeError("Incomplete tool results")
            agent.provider.restore_state(state["provider_state"])
            results = [ToolResult(**item) for item in state["tool_results"]]
            response = agent.continue_after_tools(results)
        else:
            response = agent.propose(state["prompt"])

        provider_state = agent.provider.export_state()
        if response.tool_calls:
            if any(call.name == HANDOFF_TOOL_NAME for call in response.tool_calls):
                raise RuntimeError("Handoffs are not supported by AgentToolGraph yet")
            return {
                "provider_state": provider_state,
                "pending_calls": [asdict(call) for call in response.tool_calls],
                "pending_index": 0,
                "tool_results": [],
                "llm_steps": steps + 1,
                "answer": "",
            }

        if not response.text:
            raise RuntimeError("LLM returned neither text nor tool calls")
        return {
            "provider_state": provider_state,
            "pending_calls": [],
            "pending_index": 0,
            "tool_results": [],
            "llm_steps": steps + 1,
            "answer": response.text,
        }

    @staticmethod
    def _after_llm(state: AgentToolState) -> str:
        return "policy" if state.get("pending_calls") else "end"

    def _policy_node(self, state: AgentToolState) -> dict:
        action = state["pending_calls"][state["pending_index"]]
        agent = self._selected_agent(state)
        available = {definition.name for definition in agent.registry.definitions()}
        mode = self.policy.mode(action["name"], available)
        return {
            "approved": mode == "auto",
            "effective_action": action,
            "approval_status": mode,
        }

    @staticmethod
    def _after_policy(state: AgentToolState) -> str:
        return (
            "approval" if state["approval_status"] == "review"
            else "execute_tool"
        )

    @staticmethod
    def _approval_node(state: AgentToolState) -> dict:
        original = state["pending_calls"][state["pending_index"]]
        response = interrupt({
            "type": "tool_approval",
            "agent": state["agent_name"],
            "action": original,
            "allowed_decisions": ["approve", "reject", "edit"],
        })

        # Backward compatibility with V4.3 boolean decisions.
        if type(response) is bool:
            response = {"decision": "approve" if response else "reject"}

        if not isinstance(response, dict):
            raise ValueError("Invalid human decision")

        decision = response.get("decision")
        if decision == "approve" and set(response) == {"decision"}:
            return {
                "approved": True,
                "effective_action": original,
                "approval_status": "approved",
            }

        if decision == "reject" and set(response) == {"decision"}:
            return {
                "approved": False,
                "effective_action": original,
                "approval_status": "rejected",
            }

        if decision == "edit" and set(response) == {"decision", "arguments"}:
            new_args = response["arguments"]
            old_args = original["arguments"]
            if not isinstance(new_args, dict) or set(new_args) != set(old_args):
                raise ValueError("Edited arguments must retain the original keys")
            if any(type(new_args[k]) is not type(old_args[k]) for k in old_args):
                raise ValueError("Edited argument types must not change")
            return {
                "approved": True,
                "effective_action": {**original, "arguments": dict(new_args)},
                "approval_status": "edited",
            }

        raise ValueError("Invalid human decision")

    def _execute_tool_node(self, state: AgentToolState) -> dict:
        agent = self._selected_agent(state)
        call = ToolCall(**state["effective_action"])
        available = {definition.name for definition in agent.registry.definitions()}
        current_mode = self.policy.mode(call.name, available)

        if current_mode == "deny" or (
            current_mode == "review" and state["approval_status"] == "auto"
        ):
            result = ToolResult(
                call_id=call.id, name=call.name,
                output="Action blocked by system policy.", is_error=True,
            )
        elif not state["approved"]:
            result = ToolResult(
                call_id=call.id, name=call.name,
                output="Action refused by human reviewer.", is_error=True,
            )
        else:
            result = agent.registry.execute(call)

        return {
            "tool_results": [*state["tool_results"], asdict(result)],
            "pending_index": state["pending_index"] + 1,
        }

    @staticmethod
    def _after_tool(state: AgentToolState) -> str:
        return (
            "policy"
            if state["pending_index"] < len(state["pending_calls"])
            else "llm"
        )
