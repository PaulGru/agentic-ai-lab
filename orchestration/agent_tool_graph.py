"""V4.5: checkpointed routing, handoffs and tool approval in one graph.

Keep dangerous side effects in dedicated execute_tool nodes. Their own
idempotency is the tool's responsibility, not a LangGraph guarantee.
"""

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
from research.source_audit import audit_sources


class HandoffFrame(TypedDict):
    source_agent: str
    target_agent: str
    call_id: str
    task: str


class AgentToolState(TypedDict):
    prompt: str
    agent_name: NotRequired[str]       # Initial route, for reporting
    current_agent: NotRequired[str]   # Agent currently running
    current_prompt: NotRequired[str]
    handoff_stack: NotRequired[list[HandoffFrame]]
    handoff_request: NotRequired[dict[str, Any] | None]
    handoff_result: NotRequired[dict[str, Any] | None]
    return_text: NotRequired[str | None]
    provider_states: NotRequired[dict[str, dict[str, Any]]]
    pending_calls: NotRequired[list[dict[str, Any]]]
    pending_index: NotRequired[int]
    tool_results: NotRequired[list[dict[str, Any]]]
    approved: NotRequired[bool]
    effective_action: NotRequired[dict[str, Any]]
    approval_status: NotRequired[str]
    llm_steps: NotRequired[int]
    answer: NotRequired[str]
    sources: NotRequired[list[dict[str, str]]]
    web_search_used: NotRequired[bool]
    source_audit: NotRequired[dict[str, Any] | None]


class AgentToolGraph:
    def __init__(
        self,
        agent: AgentRunner | None,
        checkpointer: BaseCheckpointSaver,
        policy: ApprovalPolicy | None = None,
        router: Router | None = None,
        max_handoff_depth: int = 3,
        max_llm_steps: int = 20,
    ):
        if (agent is None) == (router is None):
            raise ValueError("Provide exactly one of agent or router")
        if max_handoff_depth <= 0 or max_llm_steps <= 0:
            raise ValueError("Graph limits must be positive")

        self.agent = agent
        self.router = router
        self.policy = policy if policy is not None else ApprovalPolicy()
        self.max_handoff_depth = max_handoff_depth
        self.max_llm_steps = max_llm_steps

        builder = StateGraph(AgentToolState)
        builder.add_node("route", self._route_node)
        builder.add_node("llm", self._llm_node)
        builder.add_node("handoff", self._handoff_node)
        builder.add_node("return_handoff", self._return_handoff_node)
        builder.add_node("policy", self._policy_node)
        builder.add_node("approval", self._approval_node)
        builder.add_node("execute_tool", self._execute_tool_node)
        builder.add_node("audit_sources", self._audit_sources_node)

        builder.add_edge(START, "route")
        builder.add_edge("route", "llm")
        builder.add_edge("audit_sources", END)
        builder.add_conditional_edges(
            "llm", self._after_llm,
            {"policy": "policy", "handoff": "handoff",
             "return_handoff": "return_handoff", "audit_sources": "audit_sources",},
        )
        builder.add_edge("handoff", "llm")
        builder.add_edge("return_handoff", "llm")
        builder.add_conditional_edges(
            "policy", self._after_policy,
            {"approval": "approval", "execute_tool": "execute_tool"},
        )
        builder.add_edge("approval", "execute_tool")
        builder.add_conditional_edges(
            "execute_tool", self._after_tool,
            {"policy": "policy", "llm": "llm"},
        )
        self.graph = builder.compile(checkpointer=checkpointer)

    def _route_node(self, state: AgentToolState) -> dict:
        selected = (
            self.router.route(state["prompt"])
            if self.router is not None else self.agent
        )
        return {
            "agent_name": selected.name,
            "current_agent": selected.name,
            "current_prompt": state["prompt"],
            "handoff_stack": [],
            "handoff_request": None,
            "handoff_result": None,
            "return_text": None,
            "provider_states": {},
            "pending_calls": [],
            "pending_index": 0,
            "tool_results": [],
            "llm_steps": 0,
            "answer": "",
            "sources": [],
            "web_search_used": False,
            "source_audit": None,
        }

    @staticmethod
    def _audit_sources_node(state: AgentToolState) -> dict:
        cleaned_sources, report = audit_sources(
            sources=state.get("sources", []),
            web_search_used=state.get(
                "web_search_used", False
            ),
        )

        return {
            "sources": cleaned_sources,
            "source_audit": report,
        }

    def _selected_agent(self, state: AgentToolState) -> AgentRunner:
        name = state["current_agent"]
        if self.router is not None:
            return self.router.agents[name]
        if self.agent is None or self.agent.name != name:
            raise RuntimeError(f"Unknown current agent: {name}")
        return self.agent

    def _llm_node(self, state: AgentToolState) -> dict:
        agent = self._selected_agent(state)
        steps = state.get("llm_steps", 0)
        if steps >= self.max_llm_steps:
            raise RuntimeError("Maximum graph LLM steps reached")

        if state.get("handoff_result") is not None:
            agent.provider.restore_state(state["provider_states"][agent.name])
            response = agent.continue_after_tools([
                ToolResult(**state["handoff_result"])
            ])
        elif state.get("pending_calls"):
            if state["pending_index"] != len(state["pending_calls"]):
                raise RuntimeError("Incomplete tool results")
            if len(state["tool_results"]) != len(state["pending_calls"]):
                raise RuntimeError("Tool results do not match pending calls")
            for pending, result in zip(state["pending_calls"], state["tool_results"]):
                if pending["id"] != result["call_id"] or pending["name"] != result["name"]:
                    raise RuntimeError("Tool result identity mismatch")
            agent.provider.restore_state(state["provider_states"][agent.name])
            response = agent.continue_after_tools([
                ToolResult(**item) for item in state["tool_results"]
            ])
        else:
            response = agent.propose(state["current_prompt"])
        
        states = dict(state.get("provider_states", {}))
        states[agent.name] = agent.provider.export_state()

        sources = list(state.get("sources", []))
        seen_urls = {source["url"] for source in sources}

        for source in response.sources:
            if source["url"] not in seen_urls:
                sources.append(source)
                seen_urls.add(source["url"])

        web_search_used = (
            state.get("web_search_used", False)
            or response.web_search_used
        )

        common = {
            "provider_states": states,
            "llm_steps": steps + 1,
            "pending_calls": [],
            "pending_index": 0,
            "tool_results": [],
            "handoff_request": None,
            "handoff_result": None,
            "return_text": None,
            "sources": sources,
            "web_search_used": web_search_used,
        }

        handoffs = [c for c in response.tool_calls if c.name == HANDOFF_TOOL_NAME]
        if handoffs:
            if len(response.tool_calls) != 1:
                raise RuntimeError("A handoff must be the only tool call")
            return {
                **common,
                "handoff_request": asdict(handoffs[0]),
                "answer": "",
            }
        if response.tool_calls:
            return {
                **common,
                "pending_calls": [asdict(call) for call in response.tool_calls],
                "answer": "",
            }
        if not response.text:
            raise RuntimeError("LLM returned neither text nor tool calls")

        if state.get("handoff_stack"):
            return {**common, "return_text": response.text, "answer": ""}
        return {**common, "answer": response.text}

    @staticmethod
    def _after_llm(state: AgentToolState) -> str:
        if state.get("handoff_request"):
            return "handoff"
        if state.get("pending_calls"):
            return "policy"
        if state.get("return_text") is not None:
            return "return_handoff"
        return "audit_sources"

    def _handoff_node(self, state: AgentToolState) -> dict:
        call = ToolCall(**state["handoff_request"])
        source = state["current_agent"]
        source_agent = self._selected_agent(state)
        target = call.arguments.get("agent")
        task = call.arguments.get("task")

        if not isinstance(target, str) or target not in source_agent.handoff_targets:
            raise RuntimeError("Handoff target not permitted")
        if self.router is None or target not in self.router.agents:
            raise RuntimeError("Handoff target is not registered")
        if not isinstance(task, str) or not task.strip():
            raise RuntimeError("Handoff task must be nonempty text")

        stack = list(state["handoff_stack"])
        if len(stack) >= self.max_handoff_depth:
            raise RuntimeError("Maximum handoff depth reached")
        active_agents = {frame["source_agent"] for frame in stack}
        active_agents.add(source)
        if target in active_agents:
            raise RuntimeError("Handoff cycle detected")

        frame: HandoffFrame = {
            "source_agent": source,
            "target_agent": target,
            "call_id": call.id,
            "task": task,
        }
        return {
            "handoff_stack": [*stack, frame],
            "current_agent": target,
            "current_prompt": task,
            "handoff_request": None,
            "pending_calls": [],
            "pending_index": 0,
            "tool_results": [],
        }

    @staticmethod
    def _return_handoff_node(state: AgentToolState) -> dict:
        stack = list(state["handoff_stack"])
        if not stack or stack[-1]["target_agent"] != state["current_agent"]:
            raise RuntimeError("Invalid handoff stack")
        result_text = state.get("return_text")
        if not isinstance(result_text, str) or not result_text.strip():
            raise RuntimeError("Missing delegated result")
        frame = stack.pop()
        return {
            "handoff_stack": stack,
            "current_agent": frame["source_agent"],
            "handoff_result": asdict(ToolResult(
                call_id=frame["call_id"], name=HANDOFF_TOOL_NAME,
                output=result_text,
            )),
            "return_text": None,
            "pending_calls": [],
            "pending_index": 0,
            "tool_results": [],
        }

    def _policy_node(self, state: AgentToolState) -> dict:
        action = state["pending_calls"][state["pending_index"]]
        agent = self._selected_agent(state)
        registered = {item.name for item in agent.registry.definitions()}
        mode = self.policy.mode(action["name"], registered)
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
        decision = interrupt({
            "type": "tool_approval",
            "agent": state["current_agent"],
            "action": original,
            "allowed_decisions": ["approve", "reject", "edit"],
        })
        if type(decision) is bool:
            decision = {"decision": "approve" if decision else "reject"}
        if not isinstance(decision, dict):
            raise ValueError("Invalid human decision")
        if decision == {"decision": "approve"}:
            return {
                "approved": True,
                "effective_action": original,
                "approval_status": "approved",
            }
        if decision == {"decision": "reject"}:
            return {
                "approved": False,
                "effective_action": original,
                "approval_status": "rejected",
            }
        if set(decision) == {"decision", "arguments"} and decision.get("decision") == "edit":
            new_args = decision["arguments"]
            old_args = original["arguments"]
            if not isinstance(new_args, dict) or set(new_args) != set(old_args):
                raise ValueError("Edited arguments must retain original keys")
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
        original = state["pending_calls"][state["pending_index"]]
        effective = state["effective_action"]
        # Identity is immutable: an edit may change arguments, not the tool/call ID.
        if (original["id"], original["name"]) != (effective["id"], effective["name"]):
            raise RuntimeError("Tool identity changed after review")
        call = ToolCall(**effective)
        available = {item.name for item in agent.registry.definitions()}
        mode_now = self.policy.mode(call.name, available)
        status = state["approval_status"]
        allowed = state.get("approved") is True and (
            (mode_now == "auto" and status in ("auto", "approved", "edited"))
            or (mode_now == "review" and status in ("approved", "edited"))
        )
        if not allowed:
            reason = (
                "Action refused by human reviewer."
                if status == "rejected" else "Action blocked by system policy."
            )
            result = ToolResult(
                call_id=call.id, name=call.name,
                output=reason, is_error=True,
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
            "policy" if state["pending_index"] < len(state["pending_calls"])
            else "llm"
        )
