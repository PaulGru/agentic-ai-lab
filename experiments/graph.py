
"""V7.4 - Workflow expérimental avec HITL LangGraph."""

from dataclasses import asdict
from hashlib import sha256
from typing import Any, Callable

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from typing_extensions import TypedDict

from experiments.agent import ExperimentAgent
from experiments.proposal import ExperimentProposal
from experiments.reviser import ExperimentReviser
from experiments.sandbox import SandboxRunner, SandboxResult
from observability.tracer import Tracer
from providers.base import LLMProvider


MAX_ATTEMPTS = 3


class ExperimentState(TypedDict, total=False):
    question: str
    initial_proposal: dict[str, Any]
    proposal: dict[str, Any]
    attempts: list[dict[str, Any]]
    model_calls: list[dict[str, Any]]
    last_execution: dict[str, Any]
    approved: bool
    approval_sha256: str | None
    status: str
    message: str


def code_digest(code: str) -> str:
    return sha256(code.encode("utf-8")).hexdigest()


class ExperimentWorkflow:
    def __init__(
        self,
        provider_factory: Callable[[], LLMProvider],
        sandbox: SandboxRunner,
    ):
        self.provider_factory = provider_factory
        self.sandbox = sandbox

    def propose_node(self, state: ExperimentState):
        tracer = Tracer(print_events=False)
        # Une proposition fournie permet les démos déterministes.
        # Sinon, la proposition vient de notre Experiment Agent.
        if "initial_proposal" in state:
            proposal = ExperimentProposal.model_validate(
                state["initial_proposal"]
            )
        else:
            agent = ExperimentAgent(self.provider_factory(), tracer=tracer)
            proposal = agent.propose(state["question"])

        calls = [event.data for event in tracer.events if event.type == "model_call"]
        if proposal.dependencies:
            return {
                "proposal": proposal.model_dump(),
                "attempts": [],
                "model_calls": calls,
                "status": "unsupported_dependencies",
                "message": "V7.4 requires standard-library-only code",
            }

        return {
            "proposal": proposal.model_dump(),
            "attempts": [],
            "model_calls": calls,
            "approved": False,
            "approval_sha256": None,
            "status": "awaiting_approval",
            "message": "",
        }

    @staticmethod
    def after_propose(state: ExperimentState):
        if state["status"] == "awaiting_approval":
            return "approve"
        return END

    @staticmethod
    def approve_node(state: ExperimentState):
        proposal = ExperimentProposal.model_validate(
            state["proposal"]
        )
        digest = code_digest(proposal.code)

        # IMPORTANT : aucun effet externe avant interrupt().
        decision = interrupt({
            "kind": "experiment_approval",
            "attempt": len(state.get("attempts", [])) + 1,
            "hypothesis": proposal.hypothesis,
            "code": proposal.code,
            "sha256": digest,
        })

        authorized = (
            isinstance(decision, dict)
            and decision.get("decision") == "approve"
            and decision.get("sha256") == digest
        )

        return {
            "approved": authorized,
            "approval_sha256": digest if authorized else None,
            "status": "approved" if authorized else "rejected",
        }

    @staticmethod
    def after_approval(state: ExperimentState):
        return "execute" if state["approved"] is True else END

    def execute_node(self, state: ExperimentState):
        proposal = ExperimentProposal.model_validate(
            state["proposal"]
        )

        digest = code_digest(proposal.code)

        # Défense supplémentaire contre une approbation obsolète.
        if (
            state.get("approved") is not True
            or state.get("approval_sha256") != digest
        ):
            raise PermissionError(
                "Current code was not approved"
            )

        if proposal.dependencies:
            raise ValueError(
                "External dependencies are not allowed"
            )

        result = self.sandbox.run(
            proposal.code,
            approved=True,
            dependencies=proposal.dependencies,
        )

        execution = asdict(result)

        attempts = [
            *state.get("attempts", []),
            {
                "number": len(state.get("attempts", [])) + 1,
                "code": proposal.code,
                "code_sha256": digest,
                "execution": execution,
            },
        ]

        return {
            "attempts": attempts,
            "last_execution": execution,
            "approved": False,
            "approval_sha256": None,
            "status": "executed",
        }

    @staticmethod
    def review_node(state: ExperimentState):
        result = state["last_execution"]
        status = result["status"]

        if status == "success":
            return {
                "status": "completed",
                "message": "Technical execution succeeded",
            }

        if status not in {"error", "timeout", "output_limit"}:
            return {
                "status": "stopped",
                "message": "Unknown execution status",
            }

        if len(state["attempts"]) >= MAX_ATTEMPTS:
            return {
                "status": "max_attempts",
                "message": "Maximum attempts reached",
            }

        return {
            "status": "needs_revision",
            "message": "A correction may be proposed",
        }

    @staticmethod
    def after_review(state: ExperimentState):
        if state["status"] == "needs_revision":
            return "revise"
        return END

    def revise_node(self, state: ExperimentState):
        current = ExperimentProposal.model_validate(
            state["proposal"]
        )
        execution = SandboxResult(
            **state["last_execution"]
        )

        tracer = Tracer(print_events=False)
        try:
            revised = ExperimentReviser(
                self.provider_factory(), tracer=tracer
            ).revise(current, execution)
        except ValueError:
            return {
                "status": "revision_failed",
                "message": "LLM revision validation failed",
                "model_calls": [*state.get("model_calls", []), *(event.data for event in tracer.events
                                                                 if event.type == "model_call")],
            }

        calls = [*state.get("model_calls", []), *(event.data for event in tracer.events if event.type == "model_call")]
        if revised.code == current.code:
            return {
                "status": "unchanged",
                "message": "No correction was proposed",
                "model_calls": calls,
            }

        # Aucune ancienne approbation n'est réutilisée.
        return {
            "proposal": revised.model_dump(),
            "approved": False,
            "approval_sha256": None,
            "status": "awaiting_approval",
            "message": "Corrected code requires new approval",
            "model_calls": calls,
        }

    @staticmethod
    def after_revision(state: ExperimentState):
        if state["status"] == "awaiting_approval":
            return "approve"
        return END

    def build(self, saver):
        builder = StateGraph(ExperimentState)

        builder.add_node("propose", self.propose_node)
        builder.add_node("approve", self.approve_node)
        builder.add_node("execute", self.execute_node)
        builder.add_node("review", self.review_node)
        builder.add_node("revise", self.revise_node)

        builder.add_edge(START, "propose")

        builder.add_conditional_edges(
            "propose",
            self.after_propose,
            {"approve": "approve", END: END},
        )

        builder.add_conditional_edges(
            "approve",
            self.after_approval,
            {"execute": "execute", END: END},
        )

        builder.add_edge("execute", "review")

        builder.add_conditional_edges(
            "review",
            self.after_review,
            {"revise": "revise", END: END},
        )

        builder.add_conditional_edges(
            "revise",
            self.after_revision,
            {"approve": "approve", END: END},
        )

        return builder.compile(checkpointer=saver)
