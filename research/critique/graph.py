"""V8.6 LangGraph: review -> optional authorized LLM -> prospective revision HITL.

Checkpoints describe progress. SQLite holds immutable audit records and call
intents. Every sensitive resume reopens and recomputes the ORIGINAL sources.
No node has an experiment submission backend.
"""
from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict

from analysis.contracts import fingerprint
from analysis.retry_gpu import RetryGPUAnalysis
from analysis.executable_protocol import ProtocolAnalysis
from analysis.retry_pilot import RetryPilotAnalysis
from providers.base import LLMProvider
from research.critique.contracts import (
    CriticSuggestions, EvidenceBundle, HumanDecision, LLMRequest, RevisionDraft,
    RevisionProposal, RevisionSubmission, ReviewInput, ReviewReport,
)
from research.critique.evidence import load_input, open_evidence_ledger
from research.critique.llm import call_llm, create_bounded_provider, make_request, validate_suggestions
from research.critique.revision import build_revision, revision_matrix, verify_revision
from research.critique.rules import deterministic_review, with_suggestions
from research.critique.store import ReviewStore


class ReviewState(TypedDict, total=False):
    input_document: dict[str, Any]
    bundle_document: dict[str, Any]
    report_document: dict[str, Any]
    pending_call: dict[str, Any]
    authorization: dict[str, Any]
    llm_approved: bool
    draft_document: dict[str, Any]
    proposal_document: dict[str, Any]
    cycle: int
    feedback: str
    status: str
    memory_context_sha256: str
    memory_indexed_items: int


def config_for(review_id: str) -> dict:
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", review_id):
        raise ValueError("Invalid review identifier")
    return {"configurable": {"thread_id": "v86-critic:" + review_id}}


def pending_review(snapshot) -> dict | None:
    for task in snapshot.tasks:
        for item in task.interrupts:
            if isinstance(item.value, dict) and str(item.value.get("kind", "")).startswith("v86_"):
                return item.value
    return None


class ReviewWorkflow:
    def __init__(self, store: ReviewStore, *,
                 provider_factory: Callable[[LLMRequest], LLMProvider] = create_bounded_provider, memory=None):
        self.store = store
        self.provider_factory = provider_factory
        self.memory = memory

    def _memory_snapshot(self, review_id):
        binding = self.store.get("memory_binding", review_id)
        context = self.store.get("memory_context", review_id)
        if binding is None and context is None:
            return None
        from research.memory.contracts import MemoryBinding, MemoryContext
        if binding is None or context is None:
            raise ValueError("Memory retrieval is incomplete")
        checked_binding = MemoryBinding.model_validate(binding)
        checked_context = MemoryContext.model_validate(context)
        if checked_context.project_id != checked_binding.project_id:
            raise PermissionError("Stored memory context belongs to another project")
        return checked_context.model_dump(mode="json")

    @staticmethod
    def _manager(request: ReviewInput):
        return open_evidence_ledger(request)

    def _inputs(self, state: ReviewState):
        request = ReviewInput.model_validate(state["input_document"])
        if self.store.require("session", request.review_id) != request.model_dump(mode="json"):
            raise PermissionError("Checkpoint policy differs from the immutable review session")
        bundle = load_input(request)
        recorded = self.store.require("bundle", request.review_id)
        if bundle.model_dump(mode="json") != recorded or state["bundle_document"] != recorded:
            raise ValueError("Original sources or checkpoint evidence changed during the review")
        binding = self.store.get("memory_binding", request.review_id)
        if binding is not None:
            if self.memory is None or self.memory.binding.model_dump(mode="json") != binding:
                raise PermissionError("Resume requires the recorded research memory binding")
            context = self.store.get("memory_context", request.review_id)
            if context is not None:
                from research.memory.contracts import MemoryContext
                self.memory.verify(MemoryContext.model_validate(context))
        return request, bundle

    def _cycle(self, state: ReviewState, request: ReviewInput) -> int:
        cycle = state.get("cycle", 1)
        if type(cycle) is not int or not 1 <= cycle <= request.max_cycles:
            raise PermissionError("Revision cycle exceeds the immutable session policy")
        if cycle > 1:
            previous = self.store.require("proposal", f"{request.review_id}-{cycle - 1}")
            decision = self.store.require("decision", f"{request.review_id}-revision-{cycle - 1}")
            if decision["decision"] != "request_changes" or decision["target_sha256"] != fingerprint(previous):
                raise PermissionError("Another revision cycle requires a prior human request for changes")
        return cycle

    def _report(self, request: ReviewInput, bundle: EvidenceBundle) -> ReviewReport:
        report = deterministic_review(bundle, self._manager(request))
        call = self.store.call(f"{request.review_id}-critique-1")
        if call and call["phase"] == "completed":
            llm_request = LLMRequest.model_validate(call["request"])
            expected = make_request(bundle, report, request.llm, review_id=request.review_id, role="critique",
                                    memory_context=self._memory_snapshot(request.review_id))
            if llm_request != expected:
                raise ValueError("Completed critic request differs from the review evidence")
            report = with_suggestions(bundle, report, CriticSuggestions.model_validate(call["result"]))
        return report

    def _frozen_report(self, state, request, bundle):
        report = self._report(request, bundle)
        saved = self.store.require("review", request.review_id)
        if saved != report.model_dump(mode="json") or state["report_document"] != saved:
            raise ValueError("Checkpoint report differs from the recorded evidence-grounded review")
        return report

    def load_node(self, state: ReviewState):
        request = ReviewInput.model_validate(state["input_document"])
        bundle = load_input(request)  # fails before allocating any LLM call
        self.store.put("session", request.review_id, request.review_id, request.model_dump(mode="json"))
        self.store.put("bundle", request.review_id, request.review_id, bundle.model_dump(mode="json"))
        if self.memory:
            self.store.put("memory_binding", request.review_id, request.review_id, self.memory.binding.model_dump(mode="json"))
        return {"input_document": request.model_dump(mode="json"), "bundle_document": bundle.model_dump(mode="json"),
                "cycle": 1, "feedback": "", "llm_approved": False, "status": "evidence_verified"}

    def retrieve_memory_node(self, state):
        request, bundle = self._inputs(state)
        if self.store.get("memory_context", request.review_id) is not None:
            recorded = self._memory_snapshot(request.review_id)
            # A crash after the immutable write must not select a different
            # context before the LangGraph checkpoint is committed.
            from research.memory.contracts import MemoryContext
            return {"memory_context_sha256": MemoryContext.model_validate(recorded).fingerprint()}
        question = (bundle.context.mission_document["mission"]["research_question"] if bundle.context
                    else bundle.source_document.get("metric_name", "synthetic mean"))
        if isinstance(bundle.analysis, RetryPilotAnalysis):
            question = bundle.analysis.question
        elif isinstance(bundle.analysis, RetryGPUAnalysis):
            question = bundle.analysis.measured_question
        elif isinstance(bundle.analysis, ProtocolAnalysis):
            question = bundle.source_document["mission"]["mission"]["research_question"]
        context = self.memory.retrieve("scientific_critic", "réplication refus protocole " + question,
                                       exclude_review_id=request.review_id)
        self.store.put("memory_context", request.review_id, request.review_id, context.model_dump(mode="json"))
        return {"memory_context_sha256": context.fingerprint()}

    def publish_memory_node(self, state):
        request, _ = self._inputs(state)
        items = self.memory.memory.ingest_review(self.memory.binding.project_id, self.store.path, request.review_id)
        return {"memory_indexed_items": len(items)}

    def rules_node(self, state):
        request, bundle = self._inputs(state)
        report = deterministic_review(bundle, self._manager(request))
        result = {"report_document": report.model_dump(mode="json"), "status": "deterministic_review"}
        if request.assist_critique:
            pending = make_request(bundle, report, request.llm, review_id=request.review_id, role="critique",
                                   memory_context=self._memory_snapshot(request.review_id))
            result.update(pending_call=pending.model_dump(mode="json"), status="awaiting_llm_approval")
        return result

    def _exact_call(self, state, request, bundle) -> LLMRequest:
        pending = LLMRequest.model_validate(state["pending_call"])
        cycle = self._cycle(state, request)
        if pending.role == "critique":
            report = deterministic_review(bundle, self._manager(request))
            expected = make_request(bundle, report, request.llm, review_id=request.review_id, role="critique",
                                    memory_context=self._memory_snapshot(request.review_id))
        else:
            report = self._frozen_report(state, request, bundle)
            expected = make_request(bundle, report, request.llm, review_id=request.review_id,
                                    role="revision", cycle=cycle, feedback=state.get("feedback", ""),
                                    memory_context=self._memory_snapshot(request.review_id))
        if pending != expected:
            raise PermissionError("Pending LLM request differs from the exact evidence/policy")
        return pending

    def llm_approval_node(self, state):
        request, bundle = self._inputs(state)
        pending = self._exact_call(state, request, bundle)
        response = interrupt({"kind": "v86_llm_approval", "target_sha256": pending.fingerprint(),
                              "request": pending.model_dump(mode="json"),
                              "notice": "Authorize only this LLM request; no experiment or protocol adoption."})
        request, bundle = self._inputs(state)  # interrupt replays the node: verify AFTER it too
        pending = self._exact_call(state, request, bundle)
        decision = HumanDecision.model_validate(response)
        if decision.target_sha256 != pending.fingerprint() or decision.decision not in {"approve", "reject"}:
            raise PermissionError("LLM decision does not match the pending request")
        self.store.put("decision", pending.call_id + "-authorization", request.review_id,
                       {"kind": "llm_authorization", **decision.model_dump(mode="json")})
        return {"authorization": decision.model_dump(mode="json"), "llm_approved": decision.decision == "approve",
                "status": "llm_authorized" if decision.decision == "approve" else "llm_rejected"}

    def _previous_digest(self, request, cycle):
        return (fingerprint(self.store.require("proposal", f"{request.review_id}-{cycle - 1}"))
                if cycle > 1 else None)

    def llm_call_node(self, state):
        request, bundle = self._inputs(state)
        pending = self._exact_call(state, request, bundle)
        if pending.role == "critique":
            report = deterministic_review(bundle, self._manager(request))
            validate = lambda text: validate_suggestions(text, bundle, report)
        else:
            report = self._frozen_report(state, request, bundle)

            def validate(text):
                draft = RevisionDraft.model_validate_json(text)
                build_revision(bundle, report, draft, review_id=request.review_id, cycle=pending.cycle,
                               previous_proposal_sha256=self._previous_digest(request, pending.cycle),
                               manager=self._manager(request))
                return draft.model_dump(mode="json")

        result = call_llm(pending, HumanDecision.model_validate(state["authorization"]), self.store,
                          validate=validate, provider_factory=self.provider_factory)
        if pending.role == "critique":
            reviewed = with_suggestions(bundle, report, CriticSuggestions.model_validate(result))
            return {"report_document": reviewed.model_dump(mode="json"), "status": "assisted_review"}
        return {"draft_document": result, "status": "draft_ready"}

    def save_review_node(self, state):
        request, bundle = self._inputs(state)
        report = self._report(request, bundle)
        if state["report_document"] != report.model_dump(mode="json"):
            raise ValueError("Review state differs from reconstructed facts/opinions")
        self.store.put("review", request.review_id, request.review_id, report.model_dump(mode="json"))
        return {"status": "review_complete"}

    def revision_input_node(self, state):
        request, bundle = self._inputs(state)
        cycle = self._cycle(state, request)
        report = self._frozen_report(state, request, bundle)
        if request.assist_revision:
            pending = make_request(bundle, report, request.llm, review_id=request.review_id,
                                   role="revision", cycle=cycle, feedback=state.get("feedback", ""),
                                   memory_context=self._memory_snapshot(request.review_id))
            return {"pending_call": pending.model_dump(mode="json"), "llm_approved": False,
                    "status": "awaiting_llm_approval"}
        token = fingerprint({"bundle": bundle.fingerprint(), "report": report.fingerprint(), "cycle": cycle})
        response = interrupt({"kind": "v86_revision_input", "target_sha256": token, "cycle": cycle,
                              "notice": "Supply a prospective draft or skip revision; no execution authorization."})
        request, bundle = self._inputs(state)
        self._frozen_report(state, request, bundle)
        self._cycle(state, request)
        submission = RevisionSubmission.model_validate(response)
        if submission.target_sha256 != token:
            raise PermissionError("Draft submission does not match this review/cycle")
        if submission.decision == "skip":
            return {"status": "review_only"}
        return {"draft_document": submission.draft.model_dump(mode="json"), "status": "draft_ready"}

    def validate_revision_node(self, state):
        request, bundle = self._inputs(state)
        cycle = self._cycle(state, request)
        report = self._frozen_report(state, request, bundle)
        proposal = build_revision(bundle, report, RevisionDraft.model_validate(state["draft_document"]),
                                  review_id=request.review_id, cycle=cycle,
                                  previous_proposal_sha256=self._previous_digest(request, cycle),
                                  manager=self._manager(request))
        self.store.put("proposal", f"{request.review_id}-{cycle}", request.review_id, proposal.model_dump(mode="json"))
        return {"proposal_document": proposal.model_dump(mode="json"), "status": "awaiting_revision_approval"}

    def _proposal(self, state, request, bundle):
        cycle = self._cycle(state, request)
        report = self._frozen_report(state, request, bundle)
        proposal = RevisionProposal.model_validate(self.store.require("proposal", f"{request.review_id}-{cycle}"))
        if (state["proposal_document"] != proposal.model_dump(mode="json")
                or proposal.previous_proposal_sha256 != self._previous_digest(request, cycle)):
            raise ValueError("Proposal state or previous-version link changed")
        verify_revision(bundle, report, proposal, self._manager(request))
        return proposal

    def revision_approval_node(self, state):
        request, bundle = self._inputs(state)
        proposal = self._proposal(state, request, bundle)
        response = interrupt({"kind": "v86_revision_approval", "target_sha256": proposal.fingerprint(),
                              "proposal": proposal.model_dump(mode="json"),
                              "notice": "Adopt a prospective protocol version only; no job may be submitted."})
        request, bundle = self._inputs(state)
        proposal = self._proposal(state, request, bundle)
        decision = HumanDecision.model_validate(response)
        if decision.target_sha256 != proposal.fingerprint() or decision.decision not in {"approve", "reject", "request_changes"}:
            raise PermissionError("Revision decision does not match the exact proposal")
        if decision.decision == "request_changes" and not decision.comment.strip():
            raise ValueError("Requesting changes requires an explanatory human comment")
        record = {"kind": "revision_decision", **decision.model_dump(mode="json")}
        self.store.put("decision", f"{request.review_id}-revision-{proposal.cycle}", request.review_id, record)
        if decision.decision == "approve":
            self.store.put("approved_revision", f"{request.review_id}-{proposal.cycle}", request.review_id,
                           {"schema_version": "8.6", "kind": "approved_prospective_revision",
                            "proposal": proposal.model_dump(mode="json"), "decision": record,
                            "matrix": revision_matrix(proposal), "execution_status": "not_executed"})
            return {"status": "revision_approved"}
        if decision.decision == "reject":
            return {"status": "revision_rejected"}
        if proposal.cycle >= request.max_cycles:
            return {"status": "cycle_limit"}
        return {"cycle": proposal.cycle + 1, "feedback": decision.comment, "llm_approved": False,
                "status": "needs_revision"}

    @staticmethod
    def _after_llm_approval(state):
        if state["llm_approved"]:
            return "llm_call"
        return "save_review" if state["pending_call"]["role"] == "critique" else END

    def build(self, saver):
        builder = StateGraph(ReviewState)
        for name, handler in (
            ("verify_evidence", self.load_node), ("rules", self.rules_node),
            ("llm_approval", self.llm_approval_node), ("llm_call", self.llm_call_node),
            ("save_review", self.save_review_node), ("revision_input", self.revision_input_node),
            ("validate_revision", self.validate_revision_node), ("revision_approval", self.revision_approval_node),
        ):
            builder.add_node(name, handler)
        builder.add_edge(START, "verify_evidence")
        if self.memory:
            builder.add_node("retrieve_research_memory", self.retrieve_memory_node)
            builder.add_node("publish_review_memory", self.publish_memory_node)
            builder.add_node("publish_decision_memory", self.publish_memory_node)
            builder.add_edge("verify_evidence", "retrieve_research_memory")
            builder.add_edge("retrieve_research_memory", "rules")
            builder.add_edge("save_review", "publish_review_memory")
            builder.add_edge("revision_approval", "publish_decision_memory")
        else:
            builder.add_edge("verify_evidence", "rules")
        builder.add_conditional_edges("rules", lambda s: "llm_approval" if s["status"] == "awaiting_llm_approval" else "save_review",
                                      {"llm_approval": "llm_approval", "save_review": "save_review"})
        builder.add_conditional_edges("llm_approval", self._after_llm_approval,
                                      {"llm_call": "llm_call", "save_review": "save_review", END: END})
        builder.add_conditional_edges("llm_call", lambda s: "save_review" if s["pending_call"]["role"] == "critique" else "validate_revision",
                                      {"save_review": "save_review", "validate_revision": "validate_revision"})
        builder.add_conditional_edges("publish_review_memory" if self.memory else "save_review", lambda s: "revision_input" if s["bundle_document"]["context"] else END,
                                      {"revision_input": "revision_input", END: END})
        builder.add_conditional_edges("revision_input", lambda s: {"awaiting_llm_approval": "llm_approval", "draft_ready": "validate_revision"}.get(s["status"], END),
                                      {"llm_approval": "llm_approval", "validate_revision": "validate_revision", END: END})
        builder.add_edge("validate_revision", "revision_approval")
        builder.add_conditional_edges("publish_decision_memory" if self.memory else "revision_approval", lambda s: "revision_input" if s["status"] == "needs_revision" else END,
                                      {"revision_input": "revision_input", END: END})
        return builder.compile(checkpointer=saver)
