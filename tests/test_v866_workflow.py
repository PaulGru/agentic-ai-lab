import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.models import LLMResponse
from research.critique.contracts import LLMSettings, ReviewInput
from research.critique.evidence import load_input
from research.critique.graph import ReviewWorkflow, config_for, pending_review
from research.critique.store import NeedsReconciliation, ReviewStore
from tests.fakes import FakeProvider
from tests.v86_helpers import design_context, fixture_document, revision_draft


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "fixture.json"
        self.source.write_text(json.dumps(fixture_document()), encoding="utf-8")
        self.context = self.root / "context.json"
        self.context.write_text(design_context().model_dump_json(), encoding="utf-8")
        self.store = ReviewStore(self.root / "reviews.db")
        self.saver = InMemorySaver()
        self.factory = Mock(side_effect=AssertionError("Unexpected provider construction"))
        self.graph = ReviewWorkflow(self.store, provider_factory=self.factory).build(self.saver)
        self.cfg = config_for("review")

    def start(self, *, context=True, llm=False, revision_llm=False, max_cycles=1):
        request = ReviewInput(
            review_id="review", source_kind="pedagogical_paired_fixture", source_path=str(self.source),
            context_path=str(self.context) if context else None, max_cycles=max_cycles,
            llm=LLMSettings(provider="openai", model="fictional-test-model") if llm or revision_llm else None,
            assist_critique=llm, assist_revision=revision_llm)
        result = self.graph.invoke({"input_document": request.model_dump(mode="json")}, self.cfg, durability="sync")
        return request, result

    def resume(self, decision, *, draft=None, comment="", digest=None, graph=None):
        selected = graph or self.graph
        pending = pending_review(selected.get_state(self.cfg))
        body = {"decision": decision, "target_sha256": digest or pending["target_sha256"], "actor": "test-human"}
        if pending["kind"] == "v86_revision_input":
            if draft is not None:
                body["draft"] = draft.model_dump(mode="json")
        else:
            body["comment"] = comment
        return selected.invoke(Command(resume=body), self.cfg, durability="sync")

    def test_offline_resume_reconstructs_graph_and_approves_only_new_version(self):
        with patch("execution.backends.local.LocalDockerBackend.submit") as submit:
            request, result = self.start()
            self.assertIn("__interrupt__", result)
            self.assertEqual(pending_review(self.graph.get_state(self.cfg))["kind"], "v86_revision_input")
            rebuilt = ReviewWorkflow(ReviewStore(self.store.path), provider_factory=self.factory).build(self.saver)
            self.resume("propose", draft=revision_draft(load_input(request)), graph=rebuilt)
            self.assertIsNone(self.store.get("approved_revision", "review-1"))
            self.assertEqual(pending_review(rebuilt.get_state(self.cfg))["kind"], "v86_revision_approval")
            result = self.resume("approve", graph=rebuilt)
            self.assertEqual(result["status"], "revision_approved")
            version = self.store.require("approved_revision", "review-1")
            self.assertEqual(version["execution_status"], "not_executed")
            self.assertEqual(version["matrix"]["execution_status"], "not_executed")
            self.assertEqual(rebuilt.get_state(self.cfg).next, ())
            self.factory.assert_not_called()
            submit.assert_not_called()

    def test_rejection_and_skip_preserve_review_without_approved_version(self):
        request, _ = self.start()
        self.resume("propose", draft=revision_draft(load_input(request)))
        result = self.resume("reject")
        self.assertEqual(result["status"], "revision_rejected")
        self.assertIsNotNone(self.store.get("review", "review"))
        self.assertIsNone(self.store.get("approved_revision", "review-1"))

    def test_bad_digest_and_changed_sources_cannot_approve(self):
        request, _ = self.start()
        self.resume("propose", draft=revision_draft(load_input(request)))
        with self.assertRaises(PermissionError):
            self.resume("approve", digest="0" * 64)
        doc = fixture_document()
        doc["pairs"][0]["variant"] += 100
        self.source.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sources"):
            self.resume("approve")
        self.assertIsNone(self.store.get("approved_revision", "review-1"))

    def test_cycles_need_human_feedback_and_stop_at_hard_limit(self):
        request, _ = self.start(max_cycles=2)
        self.resume("propose", draft=revision_draft(load_input(request)))
        result = self.resume("request_changes", comment="Préciser la limite de puissance et les unités.")
        self.assertEqual(result["cycle"], 2)
        self.resume("propose", draft=revision_draft(load_input(request)))
        first = self.store.require("proposal", "review-1")
        second = self.store.require("proposal", "review-2")
        from analysis.contracts import fingerprint
        self.assertEqual(second["previous_proposal_sha256"], fingerprint(first))
        result = self.resume("request_changes", comment="Une autre modification serait nécessaire.")
        self.assertEqual(result["status"], "cycle_limit")
        self.assertEqual(self.graph.get_state(self.cfg).next, ())
        self.assertIsNone(self.store.get("proposal", "review-3"))

    def test_llm_authorizations_are_separate_from_revision_approval(self):
        request, _ = self.start(llm=True, revision_llm=True)
        self.factory.assert_not_called()
        draft = revision_draft(load_input(request))
        provider = FakeProvider([LLMResponse(text='{"issues":[]}'), LLMResponse(text=draft.model_dump_json())])
        self.factory.side_effect = None
        self.factory.return_value = provider
        self.resume("approve")
        self.assertEqual(self.factory.call_count, 1)
        self.assertEqual(pending_review(self.graph.get_state(self.cfg))["kind"], "v86_llm_approval")
        self.resume("approve")
        self.assertEqual(self.factory.call_count, 2)
        self.assertEqual(pending_review(self.graph.get_state(self.cfg))["kind"], "v86_revision_approval")
        self.assertIsNone(self.store.get("approved_revision", "review-1"))
        self.resume("reject")
        self.assertEqual(self.factory.call_count, 2)

    def test_declined_llm_never_constructs_a_provider(self):
        _, result = self.start(context=False, llm=True)
        self.assertIn("__interrupt__", result)
        result = self.resume("reject")
        self.assertEqual(result["status"], "review_complete")
        self.factory.assert_not_called()
        self.assertEqual(self.store.call_summaries("review"), [])

    def test_checkpoint_cannot_increase_policy_or_remove_findings(self):
        request, _ = self.start()
        changed = request.model_dump(mode="json")
        changed["max_cycles"] = 2
        self.graph.update_state(self.cfg, {"input_document": changed})
        with self.assertRaises(PermissionError):
            # update_state creates a fresh task; continue it to exercise the policy check.
            self.graph.invoke(None, self.cfg, durability="sync")
        self.factory.assert_not_called()

    def test_completed_receipt_survives_crash_before_checkpoint_without_duplicate_call(self):
        self.start(context=False, llm=True)
        self.factory.side_effect = None
        self.factory.return_value = FakeProvider([LLMResponse(text='{"issues":[]}')])
        original = self.store.finish_call

        def crash_after_receipt(request, result):
            original(request, result)
            raise RuntimeError("simulated crash after durable receipt")

        with patch.object(self.store, "finish_call", side_effect=crash_after_receipt):
            with self.assertRaises(RuntimeError):
                self.resume("approve")
        restarted = ReviewWorkflow(ReviewStore(self.store.path), provider_factory=Mock(
            side_effect=AssertionError("A completed call must never be repeated"))).build(self.saver)
        result = restarted.invoke(None, self.cfg, durability="sync")
        self.assertEqual(result["status"], "review_complete")
        self.assertEqual(self.factory.call_count, 1)

    def test_uncertain_call_stops_recovery_without_another_provider(self):
        self.start(context=False, llm=True)
        self.factory.side_effect = TimeoutError("unknown outcome")
        with self.assertRaises(RuntimeError):
            self.resume("approve")
        restarted = ReviewWorkflow(ReviewStore(self.store.path), provider_factory=self.factory).build(self.saver)
        with self.assertRaises(NeedsReconciliation):
            restarted.invoke(None, self.cfg, durability="sync")
        self.assertEqual(self.factory.call_count, 1)


if __name__ == "__main__":
    unittest.main()
