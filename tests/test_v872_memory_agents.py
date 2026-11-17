import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.models import LLMResponse
from providers.openai_provider import OpenAIProvider
from providers.anthropic_provider import AnthropicProvider
from research.critique.contracts import LLMSettings, ReviewInput
from research.critique.evidence import load_input
from research.critique.graph import ReviewWorkflow, config_for, pending_review
from research.critique.store import ReviewStore
from research.design_graph import build_design_graph, pending_design_approval
from research.design_planner import ExperimentalDesignPlanner
from research.literature_analyst import LiteratureAnalyst
from research.memory.contracts import MemoryBinding
from research.memory.integration import MemorySession
from research.memory.service import ResearchMemory
from research.memory.stores import SQLiteMemoryStore
from research.mission import ResearchMission
from research.mission_planner import MissionPlanner
from tests.fakes import FakeProvider
from tests.test_v81_mission import valid_mission
from tests.test_v82_literature import fake_catalog, valid_review
from tests.test_v83_design import valid_design
from tests.v86_helpers import design_context, fixture_document, revision_draft


class MemoryAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.context_path = self.root / "context.json"
        self.context = design_context()
        self.context_path.write_text(self.context.model_dump_json(), encoding="utf-8")
        self.fixture_path = self.root / "fixture.json"
        self.fixture_path.write_text(json.dumps(fixture_document()), encoding="utf-8")
        self.memstore = SQLiteMemoryStore(self.root / "memory.db")
        self.memstore.setup()
        self.memory = ResearchMemory(self.memstore)
        self.binding = MemoryBinding(project_id="project-A", path=str(self.memstore.path))
        self.session = MemorySession(self.memory, self.binding)
        self.reviews = ReviewStore(self.root / "reviews.db")
        # A completed, rejected research episode is retained in the durable source ledger.
        request = self.input("old-rejected")
        graph = ReviewWorkflow(self.reviews).build(InMemorySaver())
        cfg = config_for(request.review_id)
        graph.invoke({"input_document": request.model_dump(mode="json")}, cfg)
        pending = pending_review(graph.get_state(cfg))
        graph.invoke(Command(resume={"decision": "propose", "target_sha256": pending["target_sha256"],
                                     "actor": "human", "draft": revision_draft(load_input(request)).model_dump(mode="json")}), cfg)
        pending = pending_review(graph.get_state(cfg))
        graph.invoke(Command(resume={"decision": "reject", "target_sha256": pending["target_sha256"],
                                     "actor": "human", "comment": "Refus : réplication et indépendance à justifier."}), cfg)
        self.memory.ingest_review("project-A", self.reviews.path, request.review_id)

    def input(self, review_id, llm=False):
        return ReviewInput(review_id=review_id, source_kind="pedagogical_paired_fixture",
                            source_path=str(self.fixture_path), context_path=str(self.context_path),
                            llm=LLMSettings(provider="openai", model="fictional-model") if llm else None,
                            assist_critique=llm)

    def test_new_session_recalls_the_reason_for_rejection_with_exact_references(self):
        reopened = ResearchMemory(SQLiteMemoryStore(self.memstore.path))
        context = reopened.search("project-A", "réplication rejeté protocole", kinds=["decision"])
        self.assertEqual(len(context.hits), 1)
        hit = context.hits[0]
        self.assertIn("Refus", hit.excerpt)
        self.assertFalse(hit.action_authorization)
        self.assertFalse(hit.scientific_evidence)
        source = reopened.get("project-A", hit.memory_id)["source"]
        self.assertEqual(source["origin"]["review_id"], "old-rejected")
        self.assertEqual(hit.source_pointer, "/decision")
        self.assertFalse(reopened.search("project-B", "réplication rejeté protocole").hits)

    def test_planners_and_literature_receive_memory_as_data_without_changing_evidence_catalog(self):
        mission = ResearchMission.model_validate(self.context.mission_document["mission"])
        mission_provider = FakeProvider([LLMResponse(text=json.dumps(valid_mission()))])
        planner = MissionPlanner(mission_provider, memory=self.session)
        planner.propose({"Question": "réplication refus protocole"})
        self.assertTrue(planner.last_memory_context.hits)
        self.assertIn("research_memory", json.loads(mission_provider.start_calls[0]["prompt"]))
        lit_provider = FakeProvider([LLMResponse(text=json.dumps(valid_review()))])
        LiteratureAnalyst(lit_provider, memory=self.session).analyze(mission, fake_catalog())
        payload = json.loads(lit_provider.start_calls[0]["prompt"])
        self.assertEqual(len(payload["catalog"]), 3)
        self.assertTrue(payload["research_memory"]["hits"])
        design_provider = FakeProvider([LLMResponse(text=json.dumps(valid_design()))])
        from research.literature import LiteratureReview
        ExperimentalDesignPlanner(design_provider, memory=self.session).propose(mission, LiteratureReview.model_validate(valid_review()))
        payload = json.loads(design_provider.start_calls[0]["prompt"])
        self.assertTrue(payload["research_memory"]["hits"])
        self.assertEqual(len(payload["literature"]["findings"]), 1)
        for provider in (mission_provider, lit_provider, design_provider):
            self.assertEqual(provider.start_calls[0]["tools"], [])
            self.assertEqual(provider.continue_calls, [])

    def test_same_memory_with_mocked_openai_and_anthropic_clients(self):
        text = json.dumps(valid_mission())
        openai = Mock()
        openai.responses.create.return_value = SimpleNamespace(id="resp", output_text=text, output=[])
        anthropic = Mock()
        block = Mock(type="text", text=text, citations=[])
        block.model_dump.return_value = {"type": "text", "text": text}
        anthropic.messages.create.return_value = SimpleNamespace(content=[block])
        question = {"Question": "réplication refus protocole"}
        with patch("providers.openai_provider.OpenAI", return_value=openai):
            MissionPlanner(OpenAIProvider("fictional-model"), memory=self.session).propose(question)
        with patch("providers.anthropic_provider.Anthropic", return_value=anthropic):
            other_session = MemorySession(ResearchMemory(SQLiteMemoryStore(self.memstore.path)), self.binding)
            MissionPlanner(AnthropicProvider("fictional-model"), memory=other_session).propose(question)
        first = json.loads(openai.responses.create.call_args.kwargs["input"])["research_memory"]
        second = json.loads(anthropic.messages.create.call_args.kwargs["messages"][0]["content"])["research_memory"]
        self.assertEqual(first, second)

    def test_design_human_approval_binds_memory_and_checks_it_after_pause(self):
        saver = InMemorySaver()
        graph = build_design_graph(saver, provider_factory=lambda: FakeProvider([LLMResponse(text=json.dumps(valid_design()))]), memory=self.session)
        cfg = {"configurable": {"thread_id": "v83-memory-test"}}
        graph.invoke({"mission_document": self.context.mission_document,
                      "literature_document": self.context.literature_document}, cfg)
        pending = pending_design_approval(graph.get_state(cfg))
        self.assertIn("memory_context_sha256", pending)
        context = graph.get_state(cfg).values["memory_context"]
        item = self.memory.store.item("project-A", context["hits"][0]["memory_id"])
        self.memory.change("project-A", item.memory_id, action="invalidate", target_sha256=item.fingerprint(),
                           actor="human", reason="Ce souvenir est invalidé avant la décision.")
        resume = {"decision": "approve", **{key: pending[key] for key in (
            "mission_sha256", "literature_input_sha256", "design_sha256", "memory_context_sha256")}}
        fresh = build_design_graph(saver, provider_factory=lambda: self.fail("No LLM during resume"), memory=self.session)
        with self.assertRaises(ValueError):
            fresh.invoke(Command(resume=resume), cfg)

    def test_critic_uses_frozen_context_then_indexes_its_completed_review(self):
        saver = InMemorySaver()
        factory = Mock(return_value=FakeProvider([LLMResponse(text='{"issues":[]}')]))
        graph = ReviewWorkflow(self.reviews, provider_factory=factory, memory=self.session).build(saver)
        request = self.input("new-session", llm=True)
        cfg = config_for(request.review_id)
        graph.invoke({"input_document": request.model_dump(mode="json")}, cfg)
        pending = pending_review(graph.get_state(cfg))
        payload = json.loads(pending["request"]["prompt"])
        self.assertTrue(payload["research_memory"]["hits"])
        factory.assert_not_called()
        graph.invoke(Command(resume={"decision": "approve", "target_sha256": pending["target_sha256"], "actor": "human"}), cfg)
        self.assertEqual(factory.call_count, 1)
        self.assertGreater(graph.get_state(cfg).values["memory_indexed_items"], 0)
        self.assertEqual(pending_review(graph.get_state(cfg))["kind"], "v86_revision_input")
        # A historical reconstruction can inspect the recorded LLM request without reconnecting memory.
        from scripts.demo_v86 import verify_recorded_review
        verify_recorded_review(self.reviews, request.review_id)

    def test_invalidated_selected_memory_blocks_llm_before_any_call(self):
        factory = Mock(side_effect=AssertionError("No provider before exact verified approval"))
        saver = InMemorySaver()
        graph = ReviewWorkflow(self.reviews, provider_factory=factory, memory=self.session).build(saver)
        request = self.input("paused", llm=True)
        cfg = config_for(request.review_id)
        graph.invoke({"input_document": request.model_dump(mode="json")}, cfg)
        pending = pending_review(graph.get_state(cfg))
        selected = self.reviews.require("memory_context", request.review_id)["hits"][0]
        self.memory.change("project-A", selected["memory_id"], action="invalidate", target_sha256=selected["item_sha256"],
                           actor="human", reason="Ce souvenir a été retiré pendant la pause.")
        with self.assertRaises(ValueError):
            graph.invoke(Command(resume={"decision": "approve", "target_sha256": pending["target_sha256"], "actor": "human"}), cfg)
        factory.assert_not_called()
        self.assertEqual(self.reviews.call_summaries(request.review_id), [])

    def test_crash_after_retrieval_write_reuses_the_frozen_context(self):
        saver = InMemorySaver()
        graph = ReviewWorkflow(self.reviews, memory=self.session).build(saver)
        request = self.input("retry-retrieval")
        cfg = config_for(request.review_id)
        original_put = self.reviews.put

        def crash_after_write(kind, *args, **kwargs):
            result = original_put(kind, *args, **kwargs)
            if kind == "memory_context":
                raise OSError("simulated crash before checkpoint")
            return result

        with patch.object(self.reviews, "put", side_effect=crash_after_write):
            with self.assertRaises(OSError):
                graph.invoke({"input_document": request.model_dump(mode="json")}, cfg, durability="sync")
        recorded = self.reviews.require("memory_context", request.review_id)
        note = self.memory.note("project-A", title="Nouvelle connaissance après arrêt",
                                text="Réplication, refus et protocole à revoir.", logical_key="arrived-after-crash", actor="human")
        self.memory.change("project-A", note.memory_id, action="confirm", target_sha256=note.fingerprint(),
                           actor="human", reason="Note relue après l'interruption du processus.")
        rebuilt = ReviewWorkflow(self.reviews, memory=self.session).build(saver)
        with patch.object(self.session, "retrieve", side_effect=AssertionError("Do not replace frozen context")):
            rebuilt.invoke(None, cfg, durability="sync")
        self.assertEqual(self.reviews.require("memory_context", request.review_id), recorded)
        self.assertNotIn(note.memory_id, [hit["memory_id"] for hit in recorded["hits"]])
        self.assertGreater(rebuilt.get_state(cfg).values["memory_indexed_items"], 0)

    def test_context_retrieval_failure_never_creates_provider_and_publication_is_retryable(self):
        saver = InMemorySaver()
        factory = Mock(side_effect=AssertionError("No provider"))
        graph = ReviewWorkflow(self.reviews, provider_factory=factory, memory=self.session).build(saver)
        request = self.input("retry-publication")
        cfg = config_for(request.review_id)
        with patch.object(self.memory, "ingest_review", side_effect=OSError("simulated index unavailable")):
            with self.assertRaises(OSError):
                graph.invoke({"input_document": request.model_dump(mode="json")}, cfg, durability="sync")
        self.assertIsNotNone(self.reviews.get("review", request.review_id))
        rebuilt = ReviewWorkflow(self.reviews, provider_factory=factory, memory=self.session).build(saver)
        rebuilt.invoke(None, cfg, durability="sync")
        self.assertGreater(rebuilt.get_state(cfg).values["memory_indexed_items"], 0)
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
