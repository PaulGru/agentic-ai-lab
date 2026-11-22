"""Offline V8.2 tests: fake OpenAlex payloads, fake LLM, in-memory checkpoints."""

import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import ValidationError

from agent.models import LLMResponse, ToolCall
from research.literature import (
    LiteratureCatalog, LiteratureReview, digest_json, validate_approved_mission,
    validate_evidence,
)
from research.literature_analyst import LiteratureAnalyst
from research.literature_graph import build_literature_graph, pending_literature_approval
from research.literature_search import (
    OpenAlexClient, InsufficientLiteratureError, reconstruct_abstract, retrieve_catalog,
)
from research.mission import ResearchMission, mission_digest
from scripts.tutorials.literature_review import config_for, export_approved_literature
from tests.fakes import FakeProvider


def mission_data():
    return {
        "title": "Robustesse des agents LLM",
        "research_question": "Quels facteurs influencent la robustesse des agents utilisant des outils ?",
        "objective": "Établir les facteurs de robustesse à protocole comparable",
        "hypotheses": [{"id": "H1", "claim": "Une stratégie de récupération réduit certains échecs de tâche",
                        "falsification_test": "Aucun effet positif n'est observé sous perturbation"}],
        "independent_variables": ["Stratégie d'orchestration"],
        "dependent_metrics": ["Taux de réussite par type de perturbation"],
        "controls": ["Même échantillon de tâches et mêmes outils"],
        "baselines": ["Agent sans reprise après erreur"],
        "literature_queries": ["LLM agent robustness", "tool use reliability"],
        "experiment_outline": ["Construire benchmark", "Tester chaque configuration"],
        "validity_risks": ["Budget d'inférence non comparable"],
        "success_criteria": ["Intervalles d'incertitude calculés"],
        "deliverables": ["Protocole reproductible"],
    }


def approved_document():
    m = mission_data()
    return {
        "schema_version": "8.1",
        "brief_sha256": "a" * 64,
        "mission_sha256": mission_digest(m),
        "brief_sections": {"Question": m["research_question"]},
        "mission": m,
    }


def inverted(text):
    return {word: [i] for i, word in enumerate(text.split())}


def make_work(work_id, title, abstract, doi=None):
    return {
        "id": f"https://openalex.org/{work_id}",
        "display_name": title,
        "publication_year": 2025,
        "authorships": [{"author": {"display_name": "A. Author"}}],
        "doi": doi,
        "primary_location": {"source": {"display_name": "Research Venue"}},
        "type": "article",
        "cited_by_count": 12,
        "abstract_inverted_index": inverted(abstract) if abstract else None,
    }


WORKS = [
    make_work("W101", "Agents and tools in controlled environments",
              "Agent systems can fail after tool errors and interruptions", "https://doi.org/10.5555/example.101"),
    make_work("W102", "Robust evaluation of agent systems",
              "Evaluation protocols must control costs and task complexity", "https://doi.org/10.5555/example.102"),
    make_work("W103", "Designing agent benchmarks", None),
]


class FakeOpenAlex:
    def __init__(self, works=WORKS):
        self.works = works
        self.urls = []
        self.headers = []

    def __call__(self, url, headers):
        self.urls.append(url)
        self.headers.append(headers)
        return {"meta": {"count": len(self.works)}, "results": self.works}


def valid_review():
    return {
        "overview": "Exploration limitée aux résumés disponibles et à des recherches bornées.",
        "findings": [{
            "work_id": "W101",
            "claim": "Le résumé aborde des défaillances liées aux outils et interruptions.",
            "abstract_excerpt": "can fail after tool errors and interruptions",
        }],
        "possible_gaps": ["Il reste à vérifier si la comparaison à budget fixe est déjà étudiée."],
        "methodological_risks": ["Le contrôle du budget de calcul peut changer la comparaison."],
        "next_queries": ["agent recovery ablation experiments"],
        "questions_for_v83": ["Comment mesurer séparément les échecs des outils et des agents ?"],
        "coverage_limits": ["Trois notices seulement, pas de lecture de texte intégral."],
    }


def fake_catalog():
    client = OpenAlexClient(transport=FakeOpenAlex())
    return retrieve_catalog(ResearchMission.model_validate(mission_data()), client)


def cfg(name):
    return config_for("test-" + name)


class V82LiteratureTests(unittest.TestCase):
    def test_approved_mission_is_required(self):
        doc = approved_document()
        validate_approved_mission(doc)
        doc["mission"]["title"] = "Unapproved changed title"
        with self.assertRaises(ValueError):
            validate_approved_mission(doc)

    def test_openalex_queries_are_bounded_and_traced(self):
        transport = FakeOpenAlex()
        client = OpenAlexClient(api_key="", transport=transport)
        catalog = retrieve_catalog(ResearchMission.model_validate(mission_data()), client)
        self.assertEqual(len(transport.urls), 2)
        self.assertEqual(len(catalog.papers), 3)  # Duplicate records across queries.
        self.assertEqual(catalog.papers[0].matched_queries, mission_data()["literature_queries"])
        query = parse_qs(urlsplit(transport.urls[0]).query)
        self.assertIn("search.title_and_abstract", query)
        self.assertEqual(query["per-page"], ["5"])
        self.assertNotIn("Authorization", transport.headers[0])
        self.assertEqual(catalog.provider, "openalex")
        self.assertIn("tool errors", catalog.papers[0].abstract)

    def test_query_override_keeps_original_mission_and_traces_search(self):
        original = ResearchMission.model_validate(mission_data())
        transport = FakeOpenAlex()
        queries = ["language agents evaluation", "multi agent planning"]
        catalog = retrieve_catalog(
            original, OpenAlexClient(transport=transport),
            queries_override=queries,
        )
        self.assertEqual(catalog.requested_queries, queries)
        self.assertEqual(len(catalog.papers), 3)
        self.assertEqual(catalog.query_log[0]["returned"], 3)
        self.assertEqual(catalog.query_log[1]["duplicates"], 3)
        self.assertEqual(original.literature_queries, mission_data()["literature_queries"])
        self.assertIn("language+agents+evaluation", transport.urls[0])
        self.assertEqual(len(transport.urls), 2)

    def test_empty_or_duplicate_overrides_are_rejected_without_search(self):
        transport = FakeOpenAlex()
        for queries in ([], ["topic", "topic"], ["x"], ["a\nbbbbb"], ["valid query"] * 5):
            with self.subTest(queries=queries), self.assertRaises(ValueError):
                retrieve_catalog(
                    ResearchMission.model_validate(mission_data()),
                    OpenAlexClient(transport=transport),
                    queries_override=queries,
                )
        self.assertEqual(transport.urls, [])

    def test_search_failure_includes_per_query_diagnostics(self):
        transport = FakeOpenAlex([WORKS[0]])
        with self.assertRaises(InsufficientLiteratureError) as captured:
            retrieve_catalog(
                ResearchMission.model_validate(mission_data()),
                OpenAlexClient(transport=transport),
                queries_override=["LLM agent robustness"],
            )
        self.assertIn("returned=1", str(captured.exception))
        self.assertIn("accepted=1", str(captured.exception))
        self.assertIn("minimum", str(captured.exception).lower())

    def test_key_is_header_not_url(self):
        transport = FakeOpenAlex()
        OpenAlexClient(api_key="dummy-secret", transport=transport).search("agent robustness", 2)
        self.assertNotIn("dummy-secret", transport.urls[0])
        self.assertEqual(transport.headers[0]["Authorization"], "Bearer dummy-secret")

    def test_short_abstract_and_insufficient_catalog_block(self):
        bad = FakeOpenAlex([WORKS[0]])
        with self.assertRaisesRegex(InsufficientLiteratureError, "1 unique paper"):
            retrieve_catalog(ResearchMission.model_validate(mission_data()), OpenAlexClient(transport=bad))
        self.assertIsNone(reconstruct_abstract({"bad": [2]}))

    def test_quote_must_be_from_retrieved_abstract(self):
        catalog = fake_catalog()
        good = LiteratureReview.model_validate(valid_review())
        validate_evidence(good, catalog)
        forged = valid_review()
        forged["findings"][0]["abstract_excerpt"] = "invented evidence that no source supports"
        with self.assertRaisesRegex(ValueError, "Quote not found"):
            validate_evidence(LiteratureReview.model_validate(forged), catalog)
        forged = valid_review()
        forged["findings"][0]["work_id"] = "W999999"
        with self.assertRaisesRegex(ValueError, "Unretrieved citation"):
            validate_evidence(LiteratureReview.model_validate(forged), catalog)

    def test_analyst_uses_no_local_tool(self):
        provider = FakeProvider([LLMResponse(text=json.dumps(valid_review()))])
        analysis = LiteratureAnalyst(provider).analyze(
            ResearchMission.model_validate(mission_data()), fake_catalog()
        )
        self.assertEqual(analysis.findings[0].work_id, "W101")
        self.assertEqual(provider.start_calls[0]["tools"], [])
        self.assertEqual(provider.continue_calls, [])

    def test_analyst_rejects_tool_call(self):
        provider = FakeProvider([LLMResponse(tool_calls=[
            ToolCall(id="call", name="calculate", arguments={"expression": "2+2"})
        ])])
        with self.assertRaises(ValueError):
            LiteratureAnalyst(provider).analyze(
                ResearchMission.model_validate(mission_data()), fake_catalog()
            )

    def test_workflow_interrupt_and_resume_after_reconstruction(self):
        saver = InMemorySaver()
        transport = FakeOpenAlex()
        provider = FakeProvider([LLMResponse(text=json.dumps(valid_review()))])
        graph = build_literature_graph(
            saver,
            client=OpenAlexClient(transport=transport),
            provider_factory=lambda: provider,
        )
        config = cfg("resume")
        result = graph.invoke({"mission_document": approved_document()}, config=config)
        self.assertIn("__interrupt__", result)
        pending = pending_literature_approval(graph.get_state(config))
        self.assertIsNotNone(pending)
        self.assertEqual(len(transport.urls), 2)
        self.assertEqual(pending["paper_count"], 3)
        restarted = build_literature_graph(
            saver,
            client=OpenAlexClient(transport=lambda *_: self.fail("Unexpected refetch")),
            provider_factory=lambda: self.fail("Unexpected LLM call"),
        )
        restarted.invoke(Command(resume={
            "decision": "approve",
            "mission_sha256": pending["mission_sha256"],
            "catalog_sha256": pending["catalog_sha256"],
            "review_sha256": pending["review_sha256"],
        }), config=config)
        final = restarted.get_state(config)
        self.assertEqual(final.values["status"], "approved")
        self.assertEqual(final.next, ())
        with tempfile.TemporaryDirectory() as directory:
            a, b = export_approved_literature(final.values, "resume", Path(directory))
            self.assertTrue(a.is_file())
            self.assertTrue(b.is_file())
            self.assertEqual(json.loads(a.read_text(encoding="utf-8"))["schema_version"], "8.2")
            self.assertIn("W101", b.read_text(encoding="utf-8"))
            # Export can be retried if identical; changes are never silently overwritten.
            export_approved_literature(final.values, "resume", Path(directory))
            b.write_text("tampered", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                export_approved_literature(final.values, "resume", Path(directory))

    def test_wrong_digest_rejected_without_new_calls(self):
        saver = InMemorySaver()
        graph = build_literature_graph(
            saver,
            client=OpenAlexClient(transport=FakeOpenAlex()),
            provider_factory=lambda: FakeProvider([LLMResponse(text=json.dumps(valid_review()))]),
        )
        config = cfg("digest")
        graph.invoke({"mission_document": approved_document()}, config=config)
        pending = pending_literature_approval(graph.get_state(config))
        graph.invoke(Command(resume={
            "decision": "approve",
            "mission_sha256": pending["mission_sha256"],
            "catalog_sha256": "incorrect",
            "review_sha256": pending["review_sha256"],
        }), config=config)
        self.assertEqual(graph.get_state(config).values["status"], "rejected")

    def test_insufficient_catalog_does_not_call_llm(self):
        calls = []
        graph = build_literature_graph(
            InMemorySaver(),
            client=OpenAlexClient(transport=FakeOpenAlex([WORKS[0]])),
            provider_factory=lambda: calls.append("called"),
        )
        with self.assertRaises(InsufficientLiteratureError):
            graph.invoke({"mission_document": approved_document()}, config=cfg("few"))
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
