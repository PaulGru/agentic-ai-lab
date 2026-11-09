
import unittest

from langgraph.checkpoint.memory import InMemorySaver

from agent.models import LLMResponse
from research.factory import build_research_graph
from research.source_audit import audit_sources
from tests.stateful_provider import StatefulFakeProvider


class V63SourceAuditTests(unittest.TestCase):

    def test_valid_sources_and_duplicates(self):
        sources = [
            {
                "title": "Guide A",
                "url": (
                    "https://docs.example.org/guide"
                    "?utm_source=openai"
                ),
            },
            {
                "title": "Guide A duplicate",
                "url": "https://docs.example.org/guide",
            },
            {
                "title": "Guide B",
                "url": "https://other.example.org/guide",
            },
            {
                "title": "Invalid",
                "url": "javascript:alert(1)",
            },
        ]

        cleaned, report = audit_sources(
            sources=sources,
            web_search_used=True,
        )

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(report["source_count"], 2)
        self.assertEqual(report["distinct_domains"], 2)
        self.assertEqual(report["duplicate_urls"], 1)
        self.assertEqual(report["invalid_urls"], 1)

        self.assertEqual(
            report["status"],
            "citations_available",
        )
        self.assertFalse(report["content_verified"])

    def test_search_without_citations(self):
        cleaned, report = audit_sources(
            sources=[],
            web_search_used=True,
        )

        self.assertEqual(cleaned, [])
        self.assertEqual(
            report["status"],
            "search_without_citations",
        )
        self.assertIn(
            "web_search_without_citations",
            report["warnings"],
        )

    def test_no_web_search(self):
        _, report = audit_sources(
            sources=[],
            web_search_used=False,
        )

        self.assertEqual(
            report["status"],
            "not_searched",
        )

    def test_audit_is_checkpointed(self):
        saver = InMemorySaver()

        fake = StatefulFakeProvider([
            LLMResponse(
                text="Réponse avec une référence.",
                web_search_used=True,
                sources=[
                    {
                        "title": "Documentation",
                        "url": "https://docs.example.org/guide",
                    }
                ],
            )
        ])

        config = {
            "configurable": {
                "thread_id": "test-v63-audit"
            }
        }

        graph = build_research_graph(
            saver=saver,
            provider=fake,
        )

        result = graph.invoke(
            {"prompt": "Recherche une documentation."},
            config=config,
        )

        self.assertEqual(
            result["source_audit"]["status"],
            "citations_available",
        )

        restarted_graph = build_research_graph(
            saver=saver,
            provider=StatefulFakeProvider([]),
        )

        snapshot = restarted_graph.get_state(config)

        self.assertEqual(snapshot.next, ())
        self.assertEqual(
            snapshot.values["source_audit"]["source_count"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
