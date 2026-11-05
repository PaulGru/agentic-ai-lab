
import unittest

from langgraph.checkpoint.memory import InMemorySaver

from agent.models import LLMResponse
from research.factory import build_research_graph
from tests.stateful_provider import StatefulFakeProvider


class V61ResearchTests(unittest.TestCase):

    def test_dynamic_question_and_checkpoint(self):
        saver = InMemorySaver()

        question = "Explique le principe du RAG."
        expected = "Le RAG combine recherche et génération."

        provider = StatefulFakeProvider([
            LLMResponse(text=expected),
        ])

        config = {
            "configurable": {
                "thread_id": "v6-research:test-001"
            }
        }

        graph = build_research_graph(
            saver, provider=provider
        )

        result = graph.invoke(
            {"prompt": question},
            config=config,
        )

        self.assertEqual(result["answer"], expected)
        self.assertEqual(
            provider.start_calls[0]["prompt"],
            question,
        )

        # Simuler une nouvelle instance de notre graphe.
        restarted_graph = build_research_graph(
            saver,
            provider=StatefulFakeProvider([]),
        )

        snapshot = restarted_graph.get_state(config)

        self.assertEqual(
            snapshot.values["answer"],
            expected,
        )
        self.assertEqual(snapshot.next, ())


if __name__ == "__main__":
    unittest.main()
