
import unittest

from langgraph.checkpoint.memory import InMemorySaver

from agent.models import LLMResponse
from research.pipeline import build_research_pipeline
from tests.fakes import FakeProvider


class FakeProviderFactory:
    def __init__(self, responses):
        self.responses = list(responses)
        self.web_search_flags = []

    def __call__(self, web_search):
        self.web_search_flags.append(web_search)

        if not self.responses:
            raise RuntimeError("No fake response remaining")

        response = self.responses.pop(0)

        return FakeProvider([response])


class V64PipelineTests(unittest.TestCase):

    def test_multi_agent_pipeline(self):
        factory = FakeProviderFactory([
            # Planner
            LLMResponse(
                text='["LangGraph architecture", '
                     '"CrewAI architecture"]'
            ),

            # Researcher - première recherche
            LLMResponse(
                text="LangGraph utilise des graphes.",
                web_search_used=True,
                sources=[
                    {
                        "title": "LangGraph Docs",
                        "url": "https://docs.example.org/langgraph",
                    }
                ],
            ),

            # Researcher - deuxième recherche
            LLMResponse(
                text="CrewAI organise des équipes d'agents.",
                web_search_used=True,
                sources=[
                    {
                        "title": "CrewAI Docs",
                        "url": "https://docs.example.org/crewai",
                    }
                ],
            ),

            # Synthesizer
            LLMResponse(
                text="Voici une comparaison documentée."
            ),
        ])

        saver = InMemorySaver()
        graph = build_research_pipeline(
            saver=saver,
            provider_factory=factory,
        )

        config = {
            "configurable": {
                "thread_id": "test-v64"
            }
        }

        result = graph.invoke(
            {
                "question": (
                    "Compare LangGraph et CrewAI."
                )
            },
            config=config,
        )

        self.assertEqual(len(result["plan"]), 2)
        self.assertEqual(len(result["findings"]), 2)
        self.assertEqual(
            result["answer"],
            "Voici une comparaison documentée.",
        )

        # Seul le Researcher dispose du Web.
        self.assertEqual(
            factory.web_search_flags,
            [False, True, True, False],
        )

        self.assertTrue(result["web_search_used"])
        self.assertEqual(
            result["source_audit"]["source_count"], 2
        )
        self.assertFalse(
            result["source_audit"]["content_verified"]
        )

        # Vérification du checkpoint final.
        restarted = build_research_pipeline(
            saver=saver,
            provider_factory=FakeProviderFactory([]),
        )

        snapshot = restarted.get_state(config)

        self.assertEqual(snapshot.next, ())
        self.assertEqual(
            snapshot.values["answer"],
            "Voici une comparaison documentée.",
        )

    def test_invalid_plan_fallback(self):
        factory = FakeProviderFactory([
            LLMResponse(text="Plan non JSON"),

            LLMResponse(
                text="Aucune source disponible.",
                web_search_used=False,
            ),

            LLMResponse(
                text="Impossible de vérifier la réponse."
            ),
        ])

        graph = build_research_pipeline(
            saver=InMemorySaver(),
            provider_factory=factory,
        )

        result = graph.invoke(
            {"question": "Question initiale"},
            config={
                "configurable": {
                    "thread_id": "test-v64-fallback"
                }
            },
        )

        self.assertEqual(
            result["plan"],
            ["Question initiale"],
        )
        self.assertEqual(len(result["findings"]), 1)
        self.assertEqual(
            result["source_audit"]["status"],
            "not_searched",
        )


if __name__ == "__main__":
    unittest.main()
