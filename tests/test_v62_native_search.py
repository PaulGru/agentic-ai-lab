
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langgraph.checkpoint.memory import InMemorySaver

from agent.models import LLMResponse
from providers.openai_provider import OpenAIProvider
from providers.anthropic_provider import AnthropicProvider
from research.factory import build_research_graph
from tests.stateful_provider import StatefulFakeProvider


URL = "https://example.org/documentation"


class V62NativeSearchTests(unittest.TestCase):

    def test_openai_native_search(self):
        client = Mock()

        citation = SimpleNamespace(
            type="url_citation",
            title="Documentation",
            url=URL,
        )

        response = SimpleNamespace(
            id="resp-test",
            output_text="Voici la documentation.",
            output=[
                SimpleNamespace(type="web_search_call"),
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            annotations=[citation]
                        )
                    ],
                ),
            ],
        )

        client.responses.create.return_value = response

        with patch(
            "providers.openai_provider.OpenAI",
            return_value=client,
        ):
            provider = OpenAIProvider(
                model="test-model",
                native_web_search=True,
            )

            result = provider.start("Recherche LangGraph")

        request = client.responses.create.call_args.kwargs

        self.assertIn(
            {"type": "web_search", "search_context_size": "low"},
            request["tools"],
        )
        self.assertTrue(result.web_search_used)
        self.assertEqual(result.sources[0]["url"], URL)

    def test_anthropic_native_search(self):
        client = Mock()

        search_block = Mock()
        search_block.type = "server_tool_use"
        search_block.name = "web_search"
        search_block.model_dump.return_value = {
            "type": "server_tool_use"
        }

        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Voici la documentation."
        text_block.citations = [
            SimpleNamespace(
                type="web_search_result_location",
                title="Documentation",
                url=URL,
            )
        ]
        text_block.model_dump.return_value = {
            "type": "text",
            "text": text_block.text,
        }

        client.messages.create.return_value = SimpleNamespace(
            content=[search_block, text_block]
        )

        with patch(
            "providers.anthropic_provider.Anthropic",
            return_value=client,
        ):
            provider = AnthropicProvider(
                model="test-model",
                native_web_search=True,
            )

            result = provider.start("Recherche LangGraph")

        request = client.messages.create.call_args.kwargs

        self.assertIn(
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 2,
            },
            request["tools"],
        )
        self.assertTrue(result.web_search_used)
        self.assertEqual(result.sources[0]["url"], URL)

    def test_sources_persist_in_graph(self):
        saver = InMemorySaver()

        fake = StatefulFakeProvider([
            LLMResponse(
                text="Une réponse documentée.",
                web_search_used=True,
                sources=[
                    {"title": "Documentation", "url": URL}
                ],
            ),
        ])

        config = {
            "configurable": {
                "thread_id": "test-v62-sources"
            }
        }

        graph = build_research_graph(
            saver=saver,
            provider=fake,
        )

        graph.invoke(
            {"prompt": "Recherche LangGraph"},
            config=config,
        )

        restarted = build_research_graph(
            saver=saver,
            provider=StatefulFakeProvider([]),
        )

        state = restarted.get_state(config)

        self.assertTrue(state.values["web_search_used"])
        self.assertEqual(state.values["sources"][0]["url"], URL)
        self.assertEqual(state.next, ())


if __name__ == "__main__":
    unittest.main()
