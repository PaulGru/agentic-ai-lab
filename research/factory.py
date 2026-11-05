
import os

from agent.runner import AgentRunner
from orchestration.agent_tool_graph import AgentToolGraph
from providers.base import LLMProvider
from providers.openai_provider import OpenAIProvider
from providers.anthropic_provider import AnthropicProvider
from tools.registry import ToolRegistry


RESEARCH_INSTRUCTIONS = """
Tu es un assistant de recherche pédagogique.

Réponds clairement, de manière structurée et en français.
Explique les concepts techniques avec des exemples concrets.

Ne prétends jamais avoir consulté Internet ou vérifié une
source externe si tu n'as pas réellement utilisé un outil.

Si une information est incertaine ou dépend de l'actualité,
signale cette limite explicitement.
"""


def create_provider() -> LLMProvider:
    provider_name = os.getenv(
        "LLM_PROVIDER", "openai"
    ).strip().lower()

    if provider_name == "openai":
        model = os.getenv("OPENAI_MODEL", "").strip()

        if not model:
            raise RuntimeError("OPENAI_MODEL is missing")

        return OpenAIProvider(model=model)

    if provider_name == "anthropic":
        model = os.getenv("ANTHROPIC_MODEL", "").strip()

        if not model:
            raise RuntimeError("ANTHROPIC_MODEL is missing")

        return AnthropicProvider(model=model)

    raise ValueError(
        f"Unsupported LLM provider: {provider_name}"
    )


def build_research_graph(
    saver,
    provider: LLMProvider | None = None,
):
    selected_provider = (
        provider if provider is not None
        else create_provider()
    )

    registry = ToolRegistry()

    agent = AgentRunner(
        name="research",
        description="Assistant de recherche généraliste",
        instructions=RESEARCH_INSTRUCTIONS,
        provider=selected_provider,
        registry=registry,
    )

    return AgentToolGraph(
        agent=agent,
        checkpointer=saver,
    ).graph
