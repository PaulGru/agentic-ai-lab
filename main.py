from dotenv import load_dotenv

from agent.runner import AgentRunner
from orchestration.router import Router
from providers.openai_provider import OpenAIProvider
from tools.calculator import CalculatorTool
from tools.registry import ToolRegistry


load_dotenv()


# -------------------------
# Math agent
# -------------------------

math_provider = OpenAIProvider(
    model="gpt-6-luna",
)

math_registry = ToolRegistry()
math_registry.register(CalculatorTool())

math_agent = AgentRunner(
    name="math",
    description=(
        "Handles arithmetic and mathematical calculations."
    ),
    instructions=(
        "You are a math specialist. "
        "Use the calculate tool whenever an exact arithmetic "
        "calculation is required."
    ),
    provider=math_provider,
    registry=math_registry,
)


# -------------------------
# General agent
# -------------------------

general_provider = OpenAIProvider(
    model="gpt-6-luna",
)

general_registry = ToolRegistry()

general_agent = AgentRunner(
    name="general",
    description=(
        "Handles general knowledge questions and explanations "
        "that do not require mathematical calculations."
    ),
    instructions=(
        "You are a helpful general assistant. "
        "Answer clearly and concisely."
    ),
    provider=general_provider,
    registry=general_registry,
)


# -------------------------
# Router
# -------------------------

router_provider = OpenAIProvider(
    model="gpt-6-luna",
)

router = Router(
    provider=router_provider,
    agents=[
        math_agent,
        general_agent,
    ],
)


# -------------------------
# Test
# -------------------------

answer = router.run(
    "Pourquoi le ciel paraît-il bleu ?"
)

print()
print("Assistant >", answer)