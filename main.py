from dotenv import load_dotenv

from agent.runner import AgentRunner
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
    instructions=(
        "You are a helpful general assistant. "
        "Answer clearly and concisely."
    ),
    provider=general_provider,
    registry=general_registry,
)


# -------------------------
# Manual tests
# -------------------------

print("=== Math agent ===")

answer = math_agent.run(
    "Combien font 125 multiplié par 48 ?"
)

print()
print("Assistant >", answer)


print()
print("=== General agent ===")

answer = general_agent.run(
    "Pourquoi le ciel paraît-il bleu ?"
)

print()
print("Assistant >", answer)