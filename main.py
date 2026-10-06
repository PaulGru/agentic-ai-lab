from dotenv import load_dotenv

from agent.runner import AgentRunner
from providers.openai_provider import OpenAIProvider
from tools.calculator import CalculatorTool
from tools.registry import ToolRegistry


load_dotenv()


provider = OpenAIProvider(
    model="gpt-6-luna",
)

registry = ToolRegistry()
registry.register(CalculatorTool())


math_agent = AgentRunner(
    name="math",
    instructions=(
        "You are a math specialist. "
        "Use the calculate tool whenever an exact arithmetic "
        "calculation is required."
    ),
    provider=provider,
    registry=registry,
)


answer = math_agent.run(
    "Combien font 125 multiplié par 48 ?"
)


print()
print("Assistant >", answer)