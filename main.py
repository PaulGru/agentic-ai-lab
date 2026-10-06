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

agent = AgentRunner(
    provider=provider,
    registry=registry,
)

answer = agent.run(
    "Utilise calculate pour évaluer __import__('os').getcwd()"
)

print()
print("Assistant >", answer)