from agent.runner import AgentRunner
from providers.base import LLMProvider


class Router:
    def __init__(
        self,
        provider: LLMProvider,
        agents: list[AgentRunner],
        verbose: bool = True,
    ):
        if not agents:
            raise ValueError(
                "Router requires at least one agent"
            )

        self.provider = provider
        self.verbose = verbose

        self.agents = {
            agent.name: agent
            for agent in agents
        }

        if len(self.agents) != len(agents):
            raise ValueError(
                "Agent names must be unique"
            )

    def route(self, prompt: str) -> AgentRunner:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        agent_catalog = "\n".join(
            f"- {agent.name}: {agent.description}"
            for agent in self.agents.values()
        )

        routing_prompt = (
            "Available agents:\n"
            f"{agent_catalog}\n\n"
            "User request:\n"
            f"{prompt}"
        )

        response = self.provider.start(
            prompt=routing_prompt,
            instructions=(
                "You are a routing component. "
                "Choose the single best agent for the user request. "
                "Return exactly the agent name and nothing else. "
                "Do not answer the user's request yourself."
            ),
        )

        if not response.text:
            raise RuntimeError(
                "Router returned no routing decision"
            )

        agent_name = response.text.strip()

        agent = self.agents.get(agent_name)

        if agent is None:
            raise RuntimeError(
                f"Router selected unknown agent: '{agent_name}'"
            )

        self._log(
            f"[router] route -> {agent.name}"
        )

        return agent

    def run(self, prompt: str) -> str:
        agent = self.route(prompt)
        return agent.run(prompt)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)