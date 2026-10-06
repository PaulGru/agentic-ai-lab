from providers.base import LLMProvider
from tools.registry import ToolRegistry


class AgentRunner:
    def __init__(
        self,
        name: str,
        instructions: str,
        provider: LLMProvider,
        registry: ToolRegistry,
        max_iterations: int = 5,
        verbose: bool = True,
    ):
        if not name.strip():
            raise ValueError("Agent name cannot be empty")
        if not instructions.strip():
            raise ValueError("Agent instructions cannot be empty")
        if max_iterations <= 0:
            raise ValueError(
                "max_iterations must be greater than 0"
            )

        self.name = name
        self.instructions = instructions
        self.provider = provider
        self.registry = registry
        self.max_iterations = max_iterations
        self.verbose = verbose

    def run(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        tools = self.registry.definitions()

        response = self.provider.start(
            prompt=prompt,
            tools=tools,
            instructions=self.instructions,
        )

        for iteration in range(1, self.max_iterations + 1):
            self._log(
                f"[{self.name}] iteration {iteration}"
            )

            if not response.tool_calls:
                if response.text:
                    return response.text

                raise RuntimeError(
                    "LLM returned neither text nor tool calls"
                )

            results = []

            for call in response.tool_calls:
                self._log(
                    f"[{self.name}] tool call: "
                    f"{call.name}({call.arguments})"
                )

                result = self.registry.execute(call)

                if result.is_error:
                    self._log(
                        f"[tool:error] "
                        f"{call.name} -> {result.output}"
                    )
                else:
                    self._log(
                        f"[tool] "
                        f"{call.name} -> {result.output}"
                    )

                results.append(result)

            response = (
                self.provider.continue_with_tool_results(
                    results=results,
                    tools=tools,
                    instructions=self.instructions,
                )
            )

        raise RuntimeError(
            "Maximum number of agent iterations reached"
        )

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)