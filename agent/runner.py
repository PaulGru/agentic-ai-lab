from agent.models import (
    AgentOutcome,
    HandoffRequest,
    LLMResponse,
    ToolDefinition,
    ToolResult,
)
from providers.base import LLMProvider
from tools.registry import ToolRegistry


HANDOFF_TOOL_NAME = "handoff_to_agent"


class AgentRunner:
    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        provider: LLMProvider,
        registry: ToolRegistry,
        handoff_targets: dict[str, str] | None = None,
        max_iterations: int = 5,
        verbose: bool = True,
    ):
        if not name.strip():
            raise ValueError("Agent name cannot be empty")

        if not description.strip():
            raise ValueError("Agent description cannot be empty")

        if not instructions.strip():
            raise ValueError("Agent instructions cannot be empty")

        if max_iterations <= 0:
            raise ValueError(
                "max_iterations must be greater than 0"
            )

        self.name = name
        self.description = description
        self.instructions = instructions
        self.provider = provider
        self.registry = registry
        self.handoff_targets = dict(handoff_targets or {})
        self.max_iterations = max_iterations
        self.verbose = verbose

    def run(self, prompt: str) -> AgentOutcome:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        tools = self._tool_definitions()

        response = self.provider.start(
            prompt=prompt,
            tools=tools,
            instructions=self.instructions,
        )

        return self._run_from_response(
            response=response,
            tools=tools,
        )

    def resume_handoff(
        self,
        request: HandoffRequest,
        result: str,
    ) -> AgentOutcome:
        if not result.strip():
            raise ValueError(
                "Handoff result cannot be empty"
            )

        tools = self._tool_definitions()

        response = (
            self.provider.continue_with_tool_results(
                results=[
                    ToolResult(
                        call_id=request.call_id,
                        name=HANDOFF_TOOL_NAME,
                        output=result,
                    )
                ],
                tools=tools,
                instructions=self.instructions,
            )
        )

        return self._run_from_response(
            response=response,
            tools=tools,
        )

    def _run_from_response(
        self,
        response: LLMResponse,
        tools: list[ToolDefinition],
    ) -> AgentOutcome:
        for iteration in range(
            1,
            self.max_iterations + 1,
        ):
            self._log(
                f"[{self.name}] iteration {iteration}"
            )

            if not response.tool_calls:
                if response.text:
                    return AgentOutcome(
                        text=response.text
                    )

                raise RuntimeError(
                    "LLM returned neither text nor tool calls"
                )

            handoff_calls = [
                call
                for call in response.tool_calls
                if call.name == HANDOFF_TOOL_NAME
            ]

            if handoff_calls:
                if len(response.tool_calls) != 1:
                    raise RuntimeError(
                        "A handoff must be the only tool call "
                        "in the response"
                    )

                call = handoff_calls[0]

                target_agent = call.arguments.get("agent")
                task = call.arguments.get("task")

                if target_agent not in self.handoff_targets:
                    raise RuntimeError(
                        f"Agent '{self.name}' requested an "
                        f"invalid handoff target: "
                        f"'{target_agent}'"
                    )

                if not isinstance(task, str):
                    raise RuntimeError(
                        "Handoff task must be a string"
                    )

                if not task.strip():
                    raise RuntimeError(
                        "Handoff task cannot be empty"
                    )

                self._log(
                    f"[{self.name}] handoff request -> "
                    f"{target_agent}"
                )

                return AgentOutcome(
                    handoff=HandoffRequest(
                        call_id=call.id,
                        target_agent=target_agent,
                        task=task,
                    )
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

    def _tool_definitions(
        self,
    ) -> list[ToolDefinition]:
        tools = self.registry.definitions()

        if not self.handoff_targets:
            return tools

        target_catalog = "\n".join(
            f"- {name}: {description}"
            for name, description
            in self.handoff_targets.items()
        )

        handoff_tool = ToolDefinition(
            name=HANDOFF_TOOL_NAME,
            description=(
                "Delegate a subtask to another agent. "
                "The delegated result will be returned to you "
                "so that you can continue your work.\n\n"
                "Available agents:\n"
                f"{target_catalog}"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "enum": list(
                            self.handoff_targets.keys()
                        ),
                        "description": (
                            "Agent that should handle "
                            "the delegated task."
                        ),
                    },
                    "task": {
                        "type": "string",
                        "description": (
                            "Self-contained task for "
                            "the target agent."
                        ),
                    },
                },
                "required": [
                    "agent",
                    "task",
                ],
                "additionalProperties": False,
            },
        )

        return [
            *tools,
            handoff_tool,
        ]

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)