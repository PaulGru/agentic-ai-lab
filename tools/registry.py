from agent.models import ToolCall, ToolDefinition, ToolResult
from tools.base import Tool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        name = tool.definition.name

        if name in self._tools:
            raise ValueError(
                f"Tool '{name}' is already registered"
            )

        self._tools[name] = tool

    def definitions(self) -> list[ToolDefinition]:
        return [
            tool.definition
            for tool in self._tools.values()
        ]

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)

        if tool is None:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                output=f"Unknown tool: {call.name}",
                is_error=True,
            )

        try:
            output = tool.execute(call.arguments)

            return ToolResult(
                call_id=call.id,
                name=call.name,
                output=str(output),
                is_error=False,
            )

        except Exception as error:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                output=f"{type(error).__name__}: {error}",
                is_error=True,
            )