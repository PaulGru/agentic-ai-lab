import ast
import operator
from typing import Any

from agent.models import ToolDefinition
from tools.base import Tool


class CalculatorTool(Tool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="calculate",
            description=(
                "Evaluate an arithmetic expression. "
                "Use this tool when an exact calculation is needed."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "Arithmetic expression, for example '125 * 48'."
                        ),
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: dict[str, Any]) -> str:
        expression = arguments.get("expression")

        if not isinstance(expression, str):
            raise ValueError(
                "'expression' must be a string"
            )

        if not expression.strip():
            raise ValueError(
                "'expression' cannot be empty"
            )

        result = self._evaluate(expression)

        return str(result)

    def _evaluate(self, expression: str) -> int | float:
        tree = ast.parse(expression, mode="eval")
        return self._evaluate_node(tree.body)

    def _evaluate_node(self, node):
        binary_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
        }

        unary_operators = {
            ast.UAdd: operator.pos,
            ast.USub: operator.neg,
        }

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                raise ValueError("Booleans are not allowed")

            if isinstance(node.value, (int, float)):
                return node.value

            raise ValueError("Only numbers are allowed")

        if isinstance(node, ast.BinOp):
            operator_function = binary_operators.get(type(node.op))

            if operator_function is None:
                raise ValueError("Operator not allowed")

            left = self._evaluate_node(node.left)
            right = self._evaluate_node(node.right)

            return operator_function(left, right)

        if isinstance(node, ast.UnaryOp):
            operator_function = unary_operators.get(type(node.op))

            if operator_function is None:
                raise ValueError("Operator not allowed")

            return operator_function(
                self._evaluate_node(node.operand)
            )

        raise ValueError("Invalid expression")