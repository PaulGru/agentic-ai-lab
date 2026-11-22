"""Bounded Python expressions and independently supplied evaluation cases."""
from __future__ import annotations

import ast
import hashlib
import operator
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from execution.contracts import StrictModel

NAME = r"^[a-z][a-z0-9_]{0,35}$"
MAX_VALUE = 10**9
MAX_INTERMEDIATE = 10**12
Name = Annotated[str, Field(pattern=NAME)]
Number = Annotated[int, Field(ge=-MAX_VALUE, le=MAX_VALUE)]
Features = Annotated[dict[Name, Number], Field(min_length=1, max_length=12)]
BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
           ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
INTERPRETER_SHA256 = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def interpreter_sha256():
    return INTERPRETER_SHA256


class Example(StrictModel):
    inputs: Features
    expected: Number


class EvaluationCase(Example):
    id: Name
    prompts: list[Annotated[str, Field(min_length=1, max_length=300)]] | None = Field(
        default=None, min_length=2, max_length=4)


class Witness(Example):
    levels: dict[Name, Name] = Field(min_length=1, max_length=4)


class EvaluationSet(StrictModel):
    schema_version: Literal["cpu.integer_exact.v1", "gpu.integer_prompt.v1"]
    cases: list[EvaluationCase] = Field(min_length=3, max_length=15)
    witnesses: list[Witness] = Field(min_length=2, max_length=48)

    @model_validator(mode="after")
    def consistent_inputs(self):
        keys = self.cases[0].inputs.keys()
        if any(item.inputs.keys() != keys for item in [*self.cases, *self.witnesses]):
            raise ValueError("Assessment and witness cases must use the same named inputs")
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("Evaluation case IDs must be unique")
        assessment = {tuple(sorted(case.inputs.items())) for case in self.cases}
        if len(assessment) != len(self.cases) or any(tuple(sorted(w.inputs.items())) in assessment for w in self.witnesses):
            raise ValueError("Assessment inputs must be distinct and withheld from implementation witnesses")
        if self.schema_version == "cpu.integer_exact.v1":
            if any(case.prompts is not None for case in self.cases):
                raise ValueError("Numerical CPU cases use features without prompts")
        else:
            variants = len(self.cases[0].prompts or [])
            if (not variants or any(len(case.prompts or []) != variants for case in self.cases)
                    or any(not 0 <= w.expected < variants for w in self.witnesses)):
                raise ValueError("GPU prompt cases and witnesses must share the same set of variant indices")
        return self


class RestrictedProgram:
    """Interpret a finite expression tree. No Python bytecode or user imports are loaded."""

    def __init__(self, source, *, features, levels):
        if len(source.encode("utf-8")) > 4000:
            raise ValueError("Program exceeds the expression size limit")
        try:
            module = ast.parse(source)
        except SyntaxError as error:
            raise ValueError("Expected valid Python syntax") from error
        if len(list(ast.walk(module))) > 160 or len(module.body) != 1:
            raise ValueError("Expected a single bounded predict function")
        fn = module.body[0]
        if (not isinstance(fn, ast.FunctionDef) or fn.name != "predict" or fn.decorator_list or fn.returns
                or getattr(fn, "type_params", ())):
            raise ValueError("Expected def predict(features, levels, seed)")
        args = fn.args
        if ([a.arg for a in args.args] != ["features", "levels", "seed"] or any(a.annotation for a in args.args)
                or args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg or args.defaults
                or len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return)):
            raise ValueError("predict must contain one return expression and the three unmodified inputs")
        self.expression = fn.body[0].value
        self._inspect(self.expression, {"features": set(features), "levels": set(levels)})

    @classmethod
    def _inspect(cls, node, keys):
        if isinstance(node, ast.Constant):
            if type(node.value) is int and abs(node.value) <= MAX_VALUE:
                return
            if type(node.value) is str and 1 <= len(node.value) <= 36 and node.value.replace("_", "").isalnum():
                return
        elif isinstance(node, ast.Name) and node.id == "seed":
            return
        elif (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
              and isinstance(node.slice, ast.Constant) and type(node.slice.value) is str):
            if node.slice.value in keys.get(node.value.id, ()):
                return
        elif isinstance(node, ast.IfExp):
            for child in (node.test, node.body, node.orelse):
                cls._inspect(child, keys)
            return
        elif isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
            cls._inspect(node.left, keys)
            cls._inspect(node.right, keys)
            return
        elif isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
            cls._inspect(node.operand, keys)
            return
        elif isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in COMPARE:
            cls._inspect(node.left, keys)
            cls._inspect(node.comparators[0], keys)
            return
        raise ValueError("Unsupported expression or unknown input in predict")

    def predict(self, features, levels, seed):
        def number(value):
            if type(value) is not int or abs(value) > MAX_INTERMEDIATE:
                raise ValueError("Expected a bounded integer")
            return value

        def read(node):
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name):
                return seed
            if isinstance(node, ast.Subscript):
                return (features if node.value.id == "features" else levels)[node.slice.value]
            if isinstance(node, ast.IfExp):
                condition = read(node.test)
                if type(condition) is not bool:
                    raise ValueError("Conditional expression requires a comparison")
                return read(node.body if condition else node.orelse)
            if isinstance(node, ast.BinOp):
                return number(BINOPS[type(node.op)](number(read(node.left)), number(read(node.right))))
            if isinstance(node, ast.UnaryOp):
                return number(UNARY[type(node.op)](number(read(node.operand))))
            return COMPARE[type(node.ops[0])](read(node.left), read(node.comparators[0]))

        try:
            result = number(read(self.expression))
            if abs(result) > MAX_VALUE:
                raise ValueError("Prediction exceeds the output range")
            return result
        except (ArithmeticError, TypeError) as error:
            raise ValueError("Program used invalid operands for this input") from error
