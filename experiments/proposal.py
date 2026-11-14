
import ast
import json
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExperimentProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: str = Field(min_length=1, max_length=2000)
    protocol: list[str] = Field(min_length=1, max_length=10)
    seed: int = Field(ge=0, le=2**32 - 1)
    dependencies: list[str] = Field(max_length=10)
    code: str = Field(min_length=1, max_length=12000)
    expected_output: str = Field(min_length=1, max_length=2000)

    @field_validator("dependencies")
    @classmethod
    def check_dependencies(
        cls, dependencies: list[str]
    ) -> list[str]:
        for dependency in dependencies:
            if not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*==[A-Za-z0-9][A-Za-z0-9_.+!-]*",
                dependency,
            ):
                raise ValueError(
                    "Pin each dependency, e.g. numpy==2.3.0"
                )
        return dependencies

    @field_validator("code")
    @classmethod
    def check_code_syntax(cls, code: str) -> str:
        try:
            ast.parse(code)
        except SyntaxError as error:
            raise ValueError(
                "Generated code has invalid Python syntax"
            ) from error

        return code


def parse_proposal(raw_text: str) -> ExperimentProposal:
    try:
        data = json.loads(raw_text)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            "LLM did not return valid JSON"
        ) from error

    return ExperimentProposal.model_validate(data)
