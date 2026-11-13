"""HTTP request schemas (Pydantic)."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from api.core import THREAD_PATTERN


class StartRequest(BaseModel):
    thread_id: str = Field(pattern=THREAD_PATTERN)


class DecisionRequest(BaseModel):
    decision: Literal["approve", "reject", "edit"]
    content: str | None = Field(default=None, max_length=10000)


class ResearchRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=64, pattern=THREAD_PATTERN)
    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be empty")
        return value.strip()
