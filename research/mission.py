"""V8.1 — Contrats de la mission et validation du research brief."""

import hashlib
import json
import re
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)


ShortText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=5, max_length=500),
]
LongText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=15, max_length=1500),
]

REQUIRED_SECTIONS = (
    "Question",
    "Contexte",
    "Périmètre",
    "Contraintes",
    "Critères de réussite",
)


def parse_brief(text: str) -> dict[str, str]:
    """Extrait un brief Markdown borné et vérifie ses sections obligatoires."""
    if not isinstance(text, str) or not 100 <= len(text) <= 12_000:
        raise ValueError("Research brief must contain 100 to 12000 characters")

    sections: dict[str, list[str]] = {}
    current: str | None = None

    for line in text.splitlines():
        match = re.fullmatch(r"##\s+(.+?)\s*", line)
        if match:
            current = match.group(1).strip()
            if current in sections:
                raise ValueError(f"Duplicate brief section: {current}")
            sections[current] = []
        elif current is not None:
            sections[current].append(line)

    cleaned = {name: "\n".join(lines).strip() for name, lines in sections.items()}
    for section in REQUIRED_SECTIONS:
        if len(cleaned.get(section, "")) < 10:
            raise ValueError(f"Missing or empty brief section: {section}")
    if len(cleaned["Question"]) < 20:
        raise ValueError("Research question is too short")
    return cleaned


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^H[1-4]$")
    claim: LongText
    falsification_test: LongText


class ResearchMission(BaseModel):
    """Plan proposé ; sa validité scientifique reste à examiner humainement."""
    model_config = ConfigDict(extra="forbid")

    title: ShortText
    research_question: LongText
    objective: LongText
    hypotheses: list[Hypothesis] = Field(min_length=1, max_length=4)
    independent_variables: list[ShortText] = Field(min_length=1, max_length=8)
    dependent_metrics: list[ShortText] = Field(min_length=1, max_length=8)
    controls: list[ShortText] = Field(min_length=1, max_length=8)
    baselines: list[ShortText] = Field(min_length=1, max_length=8)
    literature_queries: list[ShortText] = Field(min_length=2, max_length=8)
    experiment_outline: list[ShortText] = Field(min_length=2, max_length=10)
    validity_risks: list[ShortText] = Field(min_length=1, max_length=8)
    success_criteria: list[ShortText] = Field(min_length=1, max_length=8)
    deliverables: list[ShortText] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def unique_hypothesis_ids(self):
        ids = [hypothesis.id for hypothesis in self.hypotheses]
        if len(set(ids)) != len(ids):
            raise ValueError("Hypothesis IDs must be unique")
        return self


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def mission_digest(mission: dict) -> str:
    canonical = json.dumps(
        mission, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256_text(canonical)
