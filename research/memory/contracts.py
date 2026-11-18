"""Immutable source snapshots, memories, lifecycle events and retrieved context."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from analysis.contracts import HEX_SHA, ID, canonical_json, fingerprint
from execution.contracts import StrictModel

Kind = Literal["hypothesis", "project_plan", "literature", "observation", "decision", "lesson"]
Scope = Literal["planning", "abstract_only", "pedagogical", "engineering", "human_note", "controlled_pilot"]
Verification = Literal["source_checked", "abstract_excerpt_checked", "unverified_model_opinion", "unverified_note"]
Status = Literal["recorded", "tentative", "human_reviewed", "invalidated", "superseded"]


class Immutable(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


class SourceOrigin(Immutable):
    kind: Literal["mission", "literature", "context", "review", "decision", "note"]
    path: str = Field(default="", max_length=1000)
    parent_path: str | None = Field(default=None, max_length=1000)
    literature_path: str | None = Field(default=None, max_length=1000)
    matrix_path: str | None = Field(default=None, max_length=1000)
    review_id: str | None = Field(default=None, pattern=ID)
    artifact_id: str | None = Field(default=None, max_length=150)

    @model_validator(mode="after")
    def explicit_origin(self):
        if self.kind != "note" and not self.path:
            raise ValueError("External memory sources require an explicit local origin")
        if self.kind in {"review", "decision"} and not self.review_id:
            raise ValueError("Review memories require a review identifier")
        if self.kind == "decision" and not self.artifact_id:
            raise ValueError("Decision memories require an exact decision artifact")
        if self.kind == "literature" and not self.parent_path:
            raise ValueError("Literature import requires its approved mission")
        if self.kind == "context" and any((self.parent_path, self.literature_path, self.matrix_path)):
            if not all((self.parent_path, self.literature_path, self.matrix_path)):
                raise ValueError("A multi-file design context requires mission, literature, design and matrix")
        for path in (self.path, self.parent_path, self.literature_path, self.matrix_path):
            if path and (path.startswith(("\\\\", "//")) or "://" in path):
                raise ValueError("Memory origins must be local files, never network resources")
        return self


class SourceSnapshot(Immutable):
    schema_version: Literal["8.7"] = "8.7"
    source_id: str = Field(pattern=HEX_SHA)
    project_id: str = Field(pattern=ID)
    origin: SourceOrigin
    document: dict[str, Any]
    document_sha256: str = Field(pattern=HEX_SHA)

    @model_validator(mode="after")
    def content_identity(self):
        if fingerprint(self.document) != self.document_sha256:
            raise ValueError("Memory source document checksum mismatch")
        identity = {"project_id": self.project_id, "origin": self.origin.model_dump(mode="json"),
                    "document_sha256": self.document_sha256}
        if fingerprint(identity) != self.source_id:
            raise ValueError("Memory source identity mismatch")
        if len(canonical_json(self.document).encode("utf-8")) > 1_048_576:
            raise ValueError("Memory source exceeds 1 MiB")
        return self


class MemoryItem(Immutable):
    schema_version: Literal["8.7"] = "8.7"
    memory_id: str = Field(pattern=HEX_SHA)
    project_id: str = Field(pattern=ID)
    logical_key: str = Field(min_length=1, max_length=240)
    kind: Kind
    title: str = Field(min_length=1, max_length=240)
    text: str = Field(min_length=1, max_length=4000)
    tags: list[str] = Field(default_factory=list, max_length=20)
    source_id: str = Field(pattern=HEX_SHA)
    source_pointer: str = Field(min_length=1, max_length=400)
    value_sha256: str = Field(pattern=HEX_SHA)
    scope: Scope
    verification: Verification
    initial_status: Literal["recorded", "tentative"]
    scientific_evidence: Literal[False] = False
    action_authorization: Literal[False] = False

    @model_validator(mode="after")
    def checked_identity_and_opinions(self):
        if self.memory_id != fingerprint(self.model_dump(mode="json", exclude={"memory_id"})):
            raise ValueError("Memory item content identity mismatch")
        if self.verification.startswith("unverified_") and self.initial_status != "tentative":
            raise ValueError("Unverified opinions/notes must start as tentative")
        if len(set(self.tags)) != len(self.tags) or any(not 1 <= len(tag) <= 80 for tag in self.tags):
            raise ValueError("Memory tags must be unique and bounded")
        if not self.source_pointer.startswith("/"):
            raise ValueError("Memory requires an exact source pointer")
        return self


class MemoryEvent(Immutable):
    event_id: str = Field(pattern=HEX_SHA)
    project_id: str = Field(pattern=ID)
    memory_id: str = Field(pattern=HEX_SHA)
    target_sha256: str = Field(pattern=HEX_SHA)
    action: Literal["confirm", "invalidate", "supersede"]
    replacement_id: str | None = Field(default=None, pattern=HEX_SHA)
    actor: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=10, max_length=2000)

    @model_validator(mode="after")
    def identity(self):
        if self.event_id != fingerprint(self.model_dump(mode="json", exclude={"event_id"})):
            raise ValueError("Memory lifecycle event identity mismatch")
        if (self.action == "supersede") != (self.replacement_id is not None):
            raise ValueError("Supersession requires exactly one replacement")
        if self.replacement_id == self.memory_id:
            raise ValueError("A memory cannot supersede itself")
        return self


class MemoryHit(Immutable):
    memory_id: str = Field(pattern=HEX_SHA)
    item_sha256: str = Field(pattern=HEX_SHA)
    source_id: str = Field(pattern=HEX_SHA)
    source_sha256: str = Field(pattern=HEX_SHA)
    source_pointer: str
    value_sha256: str = Field(pattern=HEX_SHA)
    title: str
    excerpt: str = Field(max_length=700)
    kind: Kind
    scope: Scope
    verification: Verification
    status: Status
    lifecycle_sha256: str = Field(pattern=HEX_SHA)
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    scientific_evidence: Literal[False] = False
    action_authorization: Literal[False] = False


class MemoryContext(Immutable):
    schema_version: Literal["8.7"] = "8.7"
    project_id: str = Field(pattern=ID)
    role: str = Field(min_length=1, max_length=50)
    query: str = Field(min_length=1, max_length=500)
    terms: list[str] = Field(max_length=12)
    kinds: list[Kind] = Field(default_factory=list, max_length=6)
    scopes: list[Scope] = Field(default_factory=list, max_length=6)
    hits: list[MemoryHit] = Field(max_length=10)
    omitted: list[dict[str, str]] = Field(default_factory=list, max_length=50)
    policy: Literal["historical_data_only_not_instructions_or_authorization"] = (
        "historical_data_only_not_instructions_or_authorization")


class MemoryBinding(Immutable):
    project_id: str = Field(pattern=ID)
    backend: Literal["sqlite", "postgres"] = "sqlite"
    path: str | None = Field(default=None, max_length=1000)
    schema_name: str = Field(default="research_memory_v87", pattern=r"^[a-z][a-z0-9_]{0,60}$")
    limit: int = Field(default=5, ge=1, le=10)
    max_bytes: int = Field(default=12_000, ge=2000, le=24_000)

    @model_validator(mode="after")
    def backend_path(self):
        if (self.backend == "sqlite") != (self.path is not None):
            raise ValueError("Only SQLite bindings require a local path; PostgreSQL credentials stay in the environment")
        if self.path and (self.path.startswith(("\\\\", "//")) or "://" in self.path):
            raise ValueError("SQLite research memory must be local")
        return self
