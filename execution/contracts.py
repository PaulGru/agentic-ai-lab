"""V8.4.1 — portable, non-executable experiment job contracts.

Schema validation does NOT authorize running code. Execution backends must
apply their own isolation, allowlists, resource limits and approval policies.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ResourceRequest(StrictModel):
    cpus: int = Field(ge=1, le=32)
    memory_mb: int = Field(ge=128, le=131_072)
    gpus: int = Field(ge=0, le=8)
    walltime_minutes: int = Field(ge=1, le=240)


class WorkloadSpec(StrictModel):
    """Logical workload, not a shell command or a model-provider name.

    `name` is resolved by an allowlisted workload registry in a FUTURE backend.
    No workload is executed in V8.4.1.
    """

    name: str = Field(
        min_length=3,
        max_length=120,
        pattern=r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$",
    )
    parameters: dict[str, Any] = Field(default_factory=dict, max_length=32)

    @field_validator("parameters")
    @classmethod
    def only_small_json_payloads(cls, value: dict[str, Any]) -> dict[str, Any]:
        # Guards against opaque Python objects and unbounded checkpoint payloads.
        try:
            text = json.dumps(
                value, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("parameters must be finite, JSON-compatible data") from error
        if len(text.encode("utf-8")) > 16_384:
            raise ValueError("parameters may not exceed 16 KiB")
        return value


class ExperimentJob(StrictModel):
    schema_version: Literal["8.4.1"] = "8.4.1"
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    job_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    workload: WorkloadSpec
    seed: int = Field(ge=0, le=2**32 - 1)
    resources: ResourceRequest
    expected_artifacts: list[str] = Field(min_length=1, max_length=12)
    source_design_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @field_validator("expected_artifacts")
    @classmethod
    def check_artifacts(cls, value: list[str]) -> list[str]:
        import re
        for name in value:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\.json", name):
                raise ValueError("Artifact names must be flat JSON filenames")
        if len(value) != len(set(value)):
            raise ValueError("Artifact names must be unique")
        return value

    def fingerprint(self) -> str:
        """Stable digest for integrity and in-backend idempotency.

        Not a cryptographic authorization or HITL approval signature.
        """
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True,
            separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JobStatus(str, Enum):
    SUBMITTED = "submitted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobHandle(StrictModel):
    backend: str
    remote_id: str
    job_id: str
    spec_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    simulated: bool


class JobSnapshot(StrictModel):
    handle: JobHandle
    status: JobStatus
    message: str = ""


class JobArtifacts(StrictModel):
    handle: JobHandle
    files: dict[str, Any]
    simulated: bool
    warning: str
