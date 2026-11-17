"""V8.4 — Validate V8.3 lineage and turn a cell into an explicit executable canary.

This is deliberately NOT a semantic validator for an arbitrary scientific design.
Only an engineering canary on synthetic arithmetic tasks is supported here.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from research.experimental_design import ExperimentalDesign
from research.literature import digest_json


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str = Field(pattern=r"^(openai|anthropic)$")
    model: str = Field(min_length=2, max_length=140)

    @model_validator(mode="after")
    def no_placeholders(self):
        if any(x in self.model.upper() for x in ("REPLACE", "TODO", "__SET__", "MODELE_REEL_EXISTANT", "YOUR_MODEL")):
            raise ValueError("Replace placeholder model names before execution")
        return self


class PriceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    usd_per_million_input: float = Field(gt=0, le=1000, allow_inf_nan=False)
    usd_per_million_output: float = Field(gt=0, le=1000, allow_inf_nan=False)


class ExecutionBindings(BaseModel):
    """Explicit operator mapping; values from V8.3 remain untouched."""
    model_config = ConfigDict(extra="forbid")
    schema_version: str = Field(pattern=r"^8\.4$")
    fixed_model: ModelSpec | None = None
    factor_bindings: dict[str, dict[str, Any]]
    pricing: PriceSpec


def verify_approved_design(design_document: dict[str, Any], matrix_document: dict[str, Any]):
    """SHA-256 is integrity checking only, never evidence of human identity."""
    if design_document.get("schema_version") != "8.3" or design_document.get("status") != "approved":
        raise ValueError("An approved V8.3 experimental_design.json is required")
    if design_document.get("execution_status") != "not_executed":
        raise ValueError("Unexpected V8.3 execution status")
    raw = design_document.get("design")
    design = ExperimentalDesign.model_validate(raw)
    if raw != design.model_dump() or digest_json(raw) != design_document.get("design_sha256"):
        raise ValueError("V8.3 design is not canonical or its digest has changed")
    for key in ("mission_sha256", "literature_sha256"):
        value = design_document.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(x not in "0123456789abcdef" for x in value):
            raise ValueError(f"Missing V8.3 lineage digest: {key}")
    if design_document.get("planned_cells") != design.design_cells:
        raise ValueError("Incorrect number of declared cells")
    if design_document.get("planned_evaluations") != design.total_evaluations:
        raise ValueError("Incorrect number of declared evaluations")
    if design_document.get("planned_worst_case_llm_calls") != design.worst_case_llm_calls:
        raise ValueError("Incorrect number of declared worst-case calls")
    if matrix_document.get("schema_version") != "8.3" or matrix_document.get("design_sha256") != design_document["design_sha256"]:
        raise ValueError("V8.3 design_matrix.json does not belong to this design")
    if matrix_document.get("execution_status") != "not_executed":
        raise ValueError("Unexpected matrix execution status")
    if matrix_document.get("cells") != design.factor_matrix():
        raise ValueError("Matrix does not match the approved factor levels")
    limits = design.resource_limits
    if matrix_document.get("tasks_per_cell") != limits.tasks_per_cell or matrix_document.get("seeds") != limits.seeds:
        raise ValueError("Matrix tasks/seeds do not match the design")
    if design.task_suite.source_type != "synthetic_local":
        raise ValueError("V8.4 canary cannot run unverified external datasets")
    return design


KIND_ALLOWED = {
    "architecture": {"single_agent", "planner_executor"},
    "orchestration": {"one_shot", "retry_once"},
    "perturbation": {"clean", "tool_error"},
}


def binding_template(design: ExperimentalDesign) -> dict[str, Any]:
    """Template intentionally has placeholders, so no model or treatment is inferred."""
    mappings = {}
    for factor in design.factors:
        if factor.kind == "model_scale":
            mappings[factor.name] = {level: {"provider": "openai", "model": "REPLACE_MODEL"} for level in factor.levels}
        else:
            mappings[factor.name] = {level: "__SET__" for level in factor.levels}
    return {
        "schema_version": "8.4",
        "fixed_model": None if any(f.kind == "model_scale" for f in design.factors) else {"provider": "openai", "model": "REPLACE_MODEL"},
        "factor_bindings": mappings,
        "pricing": {"usd_per_million_input": 0.0, "usd_per_million_output": 0.0},
    }


def resolve_cell(design: ExperimentalDesign, bindings: ExecutionBindings, cell_id: str):
    cells = {c["cell_id"]: c for c in design.factor_matrix()}
    if cell_id not in cells:
        raise ValueError(f"Unknown V8.3 cell: {cell_id}")
    allowed_factors = {f.name for f in design.factors}
    if set(bindings.factor_bindings) != allowed_factors:
        raise ValueError("Bindings must specify exactly every factor")
    resolved: dict[str, Any] = {
        "model": bindings.fixed_model.model_dump() if bindings.fixed_model else None,
        "architecture": "single_agent",
        "orchestration": "one_shot",
        "perturbation": "clean",
    }
    for factor in design.factors:
        levels = bindings.factor_bindings[factor.name]
        if set(levels) != set(factor.levels):
            raise ValueError(f"Missing/extra level bindings for {factor.name}")
        actual = cells[cell_id]["factor_levels"][factor.name]
        value = levels[actual]
        if factor.kind == "model_scale":
            if bindings.fixed_model is not None:
                raise ValueError("fixed_model not allowed with model_scale factor")
            if any(not isinstance(v, dict) for v in levels.values()):
                raise ValueError(f"Every model level of {factor.name} needs a concrete provider and model")
            validated_models = [ModelSpec.model_validate(v).model_dump() for v in levels.values()]
            if len({(m["provider"], m["model"]) for m in validated_models}) != len(validated_models):
                raise ValueError(f"Different levels of {factor.name} must map to distinct models")
            resolved["model"] = ModelSpec.model_validate(value).model_dump()
        elif factor.kind in KIND_ALLOWED:
            if not isinstance(value, str) or value not in KIND_ALLOWED[factor.kind]:
                raise ValueError(f"Unsupported executable treatment for {factor.name}={actual}: {value!r}")
            # Validate all levels, not only selected, to prevent silent mapping errors.
            if any(not isinstance(x, str) or x not in KIND_ALLOWED[factor.kind] for x in levels.values()):
                raise ValueError(f"Unsupported treatment in {factor.name}")
            if len(set(levels.values())) != len(levels):
                raise ValueError(f"Different levels of {factor.name} must map to distinct treatments")
            resolved[factor.kind] = value
        else:
            raise ValueError(f"Factor kind {factor.kind!r} is not executable by V8.4 canary")
    if resolved["model"] is None:
        raise ValueError("A concrete model is required")
    return cells[cell_id], resolved


def prepare_canary_plan(
    design_document: dict[str, Any],
    matrix_document: dict[str, Any],
    binding_document: dict[str, Any],
    *, cell_id: str, evaluations: int,
) -> dict[str, Any]:
    design = verify_approved_design(design_document, matrix_document)
    bindings = ExecutionBindings.model_validate(binding_document)
    cell, cfg = resolve_cell(design, bindings, cell_id)
    if type(evaluations) is not int or not 1 <= evaluations <= 4:
        raise ValueError("Canary requires between 1 and 4 evaluations")
    limits = design.resource_limits
    assignments = [(task_index, seed) for task_index in range(limits.tasks_per_cell) for seed in limits.seeds]
    if evaluations > len(assignments):
        raise ValueError("More canary evaluations than registered task/seed pairs")
    n_per_eval = (2 if cfg["architecture"] == "planner_executor" else 1)
    if cfg["orchestration"] == "retry_once" and cfg["perturbation"] == "tool_error":
        n_per_eval += 1
    if n_per_eval > limits.max_llm_calls_per_evaluation:
        raise ValueError("Selected treatment exceeds V8.3 max_llm_calls_per_evaluation")
    reserved_calls = evaluations * n_per_eval
    reserved_tokens = reserved_calls * limits.max_tokens_per_llm_call
    if reserved_calls > limits.max_total_llm_calls or reserved_tokens > limits.max_total_tokens:
        raise ValueError("Selected canary exceeds V8.3 declared call/token budgets")
    price = bindings.pricing.model_dump()
    worst_case_price = max(price.values()) * reserved_tokens / 1_000_000
    if worst_case_price > limits.max_api_cost_usd:
        raise ValueError("Conservative monetary upper bound exceeds V8.3 cost budget")
    return {
        "schema_version": "8.4",
        "mode": "engineering_canary_not_scientific_results",
        "design_sha256": design_document["design_sha256"],
        "matrix_sha256": digest_json(matrix_document),
        "cell": cell,
        "treatment": cfg,
        "pricing": price,
        "task_seed_pairs": [{"task_index": t, "seed": s} for t, s in assignments[:evaluations]],
        "limits": {
            "max_llm_calls_per_evaluation": limits.max_llm_calls_per_evaluation,
            "max_tokens_per_llm_call": limits.max_tokens_per_llm_call,
            "max_total_llm_calls": limits.max_total_llm_calls,
            "max_total_tokens": limits.max_total_tokens,
            "max_api_cost_usd": limits.max_api_cost_usd,
        },
        "worst_case_calls": reserved_calls,
        "worst_case_tokens": reserved_tokens,
        "conservative_estimated_usd": round(worst_case_price, 8),
    }


def plan_sha256(plan: dict[str, Any]) -> str:
    return digest_json(plan)


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
