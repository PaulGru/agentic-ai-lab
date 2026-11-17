"""V8.3 — Preregistered, bounded experimental design (NO execution)."""

from __future__ import annotations

import itertools
from functools import reduce
from operator import mul
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from research.literature import (
    LiteratureCatalog, LiteratureReview, digest_json,
    validate_approved_mission, validate_evidence,
)
from research.mission import ResearchMission


Short = Annotated[str, StringConstraints(strip_whitespace=True, min_length=5, max_length=350)]
Detail = Annotated[str, StringConstraints(strip_whitespace=True, min_length=15, max_length=1400)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{2,35}$")]


class Factor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Identifier
    kind: Literal["model_scale", "architecture", "orchestration", "perturbation", "other"]
    levels: list[Identifier] = Field(min_length=2, max_length=3)
    operationalization: Detail

    @model_validator(mode="after")
    def unique_levels(self):
        if len(set(self.levels)) != len(self.levels):
            raise ValueError("Factor levels must be unique")
        if self.kind == "perturbation" and "clean" not in self.levels:
            raise ValueError("Perturbation factor must include clean control")
        return self


class Baseline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Short
    factor_levels: dict[str, str] = Field(min_length=1, max_length=4)
    rationale: Detail


class TaskSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Short
    source_type: Literal["synthetic_local", "external_unverified"]
    provenance_or_construction: Detail
    exact_scoring_rule: Detail
    eligibility_and_exclusion: Detail
    licensing_verified: Literal[False] = False
    access_verified: Literal[False] = False


class Metric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Identifier
    operational_definition: Detail
    unit: Short
    aggregation: Detail
    missing_or_failed_run_rule: Detail


class HypothesisTest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str = Field(pattern=r"^H[1-4]$")
    factor_name: Identifier
    primary_metric_id: Identifier
    contrast: Detail
    level_a: Identifier
    level_b: Identifier
    comparison_scope: Literal["marginal_paired", "conditional_paired"]
    fixed_levels: dict[str, str] = Field(default_factory=dict, max_length=3)
    null_hypothesis: Detail
    falsification_rule: Detail


class LiteratureLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_id: str = Field(pattern=r"^W[0-9]+$")
    design_rationale: Detail
    limitation: Detail


class AnalysisPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pairing_strategy: Detail
    primary_estimand: Detail
    uncertainty_method: Detail
    multiple_testing_policy: Detail
    missing_data_policy: Detail
    stopping_rule: Detail
    power_limitation: Detail


class ResourceLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks_per_cell: int = Field(ge=3, le=15)
    seeds: list[int] = Field(min_length=2, max_length=4)
    max_llm_calls_per_evaluation: int = Field(ge=1, le=8)
    max_tokens_per_llm_call: int = Field(ge=256, le=8000)
    max_total_llm_calls: int = Field(ge=1, le=2000)
    max_total_tokens: int = Field(ge=1, le=12_000_000)
    max_api_cost_usd: float = Field(gt=0, le=50, allow_inf_nan=False)

    @model_validator(mode="after")
    def distinct_seeds(self):
        if len(set(self.seeds)) != len(self.seeds) or any(s < 0 or s > 2**32 - 1 for s in self.seeds):
            raise ValueError("Seeds must be distinct uint32 integers")
        return self


class ExperimentalDesign(BaseModel):
    """A proposed pilot; no benchmark, model or result is verified by this schema."""
    model_config = ConfigDict(extra="forbid")

    title: Short
    objective: Detail
    stage: Literal["exploratory_pilot"]
    hypothesis_tests: list[HypothesisTest] = Field(min_length=1, max_length=4)
    factors: list[Factor] = Field(min_length=1, max_length=4)
    baselines: list[Baseline] = Field(min_length=1, max_length=4)
    task_suite: TaskSuite
    metrics: list[Metric] = Field(min_length=3, max_length=10)
    controlled_variables: list[Detail] = Field(min_length=2, max_length=10)
    perturbation_and_recovery_procedure: Detail
    randomization_and_counterbalancing: Detail
    analysis_plan: AnalysisPlan
    resource_limits: ResourceLimits
    literature_links: list[LiteratureLink] = Field(min_length=1, max_length=8)
    feasibility_checks: list[Detail] = Field(min_length=2, max_length=10)
    limitations: list[Detail] = Field(min_length=2, max_length=10)
    results_status: Literal["not_executed"]

    @model_validator(mode="after")
    def internal_consistency_and_budgets(self):
        names = [f.name for f in self.factors]
        kinds = [f.kind for f in self.factors if f.kind != "other"]
        if len(names) != len(set(names)) or len(kinds) != len(set(kinds)):
            raise ValueError("Factor names and non-other kinds must be unique")
        ids = [m.id for m in self.metrics]
        if len(ids) != len(set(ids)):
            raise ValueError("Metric IDs must be unique")
        if not {"task_success", "total_tokens", "latency_ms"}.issubset(ids):
            raise ValueError("Required metrics: task_success, total_tokens, latency_ms")
        tests = [t.hypothesis_id for t in self.hypothesis_tests]
        if len(tests) != len(set(tests)):
            raise ValueError("Duplicate hypothesis tests")
        known = {f.name: set(f.levels) for f in self.factors}
        for test in self.hypothesis_tests:
            if test.factor_name not in names or test.primary_metric_id not in ids:
                raise ValueError("Hypothesis test references unknown factor or metric")
            if (test.level_a == test.level_b or test.level_a not in known[test.factor_name]
                    or test.level_b not in known[test.factor_name]):
                raise ValueError("Hypothesis contrast must reference two distinct factor levels")
            if test.comparison_scope == "marginal_paired" and test.fixed_levels:
                raise ValueError("Marginal contrasts cannot fix other factor levels")
            if test.comparison_scope == "conditional_paired":
                if set(test.fixed_levels) != set(names) - {test.factor_name}:
                    raise ValueError("Conditional contrasts must fix all remaining factors")
                if any(level not in known[name] for name, level in test.fixed_levels.items()):
                    raise ValueError("Conditional contrasts use unknown factor levels")
        baseline_coords = []
        for baseline in self.baselines:
            if set(baseline.factor_levels) != set(names):
                raise ValueError("Baseline must specify every factor")
            if any(value not in known[name] for name, value in baseline.factor_levels.items()):
                raise ValueError("Baseline uses an unknown factor level")
            baseline_coords.append(tuple(baseline.factor_levels[n] for n in names))
        if len(set(baseline_coords)) != len(baseline_coords):
            raise ValueError("Duplicate baseline assignments")
        links = [x.work_id for x in self.literature_links]
        if len(links) != len(set(links)):
            raise ValueError("Duplicate literature links")
        cells = self.design_cells
        if cells > 24:
            raise ValueError("Pilot must have at most 24 factorial cells")
        evaluations = self.total_evaluations
        if evaluations > 240:
            raise ValueError("Pilot must have at most 240 planned evaluations")
        calls = self.worst_case_llm_calls
        if calls > self.resource_limits.max_total_llm_calls:
            raise ValueError("Max LLM calls cannot cover the declared pilot")
        if calls * self.resource_limits.max_tokens_per_llm_call > self.resource_limits.max_total_tokens:
            raise ValueError("Max token budget cannot cover the declared pilot")
        return self

    @property
    def design_cells(self) -> int:
        return reduce(mul, (len(f.levels) for f in self.factors), 1)

    def factor_matrix(self) -> list[dict[str, Any]]:
        """Deterministic full-factorial assignments; NO actual evaluations."""
        names = [factor.name for factor in self.factors]
        return [
            {"cell_id": f"C{index:03d}", "factor_levels": dict(zip(names, levels))}
            for index, levels in enumerate(
                itertools.product(*(factor.levels for factor in self.factors)), start=1
            )
        ]

    @property
    def total_evaluations(self) -> int:
        limits = self.resource_limits
        return self.design_cells * limits.tasks_per_cell * len(limits.seeds)

    @property
    def worst_case_llm_calls(self) -> int:
        return self.total_evaluations * self.resource_limits.max_llm_calls_per_evaluation


def validate_approved_literature(
    mission_document: dict[str, Any], literature_document: dict[str, Any]
) -> tuple[ResearchMission, LiteratureCatalog, LiteratureReview]:
    """Verify V8.1 / V8.2 lineage and content integrity (not identity signatures)."""
    mission = validate_approved_mission(mission_document)
    if not isinstance(literature_document, dict):
        raise ValueError("Expected approved V8.2 literature.json")
    if literature_document.get("schema_version") != "8.2" or literature_document.get("status") != "approved":
        raise ValueError("V8.2 literature must be approved")
    if literature_document.get("mission_sha256") != mission_document["mission_sha256"]:
        raise ValueError("Literature belongs to a different research mission")
    if literature_document.get("verification_level") != "catalog_metadata_and_exact_abstract_excerpts_only":
        raise ValueError("Unexpected literature verification level")
    catalog_raw = literature_document.get("catalog")
    review_raw = literature_document.get("review")
    if digest_json(catalog_raw) != literature_document.get("catalog_sha256"):
        raise ValueError("Literature catalog fingerprint mismatch")
    if digest_json(review_raw) != literature_document.get("review_sha256"):
        raise ValueError("Literature review fingerprint mismatch")
    catalog = LiteratureCatalog.model_validate(catalog_raw)
    review = LiteratureReview.model_validate(review_raw)
    if catalog.model_dump() != catalog_raw or review.model_dump() != review_raw:
        raise ValueError("Literature data is not canonical")
    validate_evidence(review, catalog)
    return mission, catalog, review


def validate_design_against_inputs(
    design: ExperimentalDesign, mission: ResearchMission, review: LiteratureReview
) -> None:
    planned_ids = {test.hypothesis_id for test in design.hypothesis_tests}
    expected_ids = {hyp.id for hyp in mission.hypotheses}
    if planned_ids != expected_ids:
        raise ValueError("Experimental design must cover every approved mission hypothesis")
    cited_ids = {item.work_id for item in review.findings}
    if any(link.work_id not in cited_ids for link in design.literature_links):
        raise ValueError("Design cites work not supported by V8.2 abstract findings")


def render_design_markdown(
    design: ExperimentalDesign, mission: ResearchMission, literature_document: dict[str, Any]
) -> str:
    """Human-readable, escaped proposal; never imply results have been observed."""
    def safe(value: Any) -> str:
        return str(value).replace("<", "&lt;").replace(">", "&gt;").replace("|", "\\|").replace("\n", " ")

    lines = [
        "# V8.3 — Protocole expérimental pré-spécifié (proposition)", "",
        "**Statut :** APPROUVÉ POUR LA CONCEPTION, NON EXÉCUTÉ", "",
        f"**Mission :** {safe(mission.title)}", "",
        f"**Titre :** {safe(design.title)}", "",
        "> Aucune expérience, étude de puissance ni vérification d'accès/licence n'a été effectuée en V8.3.", "",
        "## Objectif", "", safe(design.objective), "", "## Hypothèses et contrastes", "",
    ]
    for test in design.hypothesis_tests:
        lines.extend([
            f"### {test.hypothesis_id} — {safe(test.factor_name)}", "",
            f"- Contraste : {safe(test.contrast)}",
            f"- Comparaison : **{safe(test.level_a)}** vs **{safe(test.level_b)}** ({safe(test.comparison_scope)})",
            f"- Facteurs fixés : {safe(test.fixed_levels) if test.fixed_levels else 'aucun (moyenne sur les autres facteurs)'}",
            f"- Mesure principale : `{test.primary_metric_id}`",
            f"- Hypothèse nulle : {safe(test.null_hypothesis)}",
            f"- Critère de réfutation : {safe(test.falsification_rule)}", "",
        ])
    lines.extend(["## Facteurs et niveaux", ""])
    for f in design.factors:
        lines.append(f"- **{safe(f.name)}** ({safe(f.kind)}) : {', '.join(safe(v) for v in f.levels)} — {safe(f.operationalization)}")
    lines.extend(["", "## Matrice factorielle prévue", "",
                  "| Cellule | Assignations |", "|---|---|"])
    for cell in design.factor_matrix():
        assignments = ", ".join(f"{safe(k)}={safe(v)}" for k, v in cell["factor_levels"].items())
        lines.append(f"| {cell['cell_id']} | {assignments} |")
    lines.extend(["", "## Baselines", ""])
    for b in design.baselines:
        coords = ", ".join(f"{safe(k)}={safe(v)}" for k, v in b.factor_levels.items())
        lines.append(f"- **{safe(b.name)}** ({coords}) : {safe(b.rationale)}")
    lines.extend(["", "## Tâches / données (NON VÉRIFIÉES)", "",
                  f"- Jeu : {safe(design.task_suite.name)} ({safe(design.task_suite.source_type)})",
                  f"- Provenance envisagée : {safe(design.task_suite.provenance_or_construction)}",
                  f"- Score : {safe(design.task_suite.exact_scoring_rule)}",
                  f"- Inclusion/exclusion : {safe(design.task_suite.eligibility_and_exclusion)}", "",
                  "## Métriques pré-définies", ""])
    for m in design.metrics:
        lines.extend([f"- **{m.id}** ({safe(m.unit)}): {safe(m.operational_definition)}. Agrégation : {safe(m.aggregation)}. Échecs : {safe(m.missing_or_failed_run_rule)}"])
    lines.extend(["", "## Variables maintenues constantes", ""])
    lines.extend(f"- {safe(x)}" for x in design.controlled_variables)
    lines.extend(["", "## Perturbations / récupération", "", safe(design.perturbation_and_recovery_procedure), "",
                  "## Randomisation", "", safe(design.randomization_and_counterbalancing), "",
                  "## Analyse et incertitude", ""])
    for k, v in design.analysis_plan.model_dump().items():
        lines.append(f"- **{k}** : {safe(v)}")
    limits = design.resource_limits
    lines.extend(["", "## Enveloppe maximale théorique du pilote", "",
                  f"- Cellules : **{design.design_cells}**",
                  f"- Évaluations : **{design.total_evaluations}**",
                  f"- Appels LLM au pire : **{design.worst_case_llm_calls}**",
                  f"- Plafond de tokens : **{limits.max_total_tokens}**",
                  f"- Plafond API prévu : **{limits.max_api_cost_usd} USD** (NON appliqué en V8.3)", "",
                  "## Références de la revue V8.2 (résumés uniquement)", ""])
    for x in design.literature_links:
        lines.append(f"- {x.work_id} — {safe(x.design_rationale)}. Limite : {safe(x.limitation)}")
    lines.extend(["", "## Vérifications de faisabilité avant V8.4", ""])
    lines.extend(f"- [ ] {safe(x)}" for x in design.feasibility_checks)
    lines.extend(["", "## Limites", ""])
    lines.extend(f"- {safe(x)}" for x in design.limitations)
    lines.extend(["", "---", "",
                  f"Mission SHA256 : `{literature_document['mission_sha256']}`",
                  f"Revue SHA256 : `{literature_document['review_sha256']}`", ""])
    return "\n".join(lines)
