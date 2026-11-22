"""Compile the approved checkpoint, policy and perturbation grid into GPU jobs."""
from random import Random

from analysis.contracts import fingerprint
from execution.agentic_gpu import (
    ARTIFACT, CAPABILITY, MAX_INPUT, MAX_OUTPUT, MINUTES, MODELS, POLICIES, PROFILES, SEEDS, TASKS,
    WORKLOAD, AgenticGPUConfig, asset_manifest, execution_scope, suite,
)
from execution.contracts import ExperimentJob, ResourceRequest, WorkloadSpec
from research.experimental_design import ExperimentalDesign, validate_approved_literature, validate_design_against_inputs
from research.resource_catalog import AGENTIC_ID, ResourceChoice

STUDY_KIND = "compiled_agentic_pilot"
FACTOR_LEVELS = {"checkpoint": ("other", tuple(MODELS)), "policy": ("orchestration", POLICIES),
                 "shift": ("perturbation", PROFILES)}
EVALUATIONS = len(MODELS) * len(POLICIES) * len(PROFILES) * len(SEEDS) * len(TASKS)


def design_constraints():
    return {
        "capability": CAPABILITY, "task_count": len(TASKS), "seeds": list(SEEDS),
        "factors": [{"name": name, "kind": kind, "levels": list(levels)}
                    for name, (kind, levels) in FACTOR_LEVELS.items()],
        "checkpoints": {key: asset_manifest(key) for key in MODELS},
        "scoring": "task_success exige une lecture valide suivie d'une réponse JSON entière exacte. "
                   "Deux générations par essai, mêmes outils et budgets. Une panne transitoire bloque la première lecture.",
        "comparison": "Pilote descriptif apparié. Comparer les checkpoints sous un protocole commun. "
                      "La taille, l'architecture et l'entraînement co-varient avec le checkpoint.",
        "total_evaluations": EVALUATIONS, "calls_per_evaluation": 2, "tokens_per_call": MAX_INPUT + MAX_OUTPUT,
    }


def validate_capability(document):
    design = ExperimentalDesign.model_validate(document)
    levels = {factor.name: (factor.kind, set(factor.levels)) for factor in design.factors}
    expected = {name: (kind, set(values)) for name, (kind, values) in FACTOR_LEVELS.items()}
    if levels != expected or design.task_suite.source_type != "synthetic_local":
        raise ValueError("Agentic comparison requires the complete approved checkpoint × policy × shift grid")
    limits = design.resource_limits
    if (limits.tasks_per_cell != len(TASKS) or set(limits.seeds) != set(SEEDS)
            or limits.max_llm_calls_per_evaluation < 2 or limits.max_tokens_per_llm_call < MAX_INPUT + MAX_OUTPUT):
        raise ValueError("Agentic protocol does not cover the approved tasks, seeds and inference budgets")
    if set(metric.id for metric in design.metrics) != {"task_success", "total_tokens", "latency_ms"} or any(
            test.primary_metric_id != "task_success" for test in design.hypothesis_tests):
        raise ValueError("Agentic primary measurement is grounded task success")
    return design


def compile_study(source):
    expected = {"kind", "run_id", "version", "mission", "literature", "protocol", "evaluation"}
    if set(source) != expected or source["kind"] != STUDY_KIND:
        raise ValueError("Invalid agentic study source")
    protocol = source["protocol"]
    evaluation = suite().model_dump(mode="json")
    if (source["evaluation"] != evaluation or protocol["evaluation_sha256"] != fingerprint(evaluation)
            or protocol["execution_constraints"] != design_constraints()):
        raise ValueError("Agentic corpus or capability differs from the approved protocol")
    mission, _, review = validate_approved_literature(source["mission"], source["literature"])
    selection = protocol["resource_selection"]
    choice = ResourceChoice.model_validate(selection["decision"])
    if (choice.resource_id != AGENTIC_ID or choice.question_excerpt not in mission.research_question
            or selection["resource"]["evaluation_sha256"] != protocol["evaluation_sha256"]
            or selection["mission_sha256"] != fingerprint(source["mission"])
            or selection["literature_sha256"] != fingerprint(source["literature"])):
        raise ValueError("Agentic resource provenance mismatch")
    design = validate_capability(protocol["document"])
    validate_design_against_inputs(design, mission, review)
    config = AgenticGPUConfig.model_validate(protocol["gpu_config"])
    protocol_sha = fingerprint(protocol)
    campaign = f"{source['run_id']}-v{source['version']}-agentic"
    jobs = []
    for checkpoint in MODELS:
        trials = []
        for seed in SEEDS:
            group = []
            for cell in design.factor_matrix():
                levels = cell["factor_levels"]
                if levels["checkpoint"] == checkpoint:
                    for case in TASKS:
                        group.append({"id": f"{cell['cell_id']}-{case}-{seed}", "cell": cell["cell_id"], "case": case,
                                      "seed": seed, "profile": levels["shift"], "policy": levels["policy"]})
            Random(seed).shuffle(group)
            trials.extend(group)
        jobs.append(ExperimentJob(
            experiment_id=campaign, job_id=f"{campaign}-{checkpoint}", seed=SEEDS[0],
            source_design_sha256=protocol_sha, expected_artifacts=[ARTIFACT],
            resources=ResourceRequest(cpus=2, memory_mb=16_384, gpus=1, walltime_minutes=MINUTES),
            workload=WorkloadSpec(name=WORKLOAD, parameters={"model_key": checkpoint, "trials": trials}),
        ))
    plan = {"capability": CAPABILITY, "resource_id": AGENTIC_ID, "protocol_sha256": protocol_sha,
            "evaluation_sha256": protocol["evaluation_sha256"], "worker_sha256": protocol["worker_sha256"],
            "evaluations": EVALUATIONS, "max_generations": 2 * EVALUATIONS,
            "max_input_tokens": 2 * EVALUATIONS * MAX_INPUT, "max_output_tokens": 2 * EVALUATIONS * MAX_OUTPUT,
            **execution_scope(config), "jobs": [{"spec": job.model_dump(mode="json"), "sha256": job.fingerprint()} for job in jobs]}
    return design, jobs, {**plan, "plan_sha256": fingerprint(plan)}
