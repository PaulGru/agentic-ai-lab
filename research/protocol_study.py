"""Build an executable CPU plan from an approved design, corpus and code proposal."""
from __future__ import annotations

from analysis.contracts import fingerprint
from execution.contracts import ExperimentJob, ResourceRequest, WorkloadSpec
from execution.protocol_cpu import ARTIFACT, WORKLOAD
from execution.protocol_gpu import (
    ARTIFACT as GPU_ARTIFACT, CAPABILITY as GPU_CAPABILITY, MAX_INPUT_TOKENS, MAX_NEW_TOKENS,
    MAX_TRIALS, MINUTES, MODEL, REVISION, WORKLOAD as GPU_WORKLOAD, ProtocolGPUConfig, provisioning_scope,
)
from experiments.proposal import ExperimentProposal
from experiments.protocol_program import EvaluationSet, RestrictedProgram, interpreter_sha256
from research.experimental_design import ExperimentalDesign, validate_approved_literature, validate_design_against_inputs
from research.capability_resolution import verify_resolved_resource
from research.resource_catalog import ResourceChoice

STUDY_KIND = "compiled_protocol_pilot"
MAX_CELLS = 24
MAX_EVALUATIONS = 240
SCORING = "Correspondance exacte avec l'entier de référence. Une réponse invalide compte comme un échec."

IMPLEMENTATION_PROMPT = """Implémente le protocole approuvé avec la capacité fournie.
Le code doit contenir exactement def predict(features, levels, seed): suivi d'un unique return.
Accède aux entrées entières avec features["nom"] et aux modalités avec levels["facteur"].
Expressions admises : entiers, chaînes courtes pour les modalités, +, -, *, //, %, comparaisons simples et a if condition else b.
Utilise uniquement les entrées décrites et renvoie un entier. Aucun appel, import, boucle, accès fichier ou dépendance.
Le programme reçoit une tâche et une cellule par appel. Toutes les graines et toutes les conditions sont planifiées hors du code.
Les témoins vérifient l'implémentation. L'évaluateur indépendant conserve les cas de mesure et calcule les scores.
Conserve le protocole approuvé et utilise sa première graine pour le champ seed de ta proposition."""


def validate_capability(document, evaluation: EvaluationSet):
    design = ExperimentalDesign.model_validate(document)
    cells = design.factor_matrix()
    if design.task_suite.source_type != "synthetic_local" or any(f.kind in {"model_scale", "architecture"} for f in design.factors):
        raise ValueError("This capability supports local tasks with a fixed inference environment")
    if set(m.id for m in design.metrics) != {"task_success", "total_tokens", "latency_ms"} or any(
            test.primary_metric_id != "task_success" for test in design.hypothesis_tests):
        raise ValueError("This capability only supports integer exact match as the primary metric")
    if len(evaluation.cases) != design.resource_limits.tasks_per_cell:
        raise ValueError("The approved task count differs from the evaluation corpus")
    expected = {tuple(sorted(cell["factor_levels"].items())) for cell in cells}
    witnessed = {tuple(sorted(item.levels.items())) for item in evaluation.witnesses}
    if witnessed != expected:
        raise ValueError("The evaluation corpus needs an implementation witness for every approved cell")
    if evaluation.schema_version == GPU_CAPABILITY:
        if design.total_evaluations > MAX_TRIALS:
            raise ValueError("The GPU inference capability supports at most 32 evaluations")
        if design.resource_limits.max_tokens_per_llm_call < MAX_INPUT_TOKENS + MAX_NEW_TOKENS:
            raise ValueError("The GPU token envelope exceeds the protocol budget")
    return design


def design_constraints(evaluation: EvaluationSet):
    gpu = evaluation.schema_version == GPU_CAPABILITY
    return {
        "capability": evaluation.schema_version,
        "instructions": (
            "Conserve exactement les cellules couvertes par les témoins et le nombre de tâches indiqué. "
            "Prends task_success comme métrique primaire. L'analyse disponible est descriptive et appariée."
        ),
        "source_type": "synthetic_local", "task_count": len(evaluation.cases),
        "scoring": SCORING, "max_evaluations": MAX_TRIALS if gpu else MAX_EVALUATIONS,
        "cells": [dict(levels) for levels in dict.fromkeys(tuple(sorted(w.levels.items())) for w in evaluation.witnesses)],
        "metrics": ["task_success", "total_tokens", "latency_ms"],
        "gpu_model": {"id": MODEL, "revision": REVISION} if gpu else None,
    }


def implementation_request(document, evaluation: EvaluationSet, *, feedback):
    request = {
        "instructions": IMPLEMENTATION_PROMPT, "capability": evaluation.schema_version, "protocol": document,
        "features": sorted(evaluation.cases[0].inputs),
        "witnesses": [item.model_dump(mode="json") for item in evaluation.witnesses],
        "repair_feedback": feedback,
    }
    if evaluation.schema_version == GPU_CAPABILITY:
        request["model"] = {"id": MODEL, "revision": REVISION}
        request["instructions"] += (
            f" En mode GPU, renvoie l'indice de la variante de prompt autorisée, de 0 à {len(evaluation.cases[0].prompts) - 1}."
            " L'inférence avec le modèle fixé produira l'entier évalué.")
    return request


def compile_study(source):
    expected = {"kind", "run_id", "version", "mission", "literature", "protocol", "evaluation", "proposal"}
    if set(source) != expected or source["kind"] != STUDY_KIND:
        raise ValueError("Invalid executable study envelope")
    protocol = source["protocol"]
    mission, _, review = validate_approved_literature(source["mission"], source["literature"])
    evaluation = EvaluationSet.model_validate(source["evaluation"])
    gpu = evaluation.schema_version == GPU_CAPABILITY
    protocol_keys = {"version", "document", "workload", "evaluation_sha256", "execution_constraints", "interpreter_sha256"}
    if gpu:
        protocol_keys |= {"gpu_config", "worker_sha256"}
    if "resource_selection" in protocol:
        protocol_keys.add("resource_selection")
    if set(protocol) != protocol_keys or protocol["workload"] != "unmapped":
        raise ValueError("Expected an approved protocol and its execution capability")
    if protocol["evaluation_sha256"] != fingerprint(evaluation.model_dump(mode="json")):
        raise ValueError("Evaluation corpus differs from the approved protocol")
    if protocol["execution_constraints"] != design_constraints(evaluation):
        raise ValueError("Execution constraints differ from the approved evaluator")
    if protocol["interpreter_sha256"] != interpreter_sha256():
        raise ValueError("The interpreter source differs from the approved version")
    if selection := protocol.get("resource_selection"):
        choice = ResourceChoice.model_validate(selection["decision"])
        resource = selection["resource"]
        if (choice.resource_id != resource["id"] or choice.question_excerpt not in mission.research_question
                or resource["capability"] != evaluation.schema_version or resource["evaluation_sha256"] != protocol["evaluation_sha256"]
                or selection["mission_sha256"] != fingerprint(source["mission"])
                or selection["literature_sha256"] != fingerprint(source["literature"])):
            raise ValueError("The selected resource differs from the approved study")
        if resource["id"].startswith("generated_cpu_") or "capability_resolution" in selection:
            if "capability_resolution" not in selection:
                raise ValueError("Generated resource lacks capability resolution provenance")
            verify_resolved_resource(selection, evaluation, mission.research_question)
    design = validate_capability(protocol["document"], evaluation)
    validate_design_against_inputs(design, mission, review)
    proposed = ExperimentProposal.model_validate(source["proposal"])
    seeds = design.resource_limits.seeds
    if proposed.dependencies or proposed.seed != seeds[0]:
        raise ValueError("Expected the first approved seed and no dependencies")
    program = RestrictedProgram(proposed.code, features=evaluation.cases[0].inputs,
                                levels=[factor.name for factor in design.factors])
    for number, witness in enumerate(evaluation.witnesses, start=1):
        if program.predict(witness.inputs, witness.levels, seeds[0]) != witness.expected:
            raise ValueError(f"Implementation witness {number} failed under the approved control values")
    protocol_sha = fingerprint(protocol)
    proposal_sha = fingerprint(source["proposal"])
    if gpu:
        jobs, execution = compile_gpu(source, design, evaluation, program, protocol_sha)
    else:
        jobs = compile_cpu(source, design, evaluation, proposed.code, protocol_sha)
        execution = {"network": False, "gpu": False}
    plan = {
        "capability": evaluation.schema_version, "protocol_sha256": protocol_sha, "proposal_sha256": proposal_sha,
        "evaluation_sha256": protocol["evaluation_sha256"], "witnesses_passed": len(evaluation.witnesses),
        "interpreter_sha256": protocol["interpreter_sha256"],
        "evaluations": design.total_evaluations, **execution,
        "jobs": [{"spec": job.model_dump(mode="json"), "sha256": job.fingerprint()} for job in jobs],
    }
    if selection:
        plan["resource_id"] = selection["resource"]["id"]
        if "capability_resolution" in selection:
            plan["capability_resolution"] = selection["capability_resolution"]
    return design, evaluation, jobs, {**plan, "plan_sha256": fingerprint(plan)}


def compile_cpu(source, design, evaluation, code, protocol_sha):
    cases = [{"id": item.id, "inputs": item.inputs} for item in evaluation.cases]
    campaign = f"{source['run_id']}-v{source['version']}-cpu"
    seeds = design.resource_limits.seeds
    return [
        ExperimentJob(
            experiment_id=campaign, job_id=f"{campaign}-{cell['cell_id']}", seed=seeds[0],
            source_design_sha256=protocol_sha, expected_artifacts=[ARTIFACT],
            resources=ResourceRequest(cpus=1, memory_mb=128, gpus=0, walltime_minutes=1),
            workload=WorkloadSpec(name=WORKLOAD, parameters={
                "cell_id": cell["cell_id"], "levels": cell["factor_levels"], "seeds": seeds,
                "source": code, "cases": cases,
            }),
        )
        for cell in design.factor_matrix()
    ]


def compile_gpu(source, design, evaluation, program, protocol_sha):
    protocol = source["protocol"]
    config = ProtocolGPUConfig.model_validate(protocol["gpu_config"])
    campaign = f"{source['run_id']}-v{source['version']}-gpu"
    seeds = design.resource_limits.seeds
    trials = []
    for cell in design.factor_matrix():
        for seed in seeds:
            for case in evaluation.cases:
                variant = program.predict(case.inputs, cell["factor_levels"], seed)
                if not 0 <= variant < len(case.prompts):
                    raise ValueError("Program selected a prompt outside the approved variants")
                trials.append({"cell_id": cell["cell_id"], "case_id": case.id, "seed": seed,
                               "variant": variant, "prompt": case.prompts[variant]})
    job = ExperimentJob(
        experiment_id=campaign, job_id=f"{campaign}-batch", seed=seeds[0],
        source_design_sha256=protocol_sha, expected_artifacts=[GPU_ARTIFACT],
        resources=ResourceRequest(cpus=2, memory_mb=16_384, gpus=1, walltime_minutes=MINUTES),
        workload=WorkloadSpec(name=GPU_WORKLOAD, parameters={"trials": trials, "model": MODEL, "revision": REVISION}),
    )
    provisioning = provisioning_scope(config)
    execution = {"network": True, "gpu": True, "gpu_config": protocol["gpu_config"],
                 "worker_sha256": protocol["worker_sha256"],
                 "model_preflight": "read_only_local_snapshot", "max_ssh_commands": 2,
                 "model": {"id": MODEL, "revision": REVISION, "local_snapshot": config.snapshot_path},
                 "max_input_tokens": len(trials) * MAX_INPUT_TOKENS, "max_output_tokens": len(trials) * MAX_NEW_TOKENS}
    if provisioning:
        execution.update(model_provisioning=provisioning, max_ssh_commands=3)
    return [job], execution
