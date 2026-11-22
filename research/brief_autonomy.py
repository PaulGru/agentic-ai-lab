"""Run eligible research pilots under one journaled, bounded human delegation."""
from __future__ import annotations

from pydantic import Field

from analysis.contracts import fingerprint
from analysis.ood_replay import analyze_ood_gpu
from execution.agentic_gpu import (
    CAPABILITY as AGENTIC_CAPABILITY, MAX_INPUT as AGENTIC_INPUT, MAX_OUTPUT as AGENTIC_OUTPUT,
    MODELS as AGENTIC_MODELS, AgenticGPUConfig, execution_scope as agentic_execution_scope,
    worker_sha256 as agentic_worker_sha256,
)
from execution.contracts import StrictModel
from execution.job_manager import DurableJobManager
from execution.ood_gpu import OODGPUConfig, OODGPUProtocol
from execution.ood_replay import OODReplayProtocol, fixture_sha256
from execution.protocol_cpu import CAPABILITY as CPU_CAPABILITY
from execution.protocol_gpu import (
    CAPABILITY as GPU_PROGRAM_CAPABILITY, MAX_INPUT_TOKENS, MAX_NEW_TOKENS, MAX_TRIALS,
    MINUTES, MODEL, REVISION, ProtocolGPUConfig, provisioning_scope, worker_sha256,
)
from experiments.protocol_program import interpreter_sha256
from research.brief_workflow import (
    PROGRAM_STAGE, IMPLEMENTATION_STAGE, RESOLUTION_STAGE, RESOURCE_STAGE, Decision, GPU_STAGE, GPU_WORKLOAD,
    MAX_INPUT_BYTES, OOD_STAGE, OOD_WORKLOAD,
)
from research.capability_resolution import authorized_resolution, resolution_scope
from research.mission import sha256_text
from research.agentic_study import EVALUATIONS as AGENTIC_EVALUATIONS
from research.protocol_study import MAX_CELLS, MAX_EVALUATIONS
from research.resource_catalog import resources_for

LLM_STAGES = {"mission", "literature", RESOURCE_STAGE, RESOLUTION_STAGE, "protocol", IMPLEMENTATION_STAGE}


def _compiled_protocol(state):
    return state["workload"] == "unmapped" and bool(state.get("evaluation") or state.get("auto_resources"))


class AutonomyBudget(StrictModel):
    llm_calls: int = Field(default=4, ge=2, le=8)
    openalex_queries: int = Field(default=8, ge=2, le=16)
    revisions: int = Field(default=2, ge=0, le=4)


def _execution_scope(details):
    # Campaign IDs and provenance hashes are filled in after mission and literature review.
    return {
        **{key: value for key, value in details.items() if key not in {"protocol_sha256", "jobs"}},
        "jobs": [{key: value for key, value in item["spec"].items()
                  if key not in {"experiment_id", "job_id", "source_design_sha256"}}
                 for item in details["jobs"]],
    }


def _program_limits(state, capability):
    if capability == AGENTIC_CAPABILITY:
        config = AgenticGPUConfig.model_validate(state["gpu_config"])
        return {"capability": capability, "worker_sha256": agentic_worker_sha256(),
                "max_jobs": len(AGENTIC_MODELS), "max_evaluations": AGENTIC_EVALUATIONS,
                "max_generations": 2 * AGENTIC_EVALUATIONS,
                "max_input_tokens": 2 * AGENTIC_EVALUATIONS * AGENTIC_INPUT,
                "max_output_tokens": 2 * AGENTIC_EVALUATIONS * AGENTIC_OUTPUT, **agentic_execution_scope(config)}
    common = {"capability": capability, "interpreter_sha256": interpreter_sha256()}
    if capability == CPU_CAPABILITY:
        return {**common, "network": False, "gpu": False, "max_jobs": MAX_CELLS, "max_evaluations": MAX_EVALUATIONS}
    setup = state["gpu_config"]
    provisioning = provisioning_scope(ProtocolGPUConfig.model_validate(setup))
    return {**common, "network": True, "gpu": True, "max_jobs": 1, "max_evaluations": MAX_TRIALS,
            "gpu_config": setup, "worker_sha256": worker_sha256(), "max_input_tokens": MAX_TRIALS * MAX_INPUT_TOKENS,
            "max_output_tokens": MAX_TRIALS * MAX_NEW_TOKENS, "walltime_minutes": MINUTES,
            "max_ssh_commands": 3 if provisioning else 2, "model_provisioning": provisioning,
            "model": {"id": MODEL, "revision": REVISION, "local_snapshot": setup["snapshot_path"]}}


def _scope(workflow, state):
    workload = state["workload"]
    common = {
        "run_id": state["id"], "brief_sha256": sha256_text(state["brief"]), "workload": workload,
        "llm": state["llm"], "max_input_bytes_per_call": MAX_INPUT_BYTES,
    }
    if _compiled_protocol(state):
        if state.get("auto_resources"):
            catalog = [metadata for metadata, _ in resources_for(state.get("gpu_config")).values()]
            capabilities = {entry["capability"]: _program_limits(state, entry["capability"]) for entry in catalog}
            return {**common, "resource_catalog": catalog, "capability_resolution": resolution_scope(),
                    "execution": {"capabilities": capabilities}}
        capability = state["evaluation"]["schema_version"]
        return {**common, "task_data_sha256": fingerprint(state["evaluation"]),
                "execution": _program_limits(state, capability)}
    protocol_fields = dict(campaign_id=state["id"], requested_question="",
                           mission_sha256="0" * 64, literature_sha256="0" * 64)
    if workload == GPU_WORKLOAD:
        protocol = OODGPUProtocol(**protocol_fields, config=OODGPUConfig.model_validate(state["gpu_config"]))
        task_data_sha256 = protocol.task_sha256
    elif workload == OOD_WORKLOAD:
        protocol = OODReplayProtocol(**protocol_fields, fixture_sha256=fixture_sha256())
        task_data_sha256 = protocol.fixture_sha256
    else:
        raise ValueError("Le mode autonome exige un pilote OOD ou un corpus numérique indépendant pour un protocole inédit.")
    return {
        **common,
        "task_data_sha256": task_data_sha256,
        "execution": _execution_scope(workflow._ood_details(protocol)),
    }


def autonomy_plan(workflow, budget):
    """Describe exactly what can run and bind consent to the untouched starting checkpoint."""
    event = workflow.store.current()
    state = event["state"]
    if event["action"] != "start" or state["root"] != str(workflow.root):
        raise PermissionError("Autoriser le mode autonome au départ d'une nouvelle mission.")
    budget = AutonomyBudget.model_validate(budget)
    plan = {
        "kind": "autonomy", "checkpoint_sha256": event["sha256"],
        "prompt": "Autoriser cette mission autonome dans le périmètre et le budget affichés ?",
        "details": {
            "scope": _scope(workflow, state), "budget": budget.model_dump(),
            "openalex": {"subject": "Requêtes bibliographiques proposées à partir du brief",
                         "results_per_query": 5, "max_http_requests": 3 * budget.openalex_queries},
            "max_output_tokens_total": budget.llm_calls * state["llm"]["max_output_tokens"],
            "execution_batches": 1,
            "publication": "brouillon local à relire" if _compiled_protocol(state) else "export local du manuscrit de calibration",
        },
    }
    return {**plan, "target_sha256": fingerprint(plan)}


def authorize_autonomy(workflow, budget, decision):
    decision = Decision.model_validate(decision.model_dump())
    plan = autonomy_plan(workflow, budget)
    if not decision.actor.strip() or decision.target_sha256 != plan["target_sha256"]:
        raise PermissionError("L'autorisation ne correspond pas au périmètre affiché.")
    event = workflow.store.current()
    workflow.store.append(plan["checkpoint_sha256"], event["state"], action="authorize_autonomy",
                          detail={"plan": plan, "decision": decision.model_dump()})


def _authorized_limits(scope, details):
    if "resource_catalog" in scope:
        if "capability_resolution" in details:
            if authorized_resolution(scope.get("capability_resolution"), details):
                return scope["execution"]["capabilities"].get(CPU_CAPABILITY)
            return None
        resource = next((row for row in scope["resource_catalog"] if row["id"] == details.get("resource_id")), None)
        if (resource is None or resource["evaluation_sha256"] != details["evaluation_sha256"]
                or resource["capability"] != details["capability"]):
            return None
        return scope["execution"]["capabilities"].get(details["capability"])
    limits = scope["execution"]
    if limits.get("capability") == details["capability"] and details["evaluation_sha256"] == scope.get("task_data_sha256"):
        return limits
    return None


def _reservation_block(gate, history, plan):
    budget = AutonomyBudget.model_validate(plan["details"]["budget"])
    reservations = [event["detail"]["gate"] for event in history if event["action"] == "reserve"]
    stages = [gate["stage"] for gate in reservations]
    stage, details = gate["stage"], gate["details"]
    if gate["kind"] == "revision":
        revisions = sum(event["action"] in {"approve_revision", "revise"} for event in history)
        if revisions >= budget.revisions:
            return "Budget de révisions épuisé."
        proposal = details["proposal"]
        stage, details = proposal["stage"], proposal
    if stage in LLM_STAGES:
        if gate["kind"] != "revision":
            scope = plan["details"]["scope"]
            if stage == RESOURCE_STAGE and details["catalog"] != scope.get("resource_catalog"):
                return "Le catalogue diffère des ressources autorisées."
            if stage == RESOLUTION_STAGE:
                if details["resolution_scope"] != scope.get("capability_resolution"):
                    return "La résolution du manque est hors du périmètre autorisé."
                if stages.count(RESOLUTION_STAGE) >= details["resolution_scope"]["max_attempts"]:
                    return "La tentative de résolution déléguée a déjà été réservée."
            if stage == IMPLEMENTATION_STAGE and _authorized_limits(scope, details) is None:
                return "La préparation demandée diffère du corpus et de la capacité autorisés."
        if sum(name in LLM_STAGES for name in stages) >= budget.llm_calls:
            return "Budget d'appels au fournisseur épuisé."
    elif stage == "catalog":
        used = sum(len(item["details"]["queries"]) for item in reservations if item["stage"] == "catalog")
        if used + len(details["queries"]) > budget.openalex_queries:
            return "Budget de recherches OpenAlex épuisé."
    elif stage in {OOD_STAGE, GPU_STAGE}:
        if any(name in {OOD_STAGE, GPU_STAGE} for name in stages):
            return "Le lot autorisé a déjà été réservé."
        if _execution_scope(details) != plan["details"]["scope"]["execution"]:
            return "Le lot préparé diffère du lot autorisé."
    elif stage == PROGRAM_STAGE:
        scope = plan["details"]["scope"]
        limits = _authorized_limits(scope, details)
        if PROGRAM_STAGE in stages:
            return "Le lot autorisé a déjà été réservé."
        if (limits is None or details.get("interpreter_sha256") != limits.get("interpreter_sha256")
                or details["network"] != limits["network"] or details["gpu"] != limits["gpu"]
                or len(details["jobs"]) > limits["max_jobs"] or details["evaluations"] > limits["max_evaluations"]):
            return "Le lot dépasse le périmètre de la délégation."
        if details["capability"] == AGENTIC_CAPABILITY:
            bound = ("gpu_config", "worker_sha256", "models", "max_download_bytes",
                     "download_minutes_per_model", "walltime_minutes_per_model")
            quantities = ("max_generations", "max_input_tokens", "max_output_tokens", "max_ssh_commands")
            if (any(details[key] != limits[key] for key in bound)
                    or any(details[key] > limits[key] for key in quantities)
                    or any(job["spec"]["resources"]["walltime_minutes"] > limits["walltime_minutes_per_model"]
                           for job in details["jobs"])):
                return "Le lot agentique diffère des checkpoints, de la cible ou du budget autorisés."
            return None
        if details["gpu"] and (
            any(details.get(key) != limits.get(key) for key in ("gpu_config", "worker_sha256", "model", "model_provisioning"))
            or any(details[key] > limits[key] for key in ("max_input_tokens", "max_output_tokens", "max_ssh_commands"))
            or any(item["spec"]["resources"]["walltime_minutes"] > limits["walltime_minutes"] for item in details["jobs"])
        ):
            return "Le lot GPU diffère du modèle, de la cible ou du budget autorisés."
    elif stage == "review_manuscript" and plan["details"]["publication"] == "brouillon local à relire":
        return "Brouillon prêt pour la validation éditoriale."
    elif stage not in {"review_mission", "review_literature", "review_protocol", "review_manuscript"}:
        return "Cette étape exige une autorisation distincte."
    return None


def _pause(workflow, event, reason):
    state = event["state"]
    # Leave the reservation chain intact when recovery still needs to find the original job.
    if state["status"] == "running" or (state.get("block") or {}).get("uncertain"):
        return {**workflow.status(), "autonomy_stop": reason}
    detail = {"reason": reason}
    if event["action"] != "autonomy_pause" or event["detail"] != detail:
        workflow.store.append(event["sha256"], state, action="autonomy_pause", detail=detail)
    return {**workflow.status(), "autonomy_stop": reason}


def run_autonomously(workflow):
    """Use the same approval and durable reservation paths as the guided workflow."""
    while True:
        history = workflow.store.history()
        if len(history) < 2 or history[1]["action"] != "authorize_autonomy":
            raise PermissionError("Aucune autorisation de mission autonome enregistrée.")
        consent = history[1]["detail"]
        plan = consent["plan"]
        target = plan["target_sha256"]
        if (consent["decision"]["target_sha256"] != target or plan["checkpoint_sha256"] != history[0]["sha256"]
                or fingerprint({key: value for key, value in plan.items() if key != "target_sha256"}) != target):
            raise PermissionError("L'autorisation de mission autonome a changé.")

        status = workflow.status()
        if status["checkpoint_sha256"] != history[-1]["sha256"]:
            raise PermissionError("Le workflow a avancé pendant la vérification de la délégation.")
        gate = status["pending"]
        stage = status["stage"]
        interrupted_local_step = status["status"] == "waiting" and stage in {"protocol", "draft"}
        uncertain_job = stage in {OOD_STAGE, GPU_STAGE, PROGRAM_STAGE} and (status["block"] or {}).get("uncertain", False)
        if not (gate or interrupted_local_step or uncertain_job):
            return status
        state = history[-1]["state"]
        approved_scope = plan["details"]["scope"]
        supported = state["workload"] in {OOD_WORKLOAD, GPU_WORKLOAD} or _compiled_protocol(state)
        current_scope = _scope(workflow, state) if supported else {}
        if "capability_resolution" not in approved_scope:
            current_scope.pop("capability_resolution", None)  # Older delegations did not include this action.
        if current_scope != approved_scope:
            return _pause(workflow, history[-1], "Le périmètre de la mission a changé.")
        actor = f"autonomy:{target}"
        if not gate:
            try:
                workflow.recover(Decision(actor=actor, target_sha256=status["checkpoint_sha256"]))
            except (OSError, ValueError):
                return status  # Recovery can attach verified receipts, and never submits a job.
            continue
        stop = _reservation_block(gate, history[2:], plan)
        if stop:
            return _pause(workflow, history[-1], stop)
        if gate["stage"] == "review_manuscript" and state["workload"] == GPU_WORKLOAD:
            protocol, ledger = workflow._ood_execution_inputs(state)
            analysis = analyze_ood_gpu(protocol, DurableJobManager(path=ledger, read_only=True))
            if not analysis.reference_calibration_passed:
                return _pause(workflow, history[-1], "La calibration de référence a échoué. Relire le brouillon et le reçu.")
        workflow.approve(Decision(actor=actor, target_sha256=gate["target_sha256"]))
