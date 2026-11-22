"""Project verified workflow records into a metadata-only export."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import os
import re

from analysis.contracts import fingerprint
from analysis.ood_replay import IncompleteOODReceipt, analyze_ood_gpu, analyze_ood_replay
from execution.campaign import save_report
from execution.job_manager import DurableJobManager
from execution.ood_gpu import OODGPUProtocol, validate_receipt
from research.brief_workflow import GPU_WORKLOAD, OOD_WORKLOAD, PROMPTS


ROLES = {"mission": "mission_planner", "literature": "literature_analyst", "protocol": "experimental_design_planner"}
STAGES = {*PROMPTS, "draft", "complete"}
ACTIONS = {"start", "authorize_autonomy", "autonomy_pause", "reserve", "result", "revise", "approve_revision",
           "prepare_protocol", "prepare_draft", "recover_local", "recover_synthetic_report"}


def _agent_calls(history, records):
    reservations = {event["sha256"]: event for event in history if event["action"] == "reserve"}
    outcomes = {event["previous_sha256"]: event["state"] for event in history if event["action"] == "result"}
    calls = {}
    for record in records:
        reservation = record["reservation_sha256"]
        event = reservations.get(reservation)
        role = record["prompt_id"]
        call_id = record["call_id"]
        version = record["prompt_version"]
        digest = record["instructions_sha256"]
        started = record["started_at_ns"]
        if (not event or ROLES.get(event["state"]["stage"]) != role or record["operation"] != "start"
                or not re.fullmatch(r"[a-f0-9]{32}", call_id) or not re.fullmatch(r"[a-f0-9]{64}", digest)
                or type(version) is not int or version < 1 or type(started) is not int or started <= 0):
            raise ValueError("Model trace does not match a supported workflow reservation")
        status = record["status"]
        metadata = {"call_id": call_id, "reservation_sha256": reservation, "stage": event["state"]["stage"],
                    "agent": role, "prompt_version": version, "instructions_sha256": digest, "started_at_ns": started}
        if status == "started" and call_id not in calls:
            calls[call_id] = {**metadata, "status": status, "latency_ms": None}
            continue
        elapsed = record.get("latency_ms")
        if (call_id not in calls or calls[call_id] != {**metadata, "status": "started", "latency_ms": None}
                or status not in {"returned", "error"} or type(elapsed) is not int or elapsed < 0):
            raise ValueError("Invalid model trace sequence")
        calls[call_id] = {**metadata, "status": status, "latency_ms": elapsed}
    for call in calls.values():
        state = outcomes.get(call["reservation_sha256"])
        if state is None:
            validation = "pending"
        elif state.get("block", {}).get("uncertain"):
            validation = "uncertain"
        elif state.get("block"):
            validation = "rejected"
        else:
            validation = "accepted"
        call["stage_outcome"] = validation
    return list(calls.values())


def _experiment(workflow, state):
    if state["workload"] not in {GPU_WORKLOAD, OOD_WORKLOAD} or "protocol" not in state["artifacts"]:
        return None
    _, ledger, _ = workflow._trial_paths(state)
    if not ledger.is_file():
        return None
    protocol, ledger = workflow._ood_execution_inputs(state)
    manager = DurableJobManager(path=ledger, read_only=True)
    gpu = isinstance(protocol, OODGPUProtocol)
    try:
        analysis = analyze_ood_gpu(protocol, manager) if gpu else analyze_ood_replay(protocol, manager)
    except IncompleteOODReceipt:
        return None
    metrics = {}
    for group, arm in analysis.arms.items():
        metrics.update({f"experiment/{group}/correct": arm.correct, f"experiment/{group}/total": arm.total,
                        f"experiment/{group}/accuracy_percent": arm.accuracy_percent})
    metrics.update({f"experiment/{model}/gap_points": value for model, value in analysis.degradation_points.items()})
    result = {"protocol_sha256": analysis.protocol_sha256, "simulated": analysis.simulated,
              "scientific_evidence": analysis.scientific_evidence, "metrics": metrics, "trials": []}
    if gpu:
        job = protocol.job()
        record = manager.get(job.job_id)
        receipt = validate_receipt(protocol, job, record.artifacts.files["responses.json"])
        result.update(model_revision=receipt.revision, receipt_sha256=record.artifacts_sha256)
        result["metrics"].update({"gpu/elapsed_ms": receipt.runtime.elapsed_ms,
                                  "gpu/peak_vram_bytes": receipt.runtime.peak_vram_bytes,
                                  "gpu/reference_calibration_passed": int(analysis.reference_calibration_passed)})
        result["trials"] = [{"trial_id": response.id, "latency_ms": response.latency_ms, "input_tokens": response.input_tokens,
                             "output_tokens": response.output_tokens, "correct": trial.correct}
                            for response, trial in zip(receipt.responses, analysis.trials, strict=True)]
    return result


def export_telemetry(workflow):
    history = workflow.store.history()
    if not history or history[-1]["state"]["root"] != str(workflow.root):
        raise ValueError("An existing workflow in this directory is required")
    checkpoint = history[-1]["sha256"]
    state = history[-1]["state"]
    if state["status"] == "running":
        raise ValueError("Wait for the reserved step to return or be reconciled before exporting telemetry")
    events = [{"step": index, "version": event["state"]["version"],
               "stage": stage if (stage := event["state"]["stage"]) in STAGES else "other",
               "action": event["action"] if event["action"] in ACTIONS else "other",
               "status": status if (status := event["state"]["status"]) in {"running", "waiting", "blocked", "complete"} else "other"}
              for index, event in enumerate(history, 1)]
    payload = {"schema_version": 1, "checkpoint_sha256": checkpoint, "events": events,
               "agent_calls": _agent_calls(history, workflow.store.model_traces()), "experiment": _experiment(workflow, state)}
    if workflow.store.current()["sha256"] != checkpoint:
        raise ValueError("Workflow advanced during telemetry export; retry from the latest checkpoint")
    directory = workflow.root / "telemetry" / fingerprint(payload)
    path = directory / "metrics.json"
    if not path.resolve().is_relative_to(workflow.root):
        raise ValueError("Telemetry must stay inside the workflow directory")
    save_report(payload, path)
    return payload, path


def export_code_telemetry(state, directory):
    from experiments.graph import code_digest

    attempts = []
    for index, attempt in enumerate(state.get("attempts", []), 1):
        result = attempt["execution"]
        seconds, status, exit_code = result["duration_seconds"], result["status"], result["exit_code"]
        if (attempt["number"] != index or code_digest(attempt["code"]) != attempt["code_sha256"]
                or status not in {"success", "error", "timeout", "output_limit"}
                or type(seconds) not in {int, float} or not math.isfinite(seconds) or seconds < 0
                or (exit_code is not None and type(exit_code) is not int)):
            raise ValueError("Invalid code execution telemetry source")
        attempts.append({"attempt": index, "code_sha256": attempt["code_sha256"], "status": status,
                         "exit_code": exit_code, "duration_seconds": seconds})
    calls = []
    for record in state.get("model_calls", []):
        identity = record["prompt_id"]
        if (identity not in {"experiment", "experiment_reviser"}
                or not re.fullmatch(r"[a-f0-9]{64}", record["instructions_sha256"])
                or not re.fullmatch(r"[a-f0-9]{32}", record["call_id"]) or record["status"] not in {"returned", "error"}
                or any(type(record[key]) is not int or record[key] < 0
                       for key in ("prompt_version", "started_at_ns", "latency_ms"))):
            raise ValueError("Invalid code agent trace")
        calls.append({"agent": identity, **{key: record[key] for key in (
            "call_id", "prompt_version", "instructions_sha256", "started_at_ns", "status", "latency_ms")}})
    payload = {"schema_version": 1, "events": [], "experiment": None, "agent_calls": calls, "code_execution": attempts}
    payload["checkpoint_sha256"] = fingerprint(payload)
    path = directory / payload["checkpoint_sha256"] / "metrics.json"
    save_report(payload, path)
    return payload, path


def record_wandb_offline(payload, directory):
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("Installer requirements-observability.txt pour l'export W&B.") from error
    experiment = payload["experiment"]
    directory.mkdir(parents=True, exist_ok=True)
    config = {"telemetry_sha256": fingerprint(payload), "checkpoint_sha256": payload["checkpoint_sha256"]}
    if experiment:
        config.update({key: experiment[key] for key in ("protocol_sha256", "simulated", "scientific_evidence")})
        if "model_revision" in experiment:
            config.update(model_revision=experiment["model_revision"], receipt_sha256=experiment["receipt_sha256"])

    settings = wandb.Settings(console="off", disable_git=True, save_code=False, x_disable_meta=True,
                              x_disable_stats=True, x_disable_machine_info=True, x_save_requirements=False)
    with wandb.init(project="agentic-ai-lab", mode="offline", dir=str(directory), id=config["telemetry_sha256"][:20],
                    config=config, settings=settings) as run:
        metrics = {"workflow/events": len(payload["events"]), "workflow/model_calls": len(payload["agent_calls"])}
        attempts = payload.get("code_execution", [])
        if "code_execution" in payload:
            succeeded = sum(attempt["status"] == "success" for attempt in attempts)
            metrics.update({"code/attempts": len(attempts), "code/succeeded": succeeded,
                            "code/failed": len(attempts) - succeeded})
        if experiment:
            metrics.update(experiment["metrics"])
        run.log(metrics)
        for label, rows in (("workflow/timeline", payload["events"]), ("workflow/calls", payload["agent_calls"]),
                            ("code/executions", attempts),
                            ("gpu/trials", experiment["trials"] if experiment else [])):
            for row in rows:
                run.log({f"{label}/{key}": value for key, value in row.items() if value is not None})
        return str(run.dir)


def weave_plan(payload, project):
    if not re.fullmatch(r"[\w-]+/[\w-]+", project, flags=re.ASCII):
        raise ValueError("Indiquer le projet Weave sous la forme équipe/projet.")
    calls = [call for call in payload["agent_calls"] if call["latency_ms"] is not None]
    if not calls:
        raise ValueError("Aucun appel instrumenté et terminé à transférer vers Weave.")
    plan = {"prompt": "Autoriser le transfert de ces métadonnées vers Weave ?",
            "project": project, "calls": calls, "telemetry_sha256": fingerprint(payload)}
    return {**plan, "target_sha256": fingerprint(plan)}


def publish_weave(payload, project, *, approved_sha256):
    plan = weave_plan(payload, project)
    if approved_sha256 != plan["target_sha256"]:
        raise PermissionError("Une approbation du transfert exact vers Weave est nécessaire.")
    try:
        import weave
    except ImportError as error:
        raise RuntimeError("Installer requirements-weave.txt pour le transfert vers Weave.") from error

    # Weave environment overrides SDK settings. Refuse configuration that would capture extra data.
    settings = {"capture_code": False, "implicitly_patch_integrations": False,
                "capture_client_info": False, "capture_system_info": False}
    for key in settings:
        value = os.getenv("WEAVE_" + key.upper())
        if value is not None and value.casefold() not in {"0", "false", "no", "off"}:
            raise ValueError("Désactiver les options de capture automatique de Weave avant le transfert.")
    client = weave.init(project, settings=settings)
    try:
        for row in plan["calls"]:
            started = datetime.fromtimestamp(row["started_at_ns"] / 1_000_000_000, tz=timezone.utc)
            metadata = {key: row[key] for key in ("stage", "prompt_version", "instructions_sha256", "call_id")}
            call = client.create_call(f"agentic_ai_lab.{row['agent']}", inputs=metadata,
                                      use_stack=False, started_at=started)
            output = {key: row[key] for key in ("status", "stage_outcome", "latency_ms")}
            client.finish_call(call, output=output, ended_at=started + timedelta(milliseconds=row["latency_ms"]))
    finally:
        weave.finish()
    return {"project": project, "calls_submitted": len(plan["calls"]), "telemetry_sha256": plan["telemetry_sha256"]}
