"""Verify model receipts, recompute tool-grounded scores and compare shifts."""
from typing import Literal

from analysis.executable_protocol import ProtocolAnalysis, ScoredPrediction, _checked_receipt, summarize_cells
from execution.agentic_gpu import (
    ARTIFACT, BACKEND, MODELS, POLICIES, PROFILES, TASKS, AgenticGPUBackend, AgenticRuntime, validate_receipt,
)
from execution.agentic_gpu_worker import object_or_none, tool_result
from execution.contracts import StrictModel
from research.agentic_study import compile_study

SOURCE_KIND = "verified_agentic_gpu"


class AgenticTrial(ScoredPrediction):
    outcome: Literal["answered", "abstained", "invalid_output"]
    model_key: str
    tool_attempts: list[Literal["invalid_lookup", "transient_error", "ok"]]


class ShiftGap(StrictModel):
    drop_points: float
    task_sensitivity_min: float
    task_sensitivity_max: float
    distinct_tasks: int


class AgenticAnalysis(ProtocolAnalysis):
    source_kind: Literal["verified_agentic_gpu"] = SOURCE_KIND
    trials: list[AgenticTrial]
    models: dict[str, dict]
    runtimes: dict[str, AgenticRuntime]
    shift_gaps: dict[str, ShiftGap]


def analyze_agentic(source, manager):
    if not manager.read_only:
        raise PermissionError("Agentic analysis requires the original read-only ledger")
    design, jobs, plan = compile_study(source)
    rows, runtimes = [], {}
    for job in jobs:
        receipt, artifacts_sha = _checked_receipt(manager, job, AgenticGPUBackend, BACKEND, ARTIFACT)
        runtime, responses = validate_receipt(job, plan, receipt)
        key = job.workload.parameters["model_key"]
        runtimes[key] = runtime
        for trial, output in zip(job.workload.parameters["trials"], responses, strict=True):
            _, attempts = tool_result(trial, output.action.text)
            parsed = object_or_none(output.answer.text)
            structured = parsed is not None and set(parsed) == {"answer"}
            answer = parsed["answer"] if structured else None
            valid = type(answer) is int and abs(answer) <= 10**9
            outcome = "answered" if valid else "abstained" if structured and answer is None else "invalid_output"
            rows.append(AgenticTrial(
                cell_id=trial["cell"], case_id=trial["case"], seed=trial["seed"], model_key=key,
                tool_attempts=attempts, correct=valid and "ok" in attempts and answer == TASKS[trial["case"]]["stock"],
                value=answer if valid else None, outcome=outcome,
                elapsed_ns=output.action.elapsed_ns + output.answer.elapsed_ns,
                model_tokens=sum(turn.input_tokens + turn.output_tokens for turn in (output.action, output.answer)),
                job_sha256=job.fingerprint(), artifacts_sha256=artifacts_sha,
            ))
    cells, contrasts = summarize_cells(design, rows)
    paired = {(row.model_key, cells[row.cell_id].levels["policy"], cells[row.cell_id].levels["shift"],
               row.case_id, row.seed): row.correct for row in rows}
    gaps = {}
    for key in MODELS:
        for policy in POLICIES:
            for shift in PROFILES[1:]:
                task_means = []
                for case in TASKS:
                    diffs = [int(paired[key, policy, "clean", case, seed]) - int(paired[key, policy, shift, case, seed])
                             for seed in design.resource_limits.seeds]
                    task_means.append(100 * sum(diffs) / len(diffs))
                total = sum(task_means)
                loo = [(total - value) / (len(task_means) - 1) for value in task_means]
                gaps[f"{key}-{policy}-{shift}"] = ShiftGap(
                    drop_points=total / len(task_means), task_sensitivity_min=min(loo),
                    task_sensitivity_max=max(loo), distinct_tasks=len(task_means),
                )
    return AgenticAnalysis(
        protocol_sha256=plan["protocol_sha256"], action_plan_sha256=plan["plan_sha256"],
        evaluation_sha256=plan["evaluation_sha256"], cells=cells, contrasts=contrasts, trials=rows,
        models={key: {k: v for k, v in model.items() if k != "snapshot"} for key, model in plan["models"].items()},
        runtimes=runtimes, shift_gaps=gaps,
        limitations=[
            "Trois tâches fictives de consultation d’un même outil en lecture seule.",
            "Les checkpoints diffèrent par l’entraînement, la taille et parfois l’architecture. Les contrastes sont descriptifs.",
            "La génération gloutonne est répétée dans deux ordres. Les graines ne multiplient pas les tâches indépendantes.",
            "Les bornes omettent successivement une tâche et décrivent la sensibilité du pilote sur trois tâches.",
        ],
    )
