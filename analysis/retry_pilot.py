"""Reconstruct a descriptive V9 paired pilot from its read-only event ledger."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
from typing import Literal

from pydantic import Field

from analysis.contracts import HEX_SHA, canonical_json, fingerprint
from execution.contracts import StrictModel
from experiments.retry_pilot import LookupTask, PilotCorpus, Product, Trial, evaluate_task_lookup
from experiments.retry_pilot_runner import (
    CALIBRATION, PilotJournal, _calibration_trials, _field_object, final_plan, standard_cost,
)

SOURCE_KIND = "verified_retry_pilot"
QUESTION = "Une relance après une erreur transitoire de lecture améliore-t-elle l'exactitude des réponses ?"
LIMITS = (
    "Le pilote couvre huit tâches de catalogue, une version du modèle et un seul essai par tâche et condition.",
    "L'erreur injectée disparaît à la deuxième tentative. Les pannes persistantes et les opérations avec effets de bord restent à étudier.",
    "L'analyse est descriptive et appariée par tâche. Elle ne fournit aucune inférence sur une population de tâches ou de modèles.",
)
PROCEDURE = (
    "Chaque tâche est répétée dans quatre conditions : deux politiques de relance et deux profils de l'outil.",
    "Le profil normal autorise la lecture immédiatement. Le profil transitoire injecte une erreur avant la première lecture valide.",
    "La politique avec relance rejoue une fois les mêmes arguments après cette erreur. Les deux politiques utilisent les mêmes consignes.",
    "La réponse finale est comparée à la valeur exacte du catalogue. Une abstention compte comme une réponse non exacte.",
)


class CalibrationFailed(ValueError):
    """A complete and verified calibration missed the protocol criteria."""


class TrialResult(Trial):
    score: Literal["correct", "incorrect", "abstention", "invalid_response"]
    lookup_attempts: list[Literal["ok", "transient_error", "invalid_lookup"]]
    generations: int
    input_tokens: int
    output_tokens: int
    event_sha256: str = Field(pattern=HEX_SHA)


class Arm(StrictModel):
    trial_ids: list[str]
    correct: int
    abstentions: int
    total: int
    accuracy_percent: float


class PairedComparison(StrictModel):
    arms: dict[Literal["no_retry", "retry_once"], Arm]
    both_correct: int
    retry_only: int
    no_retry_only: int
    neither_correct: int
    gain_points: float


class PairedRetryAnalysis(StrictModel):
    model: str
    task_count: int
    trial_count: int
    generation_count: int
    procedure: list[str]
    trials: list[TrialResult]
    comparison: dict[Literal["clean", "transient"], PairedComparison]
    limitations: list[str]
    scientific_evidence: Literal[False] = False
    ready_for_scientific_inference: Literal[False] = False

    def fingerprint(self):
        return fingerprint(self.model_dump(mode="json"))


class RetryPilotAnalysis(PairedRetryAnalysis):
    schema_version: Literal["9.0"] = "9.0"
    source_kind: Literal["verified_retry_pilot"] = SOURCE_KIND
    pilot_id: str
    question: str
    plan_sha256: str = Field(pattern=HEX_SHA)
    calibration_sha256: str = Field(pattern=HEX_SHA)
    measurement_sha256: str = Field(pattern=HEX_SHA)
    generation_cost_estimate_usd: str


def _target(plan, phase, calibration_sha=None):
    base = {"phase": phase, "plan_sha256": plan["plan_sha256"], "budget": plan["budget"][phase]}
    if calibration_sha is not None:
        base["calibration_sha256"] = calibration_sha
    return {**base, "target_sha256": fingerprint(base)}


def _call(steps, plan, turn, items, tools):
    profile = plan["model_profile"]
    request = {
        "model": profile["snapshot"], "input": items, "instructions": plan["model_instructions"], "tools": tools,
        "parallel_tool_calls": False, "text": plan["answer_format"], "reasoning": {"effort": profile["reasoning_effort"]},
        "truncation": "disabled",
    }
    if steps.pop(turn + "_count_reserved") != {"request": request}:
        raise ValueError("Recorded provider request differs from approved controls")
    count = steps.pop(turn + "_count_receipt")["input_tokens"]
    if steps.pop(turn + "_generation_reserved") != {
        "max_output_tokens": profile["max_output_tokens"], "service_tier": profile["service_tier"], "store": False,
    }:
        raise ValueError("Generation reservation differs from approved limits")
    receipt = steps.pop(turn + "_generation_receipt")
    for value, cap in ((count, profile["max_input_tokens"]), (receipt["input_tokens"], profile["max_input_tokens"]),
                       (receipt["output_tokens"], profile["max_output_tokens"])):
        if type(value) is not int or not 0 <= value <= cap:
            raise ValueError("Missing or out-of-bounds token accounting")
    if receipt["model"] != profile["snapshot"] or receipt["service_tier"] != profile["service_tier"]:
        raise ValueError("Receipt differs from approved model or tier")
    if not receipt["response_id"] or not (
        receipt["status"] == "completed" or (
            receipt["status"] == "incomplete" and receipt["incomplete_reason"] in {"max_output_tokens", "content_filter"}
        )
    ):
        raise ValueError("Unconfirmed provider result")
    return receipt


def _trial(plan, trial, task: LookupTask, product: Product, events):
    steps = {event["step"]: event["document"] for event in events}
    trial_doc = trial.model_dump(mode="json")
    if steps.pop("trial_reserved") != {"trial": trial_doc}:
        raise ValueError("Trial reservation differs from approved schedule")
    tool = plan["lookup_tool"]
    user = {"role": "user", "content": task.prompt}
    first = _call(steps, plan, "decision", [user], [{
        "type": "function", "name": tool["name"], "description": tool["description"],
        "parameters": tool["input_schema"], "strict": True,
    }])
    calls = [first]
    requested = [item for item in first["output"] if item["type"] == "function_call"]
    valid_lookup = False
    attempts = []
    if (first["status"] == "completed" and len(requested) == 1 and requested[0]["name"] == tool["name"]
            and all(item["type"] in {"message", "reasoning", "function_call"} for item in first["output"])):
        call = requested[0]
        lookup = evaluate_task_lookup(task, product, trial, _field_object(call["arguments"], "product"))
        if steps.pop("lookup") != lookup.model_dump(mode="json"):
            raise ValueError("Lookup receipt differs from recomputed policy")
        attempts = lookup.attempts
        valid_lookup = attempts != ["invalid_lookup"]
        output = {"type": "function_call_output", "call_id": call["call_id"],
                  "output": canonical_json({"status": attempts[-1], "value": lookup.output})}
        function = {key: call[key] for key in ("type", "call_id", "name", "arguments")}
        calls.append(_call(steps, plan, "answer", [user, function, output], []))
    final = calls[-1]
    answer = _field_object(final["text"], "answer")
    score = "invalid_response"
    if (final["status"] == "completed" and answer.keys() == {"answer"}
            and all(item["type"] in {"message", "reasoning"} for item in final["output"])):
        value = answer["answer"]
        if value is None:
            score = "abstention"
        elif isinstance(value, str):
            score = "correct" if value == getattr(product, task.field) else "incorrect"
    expected = {"plan_sha256": plan["plan_sha256"], "trial": trial_doc,
                "lookup_arguments_valid": valid_lookup, "answer_score": score}
    if steps.pop("receipt") != expected or steps:
        raise ValueError("Trial outcome differs from raw replies or has unexpected events")
    return TrialResult(**trial_doc, score=score, lookup_attempts=attempts, generations=len(calls),
                       input_tokens=sum(call["input_tokens"] for call in calls),
                       output_tokens=sum(call["output_tokens"] for call in calls), event_sha256=fingerprint(events))


def _phase(plan, phase, events, trials, tasks, calibration_sha=None):
    target = _target(plan, phase, calibration_sha)
    if len(events) < 2 or events[0]["step"] != "approved" or events[0]["trial_id"]:
        raise ValueError("Phase has no recorded approval")
    approval = events[0]["document"]
    actor = approval["approval"].get("actor")
    if (not isinstance(actor, str) or not actor.strip() or approval["target"] != target
            or approval["approval"] != {"actor": actor, "phase": phase, "target_sha256": target["target_sha256"]}):
        raise ValueError("Approval does not bind this plan and phase")
    if events[-1] != {"phase": phase, "trial_id": "", "step": "completed",
                      "document": {"completed_trials": len(trials)}}:
        raise ValueError("Phase is incomplete")
    results = []
    position = 1
    for trial in trials:
        start = position
        while position < len(events) - 1 and events[position]["trial_id"] == trial.trial_id:
            position += 1
        sequence = events[start:position]
        names = [event["step"] for event in sequence]
        decision = ["trial_reserved", "decision_count_reserved", "decision_count_receipt",
                    "decision_generation_reserved", "decision_generation_receipt"]
        followup = ["lookup", "answer_count_reserved", "answer_count_receipt",
                    "answer_generation_reserved", "answer_generation_receipt"]
        if names not in (decision + ["receipt"], decision + followup + ["receipt"]):
            raise ValueError("Incomplete or out-of-order trial events")
        results.append(_trial(plan, trial, *tasks[trial.task_id], sequence))
    if position != len(events) - 1 or sum(row.generations for row in results) > plan["budget"][phase]["max_generations"]:
        raise ValueError("Phase has extra events or exceeds its generation limit")
    return results


def _journal_phases(corpus, journal):
    plan = final_plan(corpus)
    events = journal.events()
    if not events or events[0] != {"phase": "plan", "trial_id": "", "step": "plan", "document": plan}:
        raise ValueError("Journal plan differs from the verified corpus or runner")
    calibration = [event for event in events if event["phase"] == "calibration"]
    measurement = [event for event in events if event["phase"] == "measurement"]
    if events != [events[0], *calibration, *measurement]:
        raise ValueError("Journal phases are interleaved or unknown")
    return plan, calibration, measurement


def _verify_calibration(plan, calibration):
    checked = _phase(plan, "calibration", calibration, _calibration_trials(),
                     {task.task_id: (task, product) for task, product, _ in CALIBRATION})
    for row in checked:
        expected = "abstention" if row.profile == "transient" and row.policy == "no_retry" else "correct"
        if row.score != expected or row.lookup_attempts in ([], ["invalid_lookup"]):
            raise CalibrationFailed("Calibration gate did not pass")
    return fingerprint(calibration)


def verify_retry_calibration(corpus: PilotCorpus, journal: PilotJournal) -> str:
    """Recheck the full recorded calibration before authorizing measurement."""
    plan, calibration, _ = _journal_phases(corpus, journal)
    return _verify_calibration(plan, calibration)


def paired_comparison(corpus: PilotCorpus, results: list[TrialResult]) -> dict[str, PairedComparison]:
    task_count = len(corpus.tasks)
    comparison = {}
    for profile in ("clean", "transient"):
        subset = [row for row in results if row.profile == profile]
        arms = {}
        for policy in ("no_retry", "retry_once"):
            rows = [row for row in subset if row.policy == policy]
            correct = sum(row.score == "correct" for row in rows)
            arms[policy] = Arm(trial_ids=[row.trial_id for row in rows], correct=correct, total=len(rows),
                               abstentions=sum(row.score == "abstention" for row in rows),
                               accuracy_percent=100.0 * correct / len(rows))
        by_task = {(row.task_id, row.policy): row.score == "correct" for row in subset}
        paired = Counter((by_task[task.task_id, "no_retry"], by_task[task.task_id, "retry_once"]) for task in corpus.tasks)
        comparison[profile] = PairedComparison(arms=arms, both_correct=paired[True, True], retry_only=paired[False, True],
                                                no_retry_only=paired[True, False], neither_correct=paired[False, False],
                                                gain_points=100.0 * (paired[False, True] - paired[True, False]) / task_count)
    return comparison


def analyze_retry_pilot(corpus: PilotCorpus, journal: PilotJournal) -> RetryPilotAnalysis:
    plan, calibration, measurement = _journal_phases(corpus, journal)
    calibration_sha = _verify_calibration(plan, calibration)
    results = _phase(plan, "measurement", measurement, [Trial.model_validate(row) for row in plan["schedule"]["trials"]],
                     {task.task_id: (task, corpus.catalog[task.product]) for task in corpus.tasks}, calibration_sha)
    cost = sum((standard_cost(row.input_tokens, row.output_tokens) for row in results), Decimal(0))
    return RetryPilotAnalysis(pilot_id=corpus.pilot_id, question=QUESTION, model=plan["model_profile"]["snapshot"],
                              plan_sha256=plan["plan_sha256"], calibration_sha256=calibration_sha,
                              measurement_sha256=fingerprint(measurement), task_count=len(corpus.tasks), trial_count=len(results),
                              generation_count=sum(row.generations for row in results),
                              generation_cost_estimate_usd=str(cost), procedure=list(PROCEDURE), trials=results,
                              comparison=paired_comparison(corpus, results),
                              limitations=list(LIMITS))
