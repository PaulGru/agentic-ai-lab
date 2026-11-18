"""Run a bounded V9 pilot against a pinned API snapshot after exact phase approval."""
from __future__ import annotations

from contextlib import closing
from dataclasses import asdict
from decimal import Decimal
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sqlite3
from time import perf_counter_ns
from typing import Literal

from pydantic import Field

from agent.models import ToolDefinition
from analysis.contracts import HEX_SHA, canonical_json, fingerprint
from execution.contracts import StrictModel
from experiments.retry_pilot import LookupTask, PilotCorpus, Product, Trial, evaluate_task_lookup, prepare_plan
from providers.openai_provider import OpenAIProvider


MODEL = "gpt-5.4-mini-2026-03-17"
INPUT_CAP = 1024
OUTPUT_CAP = 256
TIMEOUT = 30
PRICES = {"input_usd_per_million": "0.75", "output_usd_per_million": "4.50",
          "source": "https://developers.openai.com/api/docs/models/gpt-5.4-mini"}
INSTRUCTIONS = (
    "Consulte l'outil de catalogue une fois pour répondre à la demande. "
    "Utilise uniquement le champ et la référence demandés. "
    "Si la lecture n'aboutit pas, abstiens-toi. Réponds seulement en JSON "
    'avec une clé answer et comme valeur la chaîne exacte du catalogue ou null.'
)
LOOKUP = ToolDefinition(
    name="read_catalog",
    description="Lis un champ pour une référence produit dans le catalogue.",
    input_schema={
        "type": "object",
        "properties": {"product": {"type": "string"}, "field": {"type": "string", "enum": ["stock", "zone"]}},
        "required": ["product", "field"],
        "additionalProperties": False,
    },
)
ANSWER_FORMAT = {
    "format": {"type": "json_schema", "name": "catalog_answer", "strict": True,
               "schema": {"type": "object", "properties": {"answer": {"type": ["string", "null"]}},
                          "required": ["answer"], "additionalProperties": False}}
}
CALIBRATION = (
    (LookupTask(task_id="T91", prompt="Combien d'unités de JR-914 sont disponibles ?", product="JR-914", field="stock"),
     Product(stock="28", zone="A-10"), "clean"),
    (LookupTask(task_id="T92", prompt="Dans quelle zone se trouve KS-925 ?", product="KS-925", field="zone"),
     Product(stock="49", zone="C-16"), "transient"),
)


class PhaseApproval(StrictModel):
    phase: Literal["calibration", "measurement"]
    target_sha256: str = Field(pattern=HEX_SHA)
    actor: str = Field(min_length=1, max_length=100)


def standard_cost(input_tokens: int, output_tokens: int) -> Decimal:
    return (input_tokens * Decimal(PRICES["input_usd_per_million"])
            + output_tokens * Decimal(PRICES["output_usd_per_million"])) / 1_000_000


def _calibration_trials() -> list[Trial]:
    return [Trial(trial_id=f"{task.task_id}-{profile}-{policy}", task_id=task.task_id,
                  profile=profile, policy=policy)
            for task, _, profile in CALIBRATION for policy in ("no_retry", "retry_once")]


def _budget(trials):
    calls = 2 * trials
    return {"trials": trials, "max_generations": calls, "max_token_count_requests": calls,
            "max_input_tokens": calls * INPUT_CAP, "max_output_tokens": calls * OUTPUT_CAP,
            "standard_generation_budget_usd": str(standard_cost(calls * INPUT_CAP, calls * OUTPUT_CAP))}


def final_plan(corpus: PilotCorpus) -> dict:
    schedule = prepare_plan(corpus)
    calibration = [{"task": task.model_dump(mode="json"), "product": product.model_dump(mode="json"),
                    "profile": profile} for task, product, profile in CALIBRATION]
    if corpus.catalog.keys() & {task.product for task, _, _ in CALIBRATION} or (
            {task.task_id for task in corpus.tasks} & {task.task_id for task, _, _ in CALIBRATION}):
        raise ValueError("Calibration and measurement must use separate tasks and products")
    calibration_trials = _calibration_trials()
    base = {
        "schema_version": "9.0-pilot-1",
        "pilot_id": corpus.pilot_id,
        "schedule": schedule,
        "measurement_prompts": {task.task_id: task.prompt for task in corpus.tasks},
        "model_profile": {"provider": "openai", "api": "responses", "snapshot": MODEL,
                          "service_tier": "default", "reasoning_effort": "none",
                          "max_input_tokens": INPUT_CAP, "max_output_tokens": OUTPUT_CAP,
                          "timeout_seconds": TIMEOUT, "automatic_retries": 0,
                          "openai_sdk_version": version("openai")},
        "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "lookup_source_sha256": hashlib.sha256(Path(__file__).with_name("retry_pilot.py").read_bytes()).hexdigest(),
        "model_instructions": INSTRUCTIONS, "lookup_tool": asdict(LOOKUP), "answer_format": ANSWER_FORMAT,
        "calibration_cases": calibration,
        "calibration_trials": [trial.model_dump(mode="json") for trial in calibration_trials],
        "calibration_gate": "valid lookup and expected exact answer or expected abstention in each trial",
        "budget": {"standard_rates": PRICES, "calibration": _budget(len(calibration_trials)),
                   "measurement": _budget(schedule["planned_trials"]),
                   "total": _budget(len(calibration_trials) + schedule["planned_trials"])},
    }
    return {**base, "plan_sha256": fingerprint(base)}


class PilotJournal:
    """Append-only SQLite events with first-writer ownership of each phase."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _connect(self, read_only=False):
        if read_only:
            return sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        return sqlite3.connect(self.path)

    @staticmethod
    def _write(db, phase, trial_id, step, value):
        db.execute("INSERT INTO pilot_events (phase, trial_id, step, document, sha256) VALUES (?, ?, ?, ?, ?)",
                   (phase, trial_id, step, canonical_json(value), fingerprint(value)))

    def begin(self, plan, approval, target):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS pilot_events (
                seq INTEGER PRIMARY KEY, phase TEXT NOT NULL, trial_id TEXT NOT NULL, step TEXT NOT NULL,
                document TEXT NOT NULL, sha256 TEXT NOT NULL, UNIQUE(phase, trial_id, step)
            )""")
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT sha256 FROM pilot_events WHERE phase='plan'").fetchone()
            if row is None:
                self._write(db, "plan", "", "plan", plan)
            elif row[0] != fingerprint(plan):
                raise ValueError("Journal belongs to a different pilot plan")
            try:
                self._write(db, approval.phase, "", "approved", {
                    "approval": approval.model_dump(mode="json"), "target": target,
                })
            except sqlite3.IntegrityError as error:
                raise PermissionError("Phase already started. Inspect the journal; no automatic replay.") from error

    def append(self, phase, trial_id, step, value):
        with closing(self._connect()) as db, db:
            self._write(db, phase, trial_id, step, value)

    def events(self):
        if not self.path.exists():
            return []
        with closing(self._connect(read_only=True)) as db:
            rows = db.execute("SELECT phase, trial_id, step, document, sha256 FROM pilot_events ORDER BY seq").fetchall()
        records = []
        for phase, trial_id, step, raw, digest in rows:
            document = json.loads(raw)
            if fingerprint(document) != digest:
                raise ValueError("Pilot journal event digest mismatch")
            records.append({"phase": phase, "trial_id": trial_id, "step": step, "document": document})
        return records


def _calibration_passed(plan, events):
    expected = {trial.trial_id: (trial.policy == "no_retry" and trial.profile == "transient")
                for trial in _calibration_trials()}
    saved = {event["trial_id"]: event["document"] for event in events
             if event["phase"] == "calibration" and event["step"] == "receipt"}
    if len(saved) != len(expected) or set(saved) != set(expected):
        return False
    completed = [event["document"] for event in events
                 if event["phase"] == "calibration" and event["step"] == "completed"]
    return completed == [{"completed_trials": len(expected)}] and all(
        receipt["plan_sha256"] == plan["plan_sha256"]
        and receipt["lookup_arguments_valid"]
        and receipt["answer_score"] == ("abstention" if expected[trial_id] else "correct")
        for trial_id, receipt in saved.items()
    )


def approval_target(plan, journal, phase):
    if phase not in {"calibration", "measurement"}:
        raise ValueError("Unknown pilot phase")
    events = journal.events()
    if events and (events[0]["phase"] != "plan" or events[0]["document"] != plan):
        raise ValueError("Journal plan differs from current preview")
    if any(event["phase"] == phase and event["step"] == "approved" for event in events):
        raise PermissionError("Phase already started. Inspect the journal; no automatic replay.")
    target = {"phase": phase, "plan_sha256": plan["plan_sha256"], "budget": plan["budget"][phase]}
    if phase == "measurement":
        if not _calibration_passed(plan, events):
            raise PermissionError("Measurement requires a passed calibration in this journal")
        target["calibration_sha256"] = fingerprint([event for event in events if event["phase"] == "calibration"])
    return {**target, "target_sha256": fingerprint(target)}


def _client():
    from openai import OpenAI
    return OpenAI(base_url="https://api.openai.com/v1", timeout=TIMEOUT, max_retries=0)


def _nonnegative_int(value, maximum):
    return type(value) is int and 0 <= value <= maximum


def _call(client, append, turn, items, tools):
    request = {"model": MODEL, "input": items, "instructions": INSTRUCTIONS, "tools": tools,
               "parallel_tool_calls": False, "text": ANSWER_FORMAT, "reasoning": {"effort": "none"},
               "truncation": "disabled"}
    append(f"{turn}_count_reserved", {"request": request})
    count = client.responses.input_tokens.count(**request).input_tokens
    append(f"{turn}_count_receipt", {"input_tokens": count})
    if not _nonnegative_int(count, INPUT_CAP):
        raise ValueError("Input estimate exceeds the approved token limit")
    append(f"{turn}_generation_reserved", {"max_output_tokens": OUTPUT_CAP, "service_tier": "default", "store": False})
    started = perf_counter_ns()
    response = client.responses.create(**request, max_output_tokens=OUTPUT_CAP, store=False, service_tier="default")
    usage = response.usage
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    parts = [part.model_dump(mode="json") for part in response.output]
    receipt = {
        "response_id": response.id, "model": response.model, "status": response.status,
        "service_tier": response.service_tier, "input_tokens": input_tokens, "output_tokens": output_tokens,
        "latency_ms": (perf_counter_ns() - started) // 1_000_000, "output": parts, "text": response.output_text,
        "incomplete_reason": getattr(getattr(response, "incomplete_details", None), "reason", None),
    }
    append(f"{turn}_generation_receipt", receipt)
    if (response.model != MODEL or response.service_tier != "default"
            or not _nonnegative_int(input_tokens, INPUT_CAP) or not _nonnegative_int(output_tokens, OUTPUT_CAP)):
        raise ValueError("Provider response exceeded limits or differs from the approved configuration")
    if response.status != "completed" and not (
            response.status == "incomplete" and receipt["incomplete_reason"] in {"max_output_tokens", "content_filter"}):
        raise RuntimeError("Unexpected provider response status")
    return receipt


def _field_object(text, key):
    """Read a single top-level JSON object and reject duplicate field names."""
    def object_pairs(pairs):
        if len({k for k, _ in pairs}) != len(pairs):
            raise ValueError("Duplicate JSON key")
        return dict(pairs)

    try:
        value = json.loads(text, object_pairs_hook=object_pairs)
    except (ValueError, TypeError, RecursionError):
        return {}
    return value if isinstance(value, dict) and key in value else {}


def _trial(client, append, plan, trial, task, product):
    user = {"role": "user", "content": task.prompt}
    first = _call(client, append, "decision", [user], [{
        **OpenAIProvider._convert_tool(LOOKUP), "strict": True,
    }])
    requested_tools = [item for item in first["output"] if item["type"] == "function_call"]
    lookup = None
    final = first
    if (first["status"] == "completed" and len(requested_tools) == 1
            and requested_tools[0]["name"] == LOOKUP.name
            and all(item["type"] in {"message", "reasoning", "function_call"} for item in first["output"])):
        call = requested_tools[0]
        args = _field_object(call["arguments"], "product")
        lookup = evaluate_task_lookup(task, product, trial, args)
        append("lookup", lookup.model_dump(mode="json"))
        tool_result = {"type": "function_call_output", "call_id": call["call_id"], "output": canonical_json({
            "status": lookup.attempts[-1], "value": lookup.output,
        })}
        tool_call = {name: call[name] for name in ("type", "call_id", "name", "arguments")}
        final = _call(client, append, "answer", [user, tool_call, tool_result], [])
    answer = _field_object(final["text"], "answer")
    score = "invalid_response"
    final_parts_valid = all(item["type"] in {"message", "reasoning"} for item in final["output"])
    if final["status"] == "completed" and final_parts_valid and answer.keys() == {"answer"}:
        value = answer["answer"]
        if value is None:
            score = "abstention"
        elif isinstance(value, str):
            score = "correct" if value == getattr(product, task.field) else "incorrect"
    append("receipt", {"plan_sha256": plan["plan_sha256"], "trial": trial.model_dump(mode="json"),
                       "lookup_arguments_valid": bool(lookup and lookup.attempts != ["invalid_lookup"]),
                       "answer_score": score})


def run_phase(corpus, journal, approval, *, client_factory=_client):
    approval = PhaseApproval.model_validate(approval.model_dump(mode="json"))
    plan = final_plan(corpus)
    target = approval_target(plan, journal, approval.phase)
    if approval.target_sha256 != target["target_sha256"] or not approval.actor.strip():
        raise PermissionError("Approval must target the exact phase and expenditure")
    if approval.phase == "calibration":
        tasks = {task.task_id: (task, product) for task, product, _ in CALIBRATION}
        trials = _calibration_trials()
    else:
        tasks = {task.task_id: (task, corpus.catalog[task.product]) for task in corpus.tasks}
        trials = [Trial.model_validate(item) for item in plan["schedule"]["trials"]]
    journal.begin(plan, approval, target)
    with client_factory() as client:
        for trial in trials:
            def append(step, value):
                journal.append(approval.phase, trial.trial_id, step, value)

            append("trial_reserved", {"trial": trial.model_dump(mode="json")})
            try:
                _trial(client, append, plan, trial, *tasks[trial.task_id])
            except Exception as error:
                append("stopped", {"reason": type(error).__name__})
                raise RuntimeError("Pilot phase stopped. Inspect the journal before any new authorization.") from None
    journal.append(approval.phase, "", "completed", {"completed_trials": len(trials)})


def summarize(plan, journal):
    events = journal.events()
    if not events or events[0]["phase"] != "plan" or events[0]["document"] != plan:
        raise ValueError("Missing journal or plan differs from preview")
    summary = {"plan_sha256": plan["plan_sha256"], "calibration_passed": _calibration_passed(plan, events)}
    for phase in ("calibration", "measurement"):
        selected = [event for event in events if event["phase"] == phase]
        receipts = [event["document"] for event in selected if event["step"] == "receipt"]
        received = [event["document"] for event in selected if event["step"].endswith("_generation_receipt")]
        reserved = sum(event["step"].endswith("_generation_reserved") for event in selected)
        usage_available = all(type(response["input_tokens"]) is int and type(response["output_tokens"]) is int
                              for response in received)
        cost = sum((standard_cost(response["input_tokens"], response["output_tokens"]) for response in received
                    if usage_available), Decimal(0))
        summary[phase] = {
            "state": ("complete" if any(event["step"] == "completed" for event in selected)
                      else "incomplete" if selected else "not_started"),
            "trial_receipts": receipts, "generations_reserved": reserved, "generations_received": len(received),
            "known_generation_cost_estimate_usd": str(cost) if usage_available else None,
            "all_generation_usage_accounted": usage_available and len(received) == reserved,
        }
    return summary
