"""Build the V9 pilot schedule and exercise its read-only tool policy."""
from __future__ import annotations

import random
from typing import Literal

from pydantic import Field, model_validator

from analysis.contracts import fingerprint
from execution.contracts import StrictModel

Policy = Literal["no_retry", "retry_once"]
Profile = Literal["clean", "transient"]
FieldName = Literal["stock", "zone"]


class Product(StrictModel):
    stock: str = Field(pattern=r"^[0-9]{1,3}$")
    zone: str = Field(pattern=r"^[A-Z]-[0-9]{2}$")


class LookupTask(StrictModel):
    task_id: str = Field(pattern=r"^T[0-9]{2}$")
    prompt: str = Field(min_length=1, max_length=300)
    product: str = Field(pattern=r"^[A-Z]{2}-[0-9]{3}$")
    field: FieldName


class PilotCorpus(StrictModel):
    pilot_id: str = Field(pattern=r"^[a-z0-9-]{1,48}$")
    order_seed: int = Field(ge=0, le=2**32 - 1)
    catalog: dict[str, Product] = Field(min_length=8, max_length=8)
    tasks: list[LookupTask] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def verify_task_coverage(self):
        if len({task.task_id for task in self.tasks}) != 8:
            raise ValueError("Each task must have a distinct identifier")
        if {task.product for task in self.tasks} != self.catalog.keys():
            raise ValueError("Each task must address a distinct catalog product")
        if sum(task.field == "stock" for task in self.tasks) != 4:
            raise ValueError("The task corpus must balance stock and zone lookups")
        return self


class Trial(StrictModel):
    trial_id: str
    task_id: str
    profile: Profile
    policy: Policy


def prepare_plan(corpus: PilotCorpus) -> dict:
    """Return a reproducible, blinded schedule with no provider or job calls."""
    tasks = sorted(corpus.tasks, key=lambda task: task.task_id)
    random.Random(corpus.order_seed).shuffle(tasks)
    trials = []
    for index, task in enumerate(tasks):
        profiles = ("clean", "transient") if index % 2 == 0 else ("transient", "clean")
        for position, profile in enumerate(profiles):
            policies = ("no_retry", "retry_once") if (index // 2 + position) % 2 == 0 else ("retry_once", "no_retry")
            for policy in policies:
                trial = Trial(trial_id=f"{task.task_id}-{profile}-{policy}", task_id=task.task_id,
                              profile=profile, policy=policy)
                trials.append(trial.model_dump(mode="json"))
    plan = {
        "schema_version": "9.0-pilot-preview",
        "pilot_id": corpus.pilot_id,
        "corpus_sha256": fingerprint(corpus.model_dump(mode="json")),
        "order_seed": corpus.order_seed,
        "trials": trials,
        "planned_trials": len(trials),
        "max_model_calls": 2 * len(trials),
        "max_physical_lookups_per_trial": 2,
        "model_profile": None,
        "execution_authorized": False,
    }
    return {**plan, "plan_sha256": fingerprint(plan)}


class LookupReceipt(StrictModel):
    output: str | None = None
    attempts: list[Literal["ok", "transient_error", "invalid_lookup"]]


def evaluate_lookup(corpus: PilotCorpus, trial: Trial, arguments: dict) -> LookupReceipt:
    """Apply one policy to a valid, task-scoped, idempotent catalog lookup."""
    task = next((task for task in corpus.tasks if task.task_id == trial.task_id), None)
    if task is None:
        raise ValueError("Trial references an unknown task")
    return evaluate_task_lookup(task, corpus.catalog[task.product], trial, arguments)


def evaluate_task_lookup(task: LookupTask, product: Product, trial: Trial, arguments: dict) -> LookupReceipt:
    if arguments != {"product": task.product, "field": task.field}:
        return LookupReceipt(attempts=["invalid_lookup"])
    attempts = []
    if trial.profile == "transient":
        attempts.append("transient_error")
        if trial.policy == "no_retry":
            return LookupReceipt(attempts=attempts)
    output = getattr(product, task.field)
    return LookupReceipt(output=output, attempts=[*attempts, "ok"])
