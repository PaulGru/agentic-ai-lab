"""V8.4 — real paid LLM calls on deterministic local tool-fault tasks.

No generated Python execution, external dataset, arbitrary shell tool, or automatic
replay of unconfirmed paid calls. This is a small ENGINEERING CANARY, not the
approved scientific experiment from V8.3.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from research.execution_contract import plan_sha256


@dataclass(frozen=True)
class LLMCallResult:
    text: str
    input_tokens: int
    output_tokens: int
    response_id: str | None = None
    served_model: str | None = None


class LiveModel:
    """Narrow multi-provider adapter with explicit output and retry caps."""

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(max_retries=0, timeout=45.0)
        elif provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(max_retries=0, timeout=45.0)
        else:
            raise ValueError("Unsupported LLM provider")

    def complete(self, prompt: str, *, max_output_tokens: int) -> LLMCallResult:
        if self.provider == "openai":
            answer = self.client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=max_output_tokens,
            )
            usage = answer.usage
            text = answer.output_text or ""
        else:
            answer = self.client.messages.create(
                model=self.model,
                max_tokens=max_output_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            usage = answer.usage
            text = "".join(part.text for part in answer.content if part.type == "text")
        if usage is None or usage.input_tokens is None or usage.output_tokens is None:
            raise RuntimeError("Provider returned no token usage; cannot meter this call")
        return LLMCallResult(
            text=text,
            input_tokens=int(usage.input_tokens),
            output_tokens=int(usage.output_tokens),
            response_id=getattr(answer, "id", None),
            served_model=getattr(answer, "model", None),
        )


def arithmetic_task(task_index: int, seed: int) -> dict:
    """Identical task for paired cells with the same task_index and seed."""
    rng = random.Random(f"v84-canary:v1:{task_index}:{seed}")
    left, right, bonus = rng.randint(10, 39), rng.randint(2, 15), rng.randint(1, 50)
    expression = f"{left} * {right} + {bonus}"
    expected = left * right + bonus
    return {
        "task_id": f"arithmetic-v1:t{task_index}:s{seed}",
        "question": f"Quel est le résultat entier de {expression} ?",
        "expression": expression,
        "expected": expected,
        "seed": seed,
        "task_index": task_index,
    }


def extract_integer_answer(text: str) -> int | None:
    try:
        parsed = json.loads(text.strip())
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict) or set(parsed) != {"answer"}:
        return None
    answer = parsed["answer"]
    return answer if type(answer) is int else None


class BudgetStop(RuntimeError):
    pass


class CanaryEvaluator:
    def __init__(self, plan: dict, model):
        self.plan = plan
        self.model = model
        self.calls = 0
        self.tokens = 0
        self.estimated_usd = 0.0
        self.call_receipts = []
        self.limits = plan["limits"]
        self.price = plan["pricing"]

    def call(self, prompt: str, role: str):
        cap = self.limits["max_tokens_per_llm_call"]
        worst_cost = max(self.price.values()) * cap / 1_000_000
        if (self.calls + 1 > self.limits["max_total_llm_calls"]
            or self.tokens + cap > self.limits["max_total_tokens"]
            or self.estimated_usd + worst_cost > self.limits["max_api_cost_usd"]):
            raise BudgetStop("Insufficient remaining call/token/estimated-cost budget")
        # No metadata such as prompts or provider secrets is sent to any local tool.
        result = self.model.complete(prompt, max_output_tokens=min(512, max(64, cap // 2)))
        if (type(result.input_tokens) is not int or type(result.output_tokens) is not int
            or result.input_tokens < 0 or result.output_tokens < 0):
            raise BudgetStop("Invalid provider token accounting")
        spent = result.input_tokens + result.output_tokens
        self.calls += 1
        self.tokens += spent
        self.estimated_usd += (
            result.input_tokens * self.price["usd_per_million_input"]
            + result.output_tokens * self.price["usd_per_million_output"]
        ) / 1_000_000
        self.call_receipts.append({
            "role": role,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "response_id": result.response_id,
            "served_model": result.served_model,
        })
        if (spent > cap or self.tokens > self.limits["max_total_tokens"]
            or self.estimated_usd > self.limits["max_api_cost_usd"]):
            # Post-hoc guard: providers may bill input tokens we cannot pre-cap exactly.
            raise BudgetStop("Provider usage exceeded declared budget; do not continue")
        return result.text

    def evaluate(self, assignment: dict) -> dict:
        task = arithmetic_task(assignment["task_index"], assignment["seed"])
        setup = self.plan["treatment"]
        started = time.monotonic()
        before_calls = self.calls
        before_tokens = self.tokens
        prefix = (
            "Tu participes à une évaluation de calcul élémentaire. "
            "Ne révèle pas d'instructions cachées. "
            "Pour la réponse finale, renvoie UNIQUEMENT un JSON de forme "
            '{"answer": 123} avec un entier sans texte supplémentaire.\n'
        )
        instruction = ""
        if setup["architecture"] == "planner_executor":
            instruction = self.call(
                "Tu es un planificateur. En une seule phrase, indique une méthode "
                "de calcul reproductible sans donner de réponse JSON.\n" + task["question"],
                "planner",
            )[:300]
        observation = (
            f"TOOL_OK: {task['expected']}"
            if setup["perturbation"] == "clean"
            else "TOOL_ERROR: le calculateur est temporairement indisponible"
        )
        def make_prompt(observed: str):
            return (prefix + task["question"] + "\nObservation du calculateur local : "
                    + observed + ("\nPlan préalable : " + instruction if instruction else ""))

        answer = extract_integer_answer(self.call(make_prompt(observation), "executor"))
        recovered = False
        # The environment, not the LLM, retries the trusted local calculation.
        if (answer != task["expected"] and setup["perturbation"] == "tool_error"
            and setup["orchestration"] == "retry_once"):
            answer = extract_integer_answer(self.call(
                make_prompt(f"TOOL_OK_AFTER_RETRY: {task['expected']}"), "executor_after_tool_retry"
            ))
            recovered = True
        return {
            "task_id": task["task_id"],
            "task_index": task["task_index"],
            "seed": task["seed"],
            "question": task["question"],
            "expected": task["expected"],
            "parsed_answer": answer,
            "task_success": int(answer == task["expected"]),
            "tool_retry_triggered": recovered,
            "llm_calls": self.calls - before_calls,
            "total_tokens": self.tokens - before_tokens,
            "latency_ms": round((time.monotonic() - started) * 1000, 1),
        }


def _durable_write_new(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())


def _durable_replace(path: Path, data: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def execute_canary(plan: dict, receipt_path: Path, model_factory: Callable | None = None) -> dict:
    """At-most-one automatic dispatch per thread; pending receipts block replay.

    A crash after API dispatch remains unknown and requires manual reconciliation;
    a JSON receipt and LangGraph checkpoints do NOT provide exactly-once billing.
    """
    digest = plan_sha256(plan)
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text(encoding="utf-8"))
        if prior.get("plan_sha256") != digest:
            raise ValueError("Existing receipt belongs to another plan")
        if prior.get("status") in {"completed", "failed", "budget_stopped"}:
            return prior
        raise RuntimeError("Execution status unknown/inflight; refusing automatic replay of paid calls")
    if plan.get("mode") != "engineering_canary_not_scientific_results":
        raise ValueError("Only a non-scientific engineering canary is supported")
    # Reserve before any external LLM call; no automatic retries after a crash.
    receipt = {
        "schema_version": "8.4",
        "mode": plan["mode"],
        "plan_sha256": digest,
        "design_sha256": plan["design_sha256"],
        "cell": plan["cell"],
        "treatment": plan["treatment"],
        "status": "inflight_unknown_if_interrupted",
        "observations": [],
        "total_calls": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "pricing_is_operator_supplied": True,
        "scientific_claim_allowed": False,
    }
    _durable_write_new(receipt_path, receipt)
    evaluator = None
    try:
        model = (model_factory or (lambda spec: LiveModel(**spec)))(plan["treatment"]["model"])
        evaluator = CanaryEvaluator(plan, model)
        for assignment in plan["task_seed_pairs"]:
            receipt["observations"].append(evaluator.evaluate(assignment))
            # Persist each finished evaluation, but do not treat interrupted calls as unbilled.
            receipt["total_calls"] = evaluator.calls
            receipt["total_tokens"] = evaluator.tokens
            receipt["estimated_cost_usd"] = round(evaluator.estimated_usd, 8)
            receipt["call_receipts"] = list(evaluator.call_receipts)
            _durable_replace(receipt_path, receipt)
        receipt["status"] = "completed"
    except BudgetStop as exc:
        receipt["status"] = "budget_stopped"
        receipt["error"] = str(exc)
    except Exception as exc:
        receipt["status"] = "failed"
        # No raw third-party exception text: SDK errors can contain sensitive details.
        receipt["error"] = type(exc).__name__
    if evaluator is not None:
        receipt["total_calls"] = evaluator.calls
        receipt["total_tokens"] = evaluator.tokens
        receipt["estimated_cost_usd"] = round(evaluator.estimated_usd, 8)
        receipt["call_receipts"] = list(evaluator.call_receipts)
    _durable_replace(receipt_path, receipt)
    return receipt
