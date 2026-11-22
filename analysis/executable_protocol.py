"""Score and pair predictions against the approved protocol and evaluation corpus."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import Field

from analysis.contracts import HEX_SHA, fingerprint
from execution.contracts import StrictModel
from execution.job_manager import DurableJobManager
from execution.protocol_cpu import ARTIFACT, BACKEND, ProtocolCPUBackend, predictions
from execution.protocol_gpu import ARTIFACT as GPU_ARTIFACT, BACKEND as GPU_BACKEND, ProtocolGPUBackend, validate_receipt
from execution.ood_gpu import GPURuntime
from experiments.protocol_program import MAX_VALUE, Number
from research.protocol_study import compile_study

SOURCE_KIND = "verified_executable_protocol"


class Prediction(StrictModel):
    case_id: str
    seed: int
    value: Number | None
    outcome: Literal["answered", "invalid_output"]
    elapsed_ns: int = Field(ge=0)
    model_tokens: int = Field(ge=0)


class ScoredPrediction(Prediction):
    cell_id: str
    correct: bool
    job_sha256: str = Field(pattern=HEX_SHA)
    artifacts_sha256: str = Field(pattern=HEX_SHA)


class CellScore(StrictModel):
    levels: dict[str, str]
    correct: int
    total: int
    accuracy_percent: float


class Contrast(StrictModel):
    factor_name: str
    level_a: str
    level_b: str
    pairs: int
    successes_a: int
    successes_b: int
    gain_points: float


class ProtocolAnalysis(StrictModel):
    source_kind: Literal["verified_executable_protocol"] = SOURCE_KIND
    protocol_sha256: str = Field(pattern=HEX_SHA)
    action_plan_sha256: str = Field(pattern=HEX_SHA)
    evaluation_sha256: str = Field(pattern=HEX_SHA)
    cells: dict[str, CellScore]
    contrasts: dict[str, Contrast]
    trials: list[ScoredPrediction]
    limitations: list[str]
    gpu_model: str | None = None
    gpu_revision: str | None = None
    gpu_runtime: GPURuntime | None = None
    scientific_evidence: Literal[False] = False
    ready_for_scientific_inference: Literal[False] = False

    def fingerprint(self):
        return fingerprint(self.model_dump(mode="json"))


def _checked_receipt(manager, job, backend_type, backend_name, filename):
    record = manager.get(job.job_id)
    if record is None or record.phase != "succeeded" or record.handle is None or record.artifacts is None:
        raise RuntimeError(f"Complete receipt required for {job.job_id}")
    handle = record.handle
    if (record.backend != backend_type.__name__ or handle.backend != backend_name
            or record.spec_sha256 != job.fingerprint() or handle.spec_sha256 != record.spec_sha256
            or handle.job_id != job.job_id or record.artifacts.handle != handle
            or handle.simulated or record.artifacts.simulated or set(record.artifacts.files) != {filename}):
        raise ValueError("Job receipt provenance mismatch")
    return record.artifacts.files[filename], record.artifacts_sha256


def _score(row, cell, job, artifacts_sha, expected):
    return ScoredPrediction(**row.model_dump(), cell_id=cell, correct=row.value == expected[row.case_id],
                            job_sha256=job.fingerprint(), artifacts_sha256=artifacts_sha)


def _cpu_predictions(manager, jobs, expected):
    scored = []
    for job in jobs:
        artifact, artifacts_sha = _checked_receipt(manager, job, ProtocolCPUBackend, BACKEND, ARTIFACT)
        actual = [Prediction.model_validate(row) for row in artifact]
        reproduced = predictions(job, clock=lambda: 0)
        if [row.model_dump(exclude={"elapsed_ns"}) for row in actual] != [
                {k: v for k, v in row.items() if k != "elapsed_ns"} for row in reproduced]:
            raise ValueError("Recorded CPU predictions differ from the executable plan")
        scored.extend(_score(row, job.workload.parameters["cell_id"], job, artifacts_sha, expected) for row in actual)
    return scored


def _gpu_predictions(manager, job, plan, expected):
    artifact, artifacts_sha = _checked_receipt(manager, job, ProtocolGPUBackend, GPU_BACKEND, GPU_ARTIFACT)
    receipt = validate_receipt(job, plan, artifact)
    scored = []
    for trial, response in zip(job.workload.parameters["trials"], receipt.responses, strict=True):
        text = response.text.strip()
        value = int(text) if re.fullmatch(r"[-+]?[0-9]{1,10}", text) else None
        if value is not None and abs(value) > MAX_VALUE:
            value = None
        row = Prediction(case_id=trial["case_id"], seed=trial["seed"], value=value,
                         outcome="invalid_output" if value is None else "answered",
                         elapsed_ns=response.elapsed_ns, model_tokens=response.input_tokens + response.output_tokens)
        scored.append(_score(row, trial["cell_id"], job, artifacts_sha, expected))
    return scored, {"gpu_model": receipt.model, "gpu_revision": receipt.revision, "gpu_runtime": receipt.runtime}


def analyze_protocol(source, manager: DurableJobManager):
    if not manager.read_only:
        raise PermissionError("Analysis requires the original read-only job ledger")
    design, evaluation, jobs, plan = compile_study(source)
    expected = {case.id: case.expected for case in evaluation.cases}
    if plan["gpu"]:
        scored, model = _gpu_predictions(manager, jobs[0], plan, expected)
    else:
        scored, model = _cpu_predictions(manager, jobs, expected), {}
    cells, contrasts = summarize_cells(design, scored)
    return ProtocolAnalysis(
        protocol_sha256=plan["protocol_sha256"], action_plan_sha256=plan["plan_sha256"],
        evaluation_sha256=plan["evaluation_sha256"], cells=cells, contrasts=contrasts, trials=scored,
        limitations=[
            "Pilote descriptif sur un corpus numérique local fixé avant la mesure.",
            "Les répétitions de graines sur les mêmes tâches peuvent être dépendantes.",
            "Les contrastes décrivent les pourcentages observés. Les intervalles et tests inférentiels restent à établir.",
        ], **model,
    )


def summarize_cells(design, scored):
    cells = {}
    for cell in design.factor_matrix():
        rows = [row for row in scored if row.cell_id == cell["cell_id"]]
        correct = sum(row.correct for row in rows)
        cells[cell["cell_id"]] = CellScore(levels=cell["factor_levels"], correct=correct, total=len(rows),
                                           accuracy_percent=100 * correct / len(rows))
    contrasts = {}
    for test in design.hypothesis_tests:
        by_level = {test.level_a: {}, test.level_b: {}}
        for row in scored:
            levels = cells[row.cell_id].levels
            level = levels[test.factor_name]
            if level not in by_level or any(levels[k] != v for k, v in test.fixed_levels.items()):
                continue
            controls = tuple(sorted((k, v) for k, v in levels.items() if k != test.factor_name))
            by_level[level][row.case_id, row.seed, controls] = row.correct
        a, b = by_level[test.level_a], by_level[test.level_b]
        if not a or a.keys() != b.keys():
            raise ValueError("Missing matched observations for the approved contrast")
        contrasts[test.hypothesis_id] = Contrast(
            factor_name=test.factor_name, level_a=test.level_a, level_b=test.level_b,
            pairs=len(a), successes_a=sum(a.values()), successes_b=sum(b.values()),
            gain_points=100 * (sum(b.values()) - sum(a.values())) / len(a),
        )
    return cells, contrasts
