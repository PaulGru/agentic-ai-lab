"""V8.4.2: allowlisted scientific workloads. No user-provided Python or shell.

The generated source is constructed only from a fixed template and safely
serialized *data*. Replacing this with generated LLM code is NOT supported.
"""
from __future__ import annotations

import json
import math
import statistics
from typing import Any

from execution.contracts import ExperimentJob

WORKLOAD_NAME = "synthetic.mean"
OUTPUT_NAMES = frozenset({"metrics.json", "run_metadata.json"})
MAX_ITEMS = 100
MAX_ABS_VALUE = 1_000_000


def validate_workload(job: ExperimentJob) -> dict[str, Any]:
    if job.workload.name != WORKLOAD_NAME:
        raise ValueError(f"Unsupported workload: {job.workload.name!r}")
    if set(job.expected_artifacts) != OUTPUT_NAMES:
        raise ValueError("synthetic.mean requires metrics.json and run_metadata.json")
    parameters = job.workload.parameters
    if set(parameters) != {"values"}:
        raise ValueError("synthetic.mean only accepts the 'values' parameter")
    values = parameters["values"]
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_ITEMS:
        raise ValueError("values must be a list of 1..100 numbers")
    for value in values:
        if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
            raise ValueError("values must contain only finite numbers (not booleans)")
        if abs(value) > MAX_ABS_VALUE:
            raise ValueError("absolute value may not exceed 1,000,000")
    if job.resources.gpus != 0:
        raise ValueError("Local V8.4.2 does not support GPUs")
    if job.resources.cpus > 2 or job.resources.memory_mb > 512:
        raise ValueError("Local V8.4.2 supports at most 2 CPU and 512 MB")
    if job.resources.walltime_minutes > 2:
        raise ValueError("Local V8.4.2 supports at most 2 minutes")
    return {"values": values}


def render_source(job: ExperimentJob) -> str:
    """Build stdlib-only Python from trusted text + repr(JSON payload)."""
    data = validate_workload(job)
    payload = {
        "values": data["values"],
        "seed": job.seed,
        "job_id": job.job_id,
        "sha256": job.fingerprint(),
        "workload": job.workload.name,
    }
    serialized = json.dumps(payload, sort_keys=True, allow_nan=False)
    # repr() forms a safe Python string literal, even if data contains quotes.
    code = r'''import json
import statistics
import sys

payload = json.loads(__PAYLOAD__)
values = payload["values"]
metrics = {
    "count": len(values),
    "mean": statistics.fmean(values),
    "minimum": min(values),
    "maximum": max(values),
}
metadata = {
    "job_id": payload["job_id"],
    "job_sha256": payload["sha256"],
    "workload": payload["workload"],
    "seed": payload["seed"],
    "python_version": sys.version.split()[0],
    "execution": "docker_sandbox",
}
print(json.dumps({
    "kind": "agentic-v842-artifacts",
    "sha256": payload["sha256"],
    "files": {"metrics.json": metrics, "run_metadata.json": metadata},
}, allow_nan=False, sort_keys=True))
'''.replace("__PAYLOAD__", repr(serialized))
    if len(code.encode("utf-8")) > 12_000:
        raise ValueError("Trusted workload exceeds sandbox input limit")
    return code


def decode_artifacts(job: ExperimentJob, stdout: str) -> dict[str, Any]:
    """Validate the single JSON message from the isolated runner."""
    if len(stdout.encode("utf-8")) > 8_192:
        raise ValueError("Sandbox output exceeds 8192 bytes")
    def reject_nonfinite(token: str):
        raise ValueError(f"Nonfinite JSON value: {token}")
    try:
        data = json.loads(stdout, parse_constant=reject_nonfinite)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Sandbox did not return one JSON artifact envelope") from error
    if not isinstance(data, dict) or set(data) != {"kind", "sha256", "files"}:
        raise ValueError("Malformed sandbox artifact envelope")
    if data["kind"] != "agentic-v842-artifacts" or data["sha256"] != job.fingerprint():
        raise ValueError("Sandbox artifact digest mismatch")
    files = data["files"]
    if not isinstance(files, dict) or set(files) != OUTPUT_NAMES:
        raise ValueError("Missing or unexpected sandbox artifacts")
    metrics, metadata = files["metrics.json"], files["run_metadata.json"]
    if not isinstance(metrics, dict) or set(metrics) != {"count", "mean", "minimum", "maximum"}:
        raise ValueError("Invalid metrics format")
    if type(metrics["count"]) is not int or metrics["count"] != len(job.workload.parameters["values"]):
        raise ValueError("Incorrect count")
    if not all(type(metrics[k]) in (int, float) and math.isfinite(metrics[k]) for k in ("mean", "minimum", "maximum")):
        raise ValueError("Invalid metric values")
    actual_values = job.workload.parameters["values"]
    expected_mean = statistics.fmean(actual_values)
    if not math.isclose(metrics["mean"], expected_mean, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("Inconsistent mean metric")
    if metrics["minimum"] != min(actual_values) or metrics["maximum"] != max(actual_values):
        raise ValueError("Inconsistent min/max metrics")
    if not isinstance(metadata, dict) or metadata.get("job_sha256") != job.fingerprint():
        raise ValueError("Invalid run metadata")
    if metadata.get("execution") != "docker_sandbox" or metadata.get("job_id") != job.job_id:
        raise ValueError("Invalid execution provenance")
    if metadata.get("workload") != job.workload.name or metadata.get("seed") != job.seed:
        raise ValueError("Invalid workload provenance")
    if not isinstance(metadata.get("python_version"), str) or not metadata["python_version"]:
        raise ValueError("Missing Python runtime version")
    return files
