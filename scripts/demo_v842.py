"""V8.4.2 real local Docker canary. No LLMs, HPC or untrusted code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from execution.backends.docker_transport import IMAGE

from execution.backends.local import LocalDockerBackend
from execution.contracts import ExperimentJob, JobStatus
from execution.workloads import render_source


def read_job(path: str) -> ExperimentJob:
    return ExperimentJob.model_validate_json(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="V8.4.2 isolated local Docker workload")
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("check", "run"):
        p = sub.add_parser(action)
        p.add_argument("--job", required=True)
        if action == "run":
            p.add_argument("--out", default=None, help="Output directory (created once)")
    args = parser.parse_args()
    try:
        job = read_job(args.job)
        render_source(job)  # fails before any Docker operation
        print("Validated:", job.job_id)
        print("SHA256:", job.fingerprint())
        print("Workload:", job.workload.name)
        print("Resources:", job.resources.model_dump())
        if args.action == "check":
            print("No execution performed. Allowed workload: synthetic.mean")
            return
        output = Path(args.out) if args.out else Path("runs_local") / job.job_id
        if output.exists():
            raise FileExistsError(f"Output directory already exists: {output}")
        if not shutil.which("docker"):
            raise RuntimeError("Docker CLI missing. Install/start Docker Desktop first")
        try:
            inspected = subprocess.run(
                ["docker", "image", "inspect", IMAGE],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=10, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Docker daemon did not respond") from exc
        if inspected.returncode != 0:
            raise RuntimeError(
                f"Docker image unavailable: {IMAGE}. "
                "Start Docker Desktop, then build it with "
                "docker build -t agentic-lab-sandbox:v7.2 -f sandbox/Dockerfile sandbox"
            )
        print("\nREAL isolated Docker execution (no LLM/API calls).")
        print("Type EXECUTE to authorize this job:")
        if input("> ").strip() != "EXECUTE":
            print("Cancelled; no job submitted")
            return
        backend = LocalDockerBackend(execution_authorized=True)
        handle = backend.submit(job)
        print("Submitted:", handle.remote_id)
        try:
            snapshot = backend.wait(handle, timeout=job.resources.walltime_minutes * 60 + 15)
        except KeyboardInterrupt:
            backend.cancel(handle)
            backend.wait(handle, timeout=8)
            raise SystemExit("Cancellation requested; check Docker Desktop for container cleanup")
        print("Status:", snapshot.status.value)
        print("Message:", snapshot.message)
        if snapshot.status != JobStatus.SUCCEEDED:
            raise SystemExit("No artifacts: job did not succeed")
        result = backend.collect(handle)
        output.mkdir(parents=True, exist_ok=False)
        for name, contents in result.files.items():
            (output / name).write_text(
                json.dumps(contents, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        receipt = {
            "job_id": handle.job_id,
            "job_spec_sha256": handle.spec_sha256,
            "backend": handle.backend,
            "simulated": result.simulated,
            "status": snapshot.status.value,
            "warning": result.warning,
            "artifacts": sorted(result.files),
        }
        (output / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print("Results saved:", output)
        print(json.dumps(result.files["metrics.json"], indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"V8.4.2 error: {exc}\n")


if __name__ == "__main__":
    main()
