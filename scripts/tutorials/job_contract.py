"""V8.4.1 CLI: generate, validate, simulate. No real execution."""
import argparse
import json
from pathlib import Path

from execution.backends.fake import FakeBackend
from execution.contracts import ExperimentJob
from execution.examples import sample_job


def main():
    parser = argparse.ArgumentParser(description="V8.4.1 generic job contract (no execution)")
    sub = parser.add_subparsers(dest="action", required=True)
    make = sub.add_parser("example", help="Write a self-contained generic sample job")
    make.add_argument("--out", required=True)
    for command in ("validate", "simulate"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--job", required=True)
    args = parser.parse_args()

    if args.action == "example":
        destination = Path(args.out)
        if destination.exists():
            parser.error("Output already exists; refusing to overwrite")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            sample_job().model_dump_json(indent=2) + "\n", encoding="utf-8",
        )
        print(f"Example job created: {destination}")
        return

    document = json.loads(Path(args.job).read_text(encoding="utf-8"))
    job = ExperimentJob.model_validate(document)
    print("Job contract validated:", job.job_id)
    print("SHA-256:", job.fingerprint())
    print("Workload:", job.workload.name, "(logical identifier, not executable code)")
    print("Resources:", job.resources.model_dump())

    if args.action == "simulate":
        backend = FakeBackend()
        handle = backend.submit(job)
        states = [backend.status(handle).status.value]
        backend.mark_running(handle)
        states.append(backend.status(handle).status.value)
        backend.finish(handle, success=True)
        states.append(backend.status(handle).status.value)
        result = backend.collect(handle)
        print(json.dumps({
            "backend": handle.backend,
            "job_id": handle.job_id,
            "states": states,
            "simulated": result.simulated,
            "files": result.files,
            "warning": result.warning,
        }, indent=2, ensure_ascii=False))
        print("No Python workload, LLM, Docker container or HPC job was executed.")


if __name__ == "__main__":
    main()
