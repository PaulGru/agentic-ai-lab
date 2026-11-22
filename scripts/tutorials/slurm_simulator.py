"""Standalone V8.4.3 offline Slurm preview/simulator."""
import argparse
from pathlib import Path
from execution.contracts import ExperimentJob
from execution.backends.slurm_sim import SlurmSimBackend, render_sbatch_preview


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", default="examples/v841_job.json")
    p.add_argument("--db", default="runs_control/v843_slurm_sim.sqlite3")
    p.add_argument("action", choices=["preview", "simulate"])
    args = p.parse_args()
    job = ExperimentJob.model_validate_json(Path(args.job).read_text(encoding="utf-8"))
    if args.action == "preview":
        print(render_sbatch_preview(job))
    else:
        backend = SlurmSimBackend(args.db)
        handle = backend.submit(job)
        print("Submitted to OFFLINE simulator:", handle.model_dump())
        print(backend.status(handle).status.value)
        # Existing simulation may already be advanced: avoid re-running real workloads.
        if backend.status(handle).status.value == "submitted":
            print(backend.advance(handle).status.value)
        if backend.status(handle).status.value == "running":
            print(backend.advance(handle).status.value)
        print(backend.collect(handle).model_dump(mode="json"))

if __name__ == "__main__":
    main()
