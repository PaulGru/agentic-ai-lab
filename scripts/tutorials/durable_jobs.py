"""V8.4.4 durable ledger demo, backed by an offline Slurm simulator."""
import argparse
import json
from pathlib import Path
from execution.contracts import ExperimentJob
from execution.backends.slurm_sim import SlurmSimBackend
from execution.job_manager import DurableJobManager


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["submit", "status", "advance", "cancel", "collect"])
    p.add_argument("--job", default="examples/v841_job.json")
    p.add_argument("--db", default="runs_control/jobs.sqlite3")
    p.add_argument("--sim-db", default="runs_control/slurm_sim.sqlite3")
    args = p.parse_args()
    job = ExperimentJob.model_validate_json(Path(args.job).read_text(encoding="utf-8"))
    backend = SlurmSimBackend(args.sim_db)
    manager = DurableJobManager(backend, args.db)
    if args.action == "submit":
        print("OFFLINE SIMULATOR ONLY: no Slurm job or scientific computation will run")
        result = manager.submit(job)
    elif args.action == "advance":
        # Refresh the ledger first: another demo may have advanced the same
        # simulated job in the shared simulator database.
        record = manager.get(job.job_id)
        if record is None or record.handle is None:
            raise SystemExit("Submit before advancing")
        record = manager.refresh(job.job_id)
        if record.phase in {"submitted", "running"}:
            backend.advance(record.handle)
            result = manager.refresh(job.job_id)
        elif record.phase in {"succeeded", "failed", "cancelled"}:
            print(f"Job already {record.phase}; nothing to advance")
            result = record
        else:
            raise SystemExit(
                f"Cannot advance job in phase {record.phase}; "
                "inspect status and reconcile if needed"
            )
    elif args.action == "cancel":
        result = manager.cancel(job.job_id)
    elif args.action == "collect":
        print(json.dumps(manager.collect(job.job_id).model_dump(mode="json"), indent=2))
        return
    else:
        result = manager.refresh(job.job_id)
    print(json.dumps(result.summary(), indent=2))

if __name__ == "__main__":
    main()
