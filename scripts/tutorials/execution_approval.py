"""V8.4.5 — PostgreSQL HITL + SQLite durable jobs. Local or OFFLINE Slurm simulator.

For local Docker, keep the process alive until completion after approval;
otherwise a process crash requires manual reconciliation. Never retry a
possibly-submitted job on your own.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import quote

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command

from execution.backends.local import LocalDockerBackend
from execution.backends.slurm_sim import SlurmSimBackend
from execution.contracts import ExperimentJob
from execution.job_manager import DurableJobManager
from execution.workflow import ExecutionWorkflow, pending_approval

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def config_for(thread: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", thread):
        raise ValueError("Invalid thread ID")
    return {
        "configurable": {"thread_id": f"v845-execution:{thread}"},
        "recursion_limit": 30,
    }


def db_url() -> str:
    password = os.environ["POSTGRES_PASSWORD"]
    return f"postgresql://lab:{quote(password, safe='')}@127.0.0.1:55432/agentic_lab"


def print_status(graph, cfg, manager):
    snap = graph.get_state(cfg)
    print("\n=== V8.4.5 WORKFLOW ===")
    if not snap.values:
        print("Workflow not found")
        return
    print("Workflow status:", snap.values.get("status"))
    print("Job:", snap.values.get("job_id"))
    pending = pending_approval(snap)
    if pending:
        print("HUMAN APPROVAL REQUIRED:", json.dumps(pending, indent=2))
    job_id = snap.values.get("job_id")
    if job_id:
        record = manager.get(job_id)
        if record:
            print("Durable execution ledger:", json.dumps(record.summary(), indent=2))
            print("Note: execution ledger may be newer than finished LangGraph snapshot")
    if not snap.next and not pending:
        print("Graph has finished its current decision; use status/advance to refresh job")


def main():
    p = argparse.ArgumentParser(description="V8.4.5 controlled execution workflow")
    sub = p.add_subparsers(dest="action", required=True)
    for action in ("start", "status", "resume", "advance", "cancel"):
        s = sub.add_parser(action)
        s.add_argument("--thread", required=True)
        s.add_argument("--backend", choices=["slurm-sim", "docker-local"], default="slurm-sim")
        s.add_argument("--jobs-db", default="runs_control/v845_jobs.sqlite3")
        s.add_argument("--sim-db", default="runs_control/v845_slurm_sim.sqlite3")
        if action == "start":
            s.add_argument("--job", required=True)
        if action == "resume":
            s.add_argument("--decision", required=True, choices=["approve", "reject"])
        if action == "advance":
            s.add_argument("--fail", action="store_true")
    args = p.parse_args()
    cfg = config_for(args.thread)
    backend = (SlurmSimBackend(args.sim_db) if args.backend == "slurm-sim"
               else LocalDockerBackend(execution_authorized=True))
    manager = DurableJobManager(backend, args.jobs_db)

    with PostgresSaver.from_conn_string(db_url()) as saver:
        saver.setup()
        graph = ExecutionWorkflow(manager, backend=args.backend).build(saver)
        snapshot = graph.get_state(cfg)
        if args.action == "start":
            if snapshot.values:
                raise SystemExit("Thread exists: use status or resume; never reuse a thread")
            job = ExperimentJob.model_validate_json(Path(args.job).read_text(encoding="utf-8"))
            graph.invoke({"job_document": job.model_dump(mode="json")},
                         config=cfg, durability="sync")
        elif args.action == "resume":
            pending = pending_approval(snapshot)
            if not pending:
                raise SystemExit("No pending approval")
            if pending["backend"] != args.backend:
                raise SystemExit("Backend mismatch; approval is backend-specific")
            if args.decision == "approve":
                print("Approve the exact spec SHA-256:", pending["sha256"])
                print("Backend:", pending["backend"], "Simulated:", pending["simulated"])
                if input("Type APPROVE to authorize submission: ").strip() != "APPROVE":
                    raise SystemExit("No approval sent")
            graph.invoke(Command(resume={
                "decision": args.decision, "sha256": pending["sha256"],
                "backend": pending["backend"],
            }), config=cfg, durability="sync")
            # The V8.4.2 local backend uses daemon worker threads. Do not exit
            # immediately or the process may die while its Docker job is active.
            if args.backend == "docker-local" and args.decision == "approve":
                job_id = snapshot.values["job_id"]
                job = ExperimentJob.model_validate(snapshot.values["job_document"])
                deadline = time.monotonic() + job.resources.walltime_minutes * 60 + 15
                try:
                    while time.monotonic() < deadline:
                        record = manager.refresh(job_id)
                        if record.phase not in {"submitted", "running"}:
                            break
                        time.sleep(0.2)
                    else:
                        manager.cancel(job_id)
                except KeyboardInterrupt:
                    manager.cancel(job_id)
                    raise SystemExit("Interrupted; cancellation requested. Check execution ledger")
        elif args.action in {"advance", "cancel"}:
            if not snapshot.values or not snapshot.values.get("job_id"):
                raise SystemExit("No job in this workflow")
            job_id = snapshot.values["job_id"]
            record = manager.get(job_id)
            if record is None or record.handle is None:
                raise SystemExit("No submitted job to advance/cancel")
            if args.action == "advance":
                if args.backend != "slurm-sim" or record.handle.backend != "slurm-sim":
                    raise SystemExit("advance works only with offline Slurm simulator")
                backend.advance(record.handle, fail=args.fail)
                manager.refresh(job_id)
            else:
                manager.cancel(job_id)
        elif args.action == "status":
            if snapshot.values and snapshot.values.get("job_id"):
                record = manager.get(snapshot.values["job_id"])
                if record and record.phase in {"submitted", "running", "submission_reserved"}:
                    manager.refresh(record.job_id)
        print_status(graph, cfg, manager)


if __name__ == "__main__":
    main()
