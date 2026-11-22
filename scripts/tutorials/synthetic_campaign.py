"""V8.4.6: local real-Docker end-to-end engineering pilot (PowerShell CLI).

Commands:
  plan --protocol examples/v846_protocol.json
  start --thread v846-001 --protocol examples/v846_protocol.json
  status --thread v846-001
  resume --thread v846-001 --decision approve|reject
  report --thread v846-001
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import quote

from dotenv import load_dotenv

from execution.campaign import PilotProtocol, build_report, save_report
from execution.contracts import ExperimentJob
from execution.backends.local import LocalDockerBackend
from execution.job_manager import DurableJobManager

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def db_url() -> str:
    password = os.environ["POSTGRES_PASSWORD"]
    return f"postgresql://lab:{quote(password, safe='')}@127.0.0.1:55432/agentic_lab"


def cfg_for(thread: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", thread):
        raise ValueError("Invalid thread ID")
    return {"configurable": {"thread_id": f"v846-pilot:{thread}"}, "recursion_limit": 30}


def check_docker_image() -> None:
    from execution.backends.docker_transport import IMAGE
    try:
        result = subprocess.run(["docker", "image", "inspect", IMAGE],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(f"Docker not available ({type(exc).__name__}); do not approve") from exc
    if result.returncode:
        raise SystemExit(f"Docker image {IMAGE!r} missing or daemon unavailable")


def print_status(graph, cfg, manager):
    from execution.campaign_workflow import pending_pilot_approval
    snap = graph.get_state(cfg)
    if not snap.values:
        print("Workflow not found")
        return
    print("\n=== V8.4.6 ENGINEERING PILOT ===")
    print("Graph status:", snap.values.get("status"))
    print("Graph message:", snap.values.get("message"))
    pending = pending_pilot_approval(snap)
    if pending:
        print("HITL PENDING:", json.dumps(pending, indent=2))
    proto = PilotProtocol.model_validate(snap.values["protocol_document"])
    for job in proto.jobs():
        record = manager.get(job.job_id)
        print(f"  {job.job_id}: {record.phase if record else 'not_submitted'}")
        if record and record.phase == "needs_reconciliation":
            print("    MANUAL RECONCILIATION REQUIRED; never auto-resubmit")
    if snap.values.get("analysis"):
        print("Analysis:", json.dumps(snap.values["analysis"], indent=2))
    if any(manager.get(j.job_id) is not None and
           manager.get(j.job_id).phase in {"submitted", "running", "submission_reserved"}
           for j in proto.jobs()):
        print("WARNING: local workers do not survive CLI exit; inspect ledger and Docker before continuing")


def main():
    parser = argparse.ArgumentParser(description="V8.4.6 end-to-end synthetic Docker pilot")
    sub = parser.add_subparsers(dest="action", required=True)
    p_plan = sub.add_parser("plan", help="Validate protocol and display derived jobs, without side effects")
    p_plan.add_argument("--protocol", required=True)
    for action in ("start", "status", "resume", "report"):
        sp = sub.add_parser(action)
        sp.add_argument("--thread", required=True)
        sp.add_argument("--jobs-db", default="runs_control/v846_jobs.sqlite3")
        if action == "start":
            sp.add_argument("--protocol", required=True)
        if action == "resume":
            sp.add_argument("--decision", choices=("approve", "reject"), required=True)
        if action == "report":
            sp.add_argument("--out", help="JSON destination; defaults to runs_control/v846_reports/<campaign>.json")
    args = parser.parse_args()

    if args.action == "plan":
        protocol = PilotProtocol.model_validate_json(Path(args.protocol).read_text(encoding="utf-8"))
        jobs = protocol.jobs()
        print(json.dumps({
            "campaign_id": protocol.campaign_id,
            "protocol_sha256": protocol.fingerprint(),
            "backend": "docker-local",
            "calls_to_llm_api": 0,
            "jobs": [{"job_id": job.job_id, "fingerprint": job.fingerprint(),
                      "workload": job.workload.name,
                      "resources": job.resources.model_dump(mode="json")}
                     for job in jobs]
        }, indent=2))
        return

    # Only the CLI imports PostgreSQL components; standalone contract/report tests
    # remain usable in a minimal Python environment.
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.types import Command
    from execution.campaign_workflow import PilotWorkflow, pending_pilot_approval

    cfg = cfg_for(args.thread)
    backend = LocalDockerBackend(execution_authorized=True)
    manager = DurableJobManager(backend, args.jobs_db)
    with PostgresSaver.from_conn_string(db_url()) as saver:
        saver.setup()
        graph = PilotWorkflow(manager).build(saver)
        snapshot = graph.get_state(cfg)
        if args.action == "start":
            if snapshot.values:
                raise SystemExit("Thread already exists; use status/resume (never overwrite checkpoints)")
            protocol = PilotProtocol.model_validate_json(Path(args.protocol).read_text(encoding="utf-8"))
            graph.invoke({"protocol_document": protocol.model_dump(mode="json")},
                         config=cfg, durability="sync")
        elif args.action == "resume":
            pending = pending_pilot_approval(snapshot)
            if not pending:
                raise SystemExit("No pending campaign approval for this thread")
            if args.decision == "approve":
                check_docker_image()
                print("Campaign:", pending["campaign_id"])
                print("Protocol SHA-256:", pending["protocol_sha256"])
                print("Backend:", pending["backend"], "Jobs:", len(pending["jobs"]))
                for job in pending["jobs"]:
                    print("  -", job["job_id"], job["sha256"])
                if input("Type APPROVE to execute all synthetic Docker jobs: ").strip() != "APPROVE":
                    raise SystemExit("No execution approved")
            graph.invoke(Command(resume={
                "decision": args.decision,
                "protocol_sha256": pending["protocol_sha256"],
                "backend": pending["backend"],
            }), config=cfg, durability="sync")
        elif args.action == "report":
            if not snapshot.values or snapshot.values.get("approved") is not True:
                raise SystemExit("No approved campaign for this thread")
            protocol = PilotProtocol.model_validate(snapshot.values["protocol_document"])
            if snapshot.values.get("protocol_sha256") != protocol.fingerprint():
                raise SystemExit("Checkpoint protocol digest mismatch")
            report = build_report(protocol, manager)
            output = args.out or f"runs_control/v846_reports/{protocol.campaign_id}.json"
            path = save_report(report, output)
            print("Report:", path)
            print("Analysis:", json.dumps(report["analysis"], indent=2))
            print("simulated:", report["simulated"], "; scientific_evidence:", report["scientific_evidence"])
        print_status(graph, cfg, manager)


if __name__ == "__main__":
    main()
