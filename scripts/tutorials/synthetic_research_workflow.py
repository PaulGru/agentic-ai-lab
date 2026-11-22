"""Run the synthetic V8.9 pilot with resumable PostgreSQL checkpoints."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from research.critique.evidence import read_document
from research.memory.contracts import MemoryBinding
from research.memory.integration import open_session
from research.research_workflow import ResearchWorkflow, WorkflowInput, config_for, make_plan, pending_decision
from scripts.tutorials.scientific_manuscript import authorize


def load_request(path):
    data = read_document(path)
    context = data.get("context_path", "")
    if not isinstance(context, str) or not context.strip() or context.startswith(("\\\\", "//")) or "://" in context:
        raise ValueError("Only local context files are supported")
    # Resolve relative to the workflow document, for consistent later resumes.
    data["context_path"] = str((Path(path).resolve().parent / context).resolve())
    return WorkflowInput.model_validate(data)


def session(root, args, *, initialize=False, read_only=False):
    if args.memory_db:
        binding = MemoryBinding(project_id=args.project or args.run_id, path=args.memory_db)
        binding = binding.model_copy(update={"path": str(Path(binding.path).resolve())})
    else:
        binding = MemoryBinding(project_id=args.project or args.run_id, path=str(root / "memory.sqlite3"))
    return open_session(binding, initialize=initialize, read_only=read_only)


def show(graph, cfg):
    snapshot = graph.get_state(cfg)
    print(json.dumps({"workflow_status": snapshot.values.get("workflow_status"),
                      "review_status": snapshot.values.get("status"),
                      "pending": pending_decision(snapshot), "next": snapshot.next},
                     ensure_ascii=False, indent=2))
    return snapshot


def resume(graph, cfg, pending, args):
    if pending is None:
        raise ValueError("No pending decision")
    decision = {"decision": args.decision, "actor": args.actor, "target_sha256": args.target_sha256}
    if args.target_sha256 != pending["target_sha256"]:
        raise PermissionError("Le SHA-256 saisi ne correspond pas à la décision en attente")
    if pending["kind"] == "v86_revision_input":
        if args.draft:
            decision["draft"] = read_document(args.draft)
    elif args.comment:
        decision["comment"] = args.comment
    if args.decision in {"approve", "propose"}:
        print(json.dumps(pending, ensure_ascii=False, indent=2))
        authorize()
    graph.invoke(Command(resume=decision), cfg, durability="sync")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--memory-db", help="Existing shared SQLite memory")
    parser.add_argument("--project")
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("plan", "start", "offline"):
        commands.add_parser(action).add_argument("--input", required=True, type=Path)
    commands.add_parser("status")
    commands.add_parser("recover")
    resume_parser = commands.add_parser("resume")
    resume_parser.add_argument("--actor", required=True)
    resume_parser.add_argument("--decision", choices=["approve", "reject", "request_changes", "skip", "propose"], required=True)
    resume_parser.add_argument("--target-sha256", required=True)
    resume_parser.add_argument("--draft", type=Path)
    resume_parser.add_argument("--comment", default="")
    args = parser.parse_args(argv)
    request = load_request(args.input) if hasattr(args, "input") else None
    if args.action == "plan":
        print(json.dumps(make_plan(request)[2], ensure_ascii=False, indent=2))
        return
    if args.root is None or not args.run_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", args.run_id):
        parser.error("--root and a valid --run-id are required")
    if request and request.run_id != args.run_id:
        raise ValueError("Workflow identifier differs from --run-id")
    if str(args.root).startswith(("\\\\", "//")):
        raise ValueError("Use a local workflow directory")
    root = args.root.resolve()
    cfg = config_for(args.run_id)
    if args.action == "offline":
        if request.critic_settings or request.writer_settings:
            raise ValueError("Offline mode requires a workflow without LLM settings")
        workflow = ResearchWorkflow(root, session(root, args, initialize=not args.memory_db))
        if workflow.store.get("workflow_plan", args.run_id):
            raise ValueError("Workflow already exists")
        graph = workflow.build(InMemorySaver())
        graph.invoke({"workflow_input": request.model_dump(mode="json")}, cfg, durability="sync")
        pending = pending_decision(show(graph, cfg))
        for expected, choice in (("v89_execution_approval", "approve"), ("v86_revision_input", "skip"),
                                 ("v89_manuscript_approval", "approve")):
            if not pending or pending["kind"] != expected:
                raise RuntimeError("Unexpected offline workflow state")
            if choice == "approve":
                authorize()
            graph.invoke(Command(resume={"decision": choice, "actor": "local-operator",
                                         "target_sha256": pending["target_sha256"]}), cfg, durability="sync")
            pending = pending_decision(show(graph, cfg))
        return

    from langgraph.checkpoint.postgres import PostgresSaver
    from scripts.tutorials.scientific_critic import postgres_url
    with PostgresSaver.from_conn_string(os.getenv("V89_CHECKPOINT_DATABASE_URL") or postgres_url()) as saver:
        if args.action == "start":
            saver.setup()
        read_only = args.action == "status"
        memory = session(root, args, initialize=args.action == "start" and not args.memory_db, read_only=read_only)
        workflow = ResearchWorkflow(root, memory, read_only=read_only)
        graph = workflow.build(saver)
        snapshot = graph.get_state(cfg)
        if args.action == "start":
            if snapshot.values or workflow.store.get("workflow_plan", args.run_id) is not None:
                raise ValueError("Workflow already exists")
            graph.invoke({"workflow_input": request.model_dump(mode="json")}, cfg, durability="sync")
        elif args.action == "resume":
            resume(graph, cfg, pending_decision(snapshot), args)
        elif args.action == "recover":
            if pending_decision(snapshot) or not snapshot.next:
                raise ValueError("Recovery requires a failed node with no decision pending")
            graph.invoke(None, cfg, durability="sync")
        show(graph, cfg)


if __name__ == "__main__":
    main()
