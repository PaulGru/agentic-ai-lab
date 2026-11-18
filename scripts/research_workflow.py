"""Start, inspect, approve and revise a research run from a Markdown brief."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.brief_workflow import BriefWorkflow, Decision, Revision, ROOT
from research.critique.contracts import LLMSettings
from research.critique.evidence import read_document
from scripts.demo_v9_review import local_path


def read_brief(path):
    with local_path(path).open("rb") as handle:
        body = handle.read(48_004)
    if len(body) > 48_003:
        raise ValueError("Brief is too large")
    return body.decode("utf-8-sig")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    commands = parser.add_subparsers(dest="action", required=True)
    start = commands.add_parser("start")
    start.add_argument("--id", required=True)
    start.add_argument("--brief", required=True)
    start.add_argument("--provider", required=True, choices=["openai", "anthropic"])
    start.add_argument("--model", required=True)
    start.add_argument("--max-output-tokens", type=int, default=4000)
    start.add_argument("--workload", choices=["retry_catalog", "unmapped"])
    start.add_argument("--corpus")
    commands.add_parser("status")
    commands.add_parser("history")
    for action in ("approve", "revise", "recover"):
        command = commands.add_parser(action)
        command.add_argument("--target-sha256", required=True)
        command.add_argument("--actor", required=True)
        if action == "revise":
            command.add_argument("--stage", required=True, choices=["mission", "catalog", "literature", "protocol", "draft"])
            command.add_argument("--reason", required=True)
            command.add_argument("--brief")
            command.add_argument("--query", action="append")
            command.add_argument("--workload", choices=["retry_catalog", "unmapped"])
            command.add_argument("--corpus")
    args = parser.parse_args(argv)
    workflow = BriefWorkflow(local_path(args.dir))
    if args.action == "start":
        settings = LLMSettings(provider=args.provider, model=args.model, max_output_tokens=args.max_output_tokens)
        result = workflow.start(args.id, read_brief(args.brief), settings, args.workload,
                                read_document(args.corpus) if args.corpus else None)
    elif args.action == "status":
        result = workflow.status()
    elif args.action == "history":
        result = workflow.store.history()
    else:
        details = {"actor": args.actor, "target_sha256": args.target_sha256}
        if args.action == "revise":
            result = workflow.revise(Revision(**details, stage=args.stage, reason=args.reason,
                                              brief=read_brief(args.brief) if args.brief else None,
                                              queries=args.query, workload=args.workload,
                                              corpus=read_document(args.corpus) if args.corpus else None))
        elif args.action == "recover":
            result = workflow.recover(Decision(**details))
        else:
            status = workflow.status()
            gate = status["pending"]
            if not gate or gate["target_sha256"] != args.target_sha256:
                raise PermissionError("The approval does not target the displayed action")
            print(json.dumps(gate, ensure_ascii=False, indent=2))
            if input("Saisir APPROVE pour autoriser cette étape : ").strip() != "APPROVE":
                raise PermissionError("Étape non autorisée")
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
            result = workflow.approve(Decision(**details))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
