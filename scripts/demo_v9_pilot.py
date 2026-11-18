"""Prepare, authorize and inspect the V9 paired pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.retry_pilot import PilotCorpus
from experiments.retry_pilot_runner import PhaseApproval, PilotJournal, approval_target, final_plan, run_phase, summarize
from research.critique.evidence import read_document

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("preview", "approval", "execute", "status"):
        command = commands.add_parser(action)
        command.add_argument("--corpus", type=Path, default=ROOT / "examples" / "v9_retry_pilot.json")
        if action != "preview":
            command.add_argument("--journal", type=Path, required=True)
        if action in {"approval", "execute"}:
            command.add_argument("--phase", choices=["calibration", "measurement"], required=True)
        if action == "execute":
            command.add_argument("--target-sha256", required=True)
            command.add_argument("--actor", required=True)
    args = parser.parse_args(argv)
    corpus = PilotCorpus.model_validate(read_document(args.corpus))
    plan = final_plan(corpus)
    if args.action == "preview":
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    journal = PilotJournal(args.journal)
    if args.action == "status":
        print(json.dumps(summarize(plan, journal), ensure_ascii=False, indent=2))
        return
    target = approval_target(plan, journal, args.phase)
    print(json.dumps({**target, "model": plan["model_profile"]["snapshot"]}, ensure_ascii=False, indent=2))
    if args.action == "approval":
        return
    approval = PhaseApproval(phase=args.phase, target_sha256=args.target_sha256, actor=args.actor)
    if approval.target_sha256 != target["target_sha256"]:
        raise PermissionError("Le SHA-256 fourni ne correspond pas à cette phase")
    if input("Saisir APPROVE pour autoriser cette phase et son budget : ").strip() != "APPROVE":
        raise PermissionError("Phase non autorisée")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    run_phase(corpus, journal, approval)
    print(json.dumps(summarize(plan, journal), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
