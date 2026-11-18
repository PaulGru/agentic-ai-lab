"""Inspect, approve and export a manuscript from a completed V8.6 review."""
from __future__ import annotations

import argparse
from pathlib import Path

from research.critique.contracts import HumanDecision, LLMSettings
from research.critique.store import ReviewStore
from research.manuscript import (
    assist_manuscript, build_manuscript, export_manuscript, make_editorial_request,
    record_manuscript, reconstruct_manuscript, render_markdown, verify_manuscript_export,
)
from research.memory.contracts import MemoryBinding, MemoryContext
from research.memory.integration import open_session


def authorize():
    if input("Saisir APPROVE pour autoriser exactement cette version : ").strip() != "APPROVE":
        raise PermissionError("Autorisation refusée")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--review", required=True)
    commands = parser.add_subparsers(dest="action", required=True)
    preview = commands.add_parser("preview")
    preview.add_argument("--assisted", action="store_true")
    assist = commands.add_parser("assist")
    assist.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    assist.add_argument("--model", required=True)
    assist.add_argument("--actor", required=True)
    decide = commands.add_parser("decide")
    decide.add_argument("--assisted", action="store_true")
    decide.add_argument("--decision", choices=["approve", "reject"], required=True)
    decide.add_argument("--target-sha256", required=True)
    decide.add_argument("--actor", required=True)
    decide.add_argument("--comment", default="")
    for action in ("export", "verify"):
        export = commands.add_parser(action)
        export.add_argument("--target-sha256", required=True)
        export.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    if not args.store.is_file():
        raise FileNotFoundError("Existing review database required")
    store = ReviewStore(args.store, read_only=args.action in {"preview", "export", "verify"})
    if args.action == "verify":
        verified = verify_manuscript_export(store, args.review, args.target_sha256, args.out_dir)
        print("Export et sources vérifiés", verified.fingerprint())
        return
    if args.action == "export":
        print(export_manuscript(store, args.review, args.target_sha256, args.out_dir))
        return
    if args.action == "assist":
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        manuscript = build_manuscript(store, args.review)
        request = make_editorial_request(store, manuscript, LLMSettings(provider=args.provider, model=args.model))
        print(request.model_dump_json(indent=2))
        print("SHA-256 de la requête", request.fingerprint())
        authorize()
        if manuscript.memory_context_sha256:
            binding = MemoryBinding.model_validate(store.require("memory_binding", args.review))
            context = MemoryContext.model_validate(store.require("memory_context", args.review))
            open_session(binding, read_only=True).verify(context)
        manuscript = assist_manuscript(store, manuscript, request, HumanDecision(
            decision="approve", target_sha256=request.fingerprint(), actor=args.actor))
    else:
        manuscript = reconstruct_manuscript(store, args.review, assisted=args.assisted)
    print(render_markdown(manuscript))
    print("SHA-256 du manuscrit", manuscript.fingerprint())
    if args.action == "decide":
        decision = HumanDecision(decision=args.decision, target_sha256=args.target_sha256, actor=args.actor,
                                 comment=args.comment)
        if decision.target_sha256 != manuscript.fingerprint():
            raise PermissionError("Le manuscrit a changé depuis la prévisualisation")
        if decision.decision == "approve":
            authorize()
        record_manuscript(store, manuscript, decision)


if __name__ == "__main__":
    main()
