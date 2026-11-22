"""V8.6 CLI: offline demonstration, persistent HITL, snapshots and revalidation.

Run from the repository root. Default assistance is offline. Real LLM calls
require --assist-* in the input, then a separate exact APPROVE confirmation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from research.critique.contracts import RevisionDraft, RevisionProposal, ReviewInput, ReviewReport
from research.critique.evidence import load_input, read_document
from research.critique.graph import ReviewWorkflow, config_for, pending_review
from research.critique.reporting import export_review, inspect_export
from research.critique.revision import verify_revision
from research.critique.rules import check_report_bindings
from research.critique.store import ReviewStore
from research.memory.integration import add_memory_options, open_session, session_from_options
from research.memory.contracts import MemoryBinding

ROOT = Path(__file__).resolve().parents[2]


def load_request(path: str) -> ReviewInput:
    document = read_document(path)
    for key in ("source_path", "context_path", "jobs_db"):
        if document.get(key) is not None:
            value = document[key]
            if not isinstance(value, str) or value.startswith(("\\\\", "//")) or "://" in value:
                raise ValueError("Review input requires explicit local paths")
            document[key] = str(Path(value).resolve())
    return ReviewInput.model_validate(document)


def show(snapshot, store: ReviewStore, review_id: str):
    print("Statut :", snapshot.values.get("status", "absent"))
    pending = pending_review(snapshot)
    if pending:
        print(json.dumps(pending, ensure_ascii=False, indent=2))
    print("Appels LLM :", json.dumps(store.call_summaries(review_id), ensure_ascii=False))


def response_for(pending: dict, *, decision: str, actor: str, comment: str = "", draft_path: str | None = None):
    if not pending:
        raise ValueError("Aucune décision en attente ; inspecter l'état ou utiliser continue pour une reprise prudente.")
    response = {"decision": decision, "target_sha256": pending["target_sha256"], "actor": actor}
    if pending["kind"] == "v86_revision_input":
        if decision == "propose" and draft_path:
            response["draft"] = RevisionDraft.model_validate(read_document(draft_path)).model_dump(mode="json")
        elif decision != "skip":
            raise ValueError("Utiliser propose --draft <fichier> ou skip.")
    else:
        response["comment"] = comment
        if decision == "approve":
            print(json.dumps(pending, ensure_ascii=False, indent=2))
            if input("Vérifier le contenu exact ci-dessus, puis saisir APPROVE : ").strip() != "APPROVE":
                raise PermissionError("Confirmation humaine exacte absente ; aucune approbation envoyée.")
    return response


def verify_recorded_review(store: ReviewStore, review_id: str) -> None:
    request = ReviewInput.model_validate(store.require("session", review_id))
    bundle = load_input(request)
    if bundle.model_dump(mode="json") != store.require("bundle", review_id):
        raise ValueError("Les sources originales ont changé")
    report = ReviewReport.model_validate(store.require("review", review_id))
    check_report_bindings(bundle, report)
    if ReviewWorkflow(store)._report(request, bundle) != report:
        raise ValueError("Le rapport ne correspond pas aux constats et aux appels LLM enregistrés")
    manager = ReviewWorkflow._manager(request)
    for artifact in store.artifacts(review_id):
        if artifact["kind"] == "proposal":
            verify_revision(bundle, report, RevisionProposal.model_validate(artifact["document"]), manager)


def postgres_url() -> str:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if os.getenv("V86_DATABASE_URL"):
        return os.environ["V86_DATABASE_URL"]
    password = os.getenv("POSTGRES_PASSWORD")
    if not password:
        raise ValueError("Configurer V86_DATABASE_URL ou POSTGRES_PASSWORD pour PostgreSQL")
    return f"postgresql://lab:{quote(password, safe='')}@127.0.0.1:55432/agentic_lab"


def main(argv=None):
    parser = argparse.ArgumentParser(description="V8.6 — critique traçable et révision prospective")
    parser.add_argument("--store", default="runs_control/v86_reviews.sqlite3")
    add_memory_options(parser)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("preview", "offline", "start"):
        command = sub.add_parser(action)
        command.add_argument("--input", required=True, help="ReviewInput JSON ; chemins relatifs à la racine du dépôt")
        if action == "offline":
            command.add_argument("--draft")
            command.add_argument("--actor", default="local-operator")
            command.add_argument("--out-dir")
    for action in ("status", "resume", "continue", "export", "verify"):
        command = sub.add_parser(action)
        command.add_argument("--review", required=True)
        if action == "resume":
            command.add_argument("--decision", choices=["approve", "reject", "request_changes", "propose", "skip"], required=True)
            command.add_argument("--actor", required=True, help="Étiquette d'opérateur ; ne constitue pas une authentification")
            command.add_argument("--comment", default="")
            command.add_argument("--draft")
        if action == "export":
            command.add_argument("--out-dir", required=True)
    command = sub.add_parser("inspect")
    command.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    if args.action == "preview":
        request = load_request(args.input)
        bundle = load_input(request)
        print(json.dumps({"input": request.model_dump(mode="json"), "bundle_sha256": bundle.fingerprint(),
                          "analysis_sha256": bundle.analysis.fingerprint(), "scientific_evidence": False},
                         ensure_ascii=False, indent=2))
        return
    if args.action == "inspect":
        package = inspect_export(args.out_dir)
        print("Export cohérent ; provenance originale NON revérifiée :", package["review_id"])
        return
    if args.action in {"export", "verify"}:
        store = ReviewStore(args.store, read_only=True)
        if args.action == "verify":
            verify_recorded_review(store, args.review)
            print("Sources originales et révisions revérifiées ; aucune conclusion scientifique autorisée.")
        else:
            print(export_review(store, args.review, args.out_dir))
        return
    if args.action == "offline":
        request = load_request(args.input)
        if request.assist_critique or request.assist_revision:
            raise PermissionError("Le démonstrateur offline refuse toute assistance LLM ; utiliser start et le HITL persistant.")
        store = ReviewStore(args.store)
        if store.get("session", request.review_id) is not None:
            raise ValueError("Cet identifiant existe déjà ; utiliser un nouvel identifiant.")
        graph = ReviewWorkflow(store, memory=session_from_options(args)).build(InMemorySaver())
        cfg = config_for(request.review_id)
        graph.invoke({"input_document": request.model_dump(mode="json")}, cfg, durability="sync")
        pending = pending_review(graph.get_state(cfg))
        if pending:
            action = "propose" if args.draft else "skip"
            response = response_for(pending, decision=action, actor=args.actor, draft_path=args.draft)
            graph.invoke(Command(resume=response), cfg, durability="sync")
        pending = pending_review(graph.get_state(cfg))
        if pending:
            print(json.dumps(pending, ensure_ascii=False, indent=2))
            approved = input("Adopter cette version prospective ? Saisir APPROVE, sinon refus : ").strip() == "APPROVE"
            graph.invoke(Command(resume={"decision": "approve" if approved else "reject",
                                         "target_sha256": pending["target_sha256"], "actor": args.actor, "comment": ""}),
                         cfg, durability="sync")
        show(graph.get_state(cfg), store, request.review_id)
        destination = args.out_dir or f"runs_control/v86_reports/{request.review_id}"
        print(export_review(store, request.review_id, destination))
        return

    from langgraph.checkpoint.postgres import PostgresSaver
    # Constructing a graph/status never creates a provider or calls an LLM.
    with PostgresSaver.from_conn_string(postgres_url()) as saver:
        saver.setup()
        store = ReviewStore(args.store, read_only=args.action == "status")
        memory = session_from_options(args, read_only=args.action == "status")
        review_id = load_request(args.input).review_id if args.action == "start" else args.review
        saved_binding = store.get("memory_binding", review_id)
        if saved_binding and memory is None:
            memory = open_session(MemoryBinding.model_validate(saved_binding), read_only=args.action == "status")
        graph = ReviewWorkflow(store, memory=memory).build(saver)
        if args.action == "start":
            request = load_request(args.input)
            cfg = config_for(request.review_id)
            if graph.get_state(cfg).values or store.get("session", request.review_id) is not None:
                raise ValueError("Cet identifiant existe déjà ; utiliser status ou un nouvel identifiant.")
            graph.invoke({"input_document": request.model_dump(mode="json")}, cfg, durability="sync")
            review_id = request.review_id
        else:
            cfg = config_for(args.review)
            review_id = args.review
            if not graph.get_state(cfg).values:
                raise ValueError("Revue introuvable")
            if args.action == "resume":
                response = response_for(pending_review(graph.get_state(cfg)), decision=args.decision,
                                        actor=args.actor, comment=args.comment, draft_path=args.draft)
                graph.invoke(Command(resume=response), cfg, durability="sync")
            elif args.action == "continue":
                if pending_review(graph.get_state(cfg)):
                    raise ValueError("Une décision humaine attendue doit passer par resume")
                graph.invoke(None, cfg, durability="sync")
        show(graph.get_state(cfg), store, review_id)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, PermissionError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        raise SystemExit(str(error)) from None
