"""V8.2 — PowerShell-friendly CLI with PostgreSQL checkpoint persistence."""

import argparse
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command

from research.literature import (
    LiteratureCatalog, LiteratureReview, digest_json, render_markdown,
    validate_approved_mission, validate_evidence,
)
from research.literature_graph import build_literature_graph, pending_literature_approval
from research.literature_search import InsufficientLiteratureError, OpenAlexRateLimitError
from scripts.demo_v81 import database_url


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def config_for(thread: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", thread):
        raise ValueError("Invalid thread ID")
    return {"configurable": {"thread_id": f"v82-literature:{thread}"}}


def export_approved_literature(values: dict, thread: str, base: Path | None = None) -> tuple[Path, Path]:
    if values.get("status") != "approved":
        raise ValueError("Literature review is not approved")
    mission = validate_approved_mission(values["mission_document"])
    catalog = LiteratureCatalog.model_validate(values["catalog"])
    review = LiteratureReview.model_validate(values["review"])
    validate_evidence(review, catalog)
    if digest_json(values["catalog"]) != values["catalog_sha256"]:
        raise ValueError("Catalog modified after review")
    if digest_json(values["review"]) != values["review_sha256"]:
        raise ValueError("Review modified after approval")
    out_dir = (base or ROOT / "missions") / thread
    json_path = out_dir / "literature.json"
    markdown_path = out_dir / "literature.md"
    document = {
        "schema_version": "8.2",
        "status": "approved",
        "mission_sha256": values["mission_document"]["mission_sha256"],
        "catalog_sha256": values["catalog_sha256"],
        "review_sha256": values["review_sha256"],
        "catalog": catalog.model_dump(),
        "review": review.model_dump(),
        "verification_level": "catalog_metadata_and_exact_abstract_excerpts_only",
    }
    outputs = {
        json_path: json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        markdown_path: render_markdown(catalog, review, mission),
    }
    # Check all collisions before attempting either write. Retry can be idempotent.
    for path, content in outputs.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, content in outputs.items():
        if not path.exists():
            with path.open("x", encoding="utf-8") as handle:
                handle.write(content)
    return json_path, markdown_path


def show_state(graph, cfg):
    snapshot = graph.get_state(cfg)
    if not snapshot.values:
        print("Revue introuvable.")
        return
    values = snapshot.values
    print("\n=== LITERATURE REVIEW ===")
    print("Statut :", values.get("status"))
    if values.get("catalog"):
        print("Notices :", len(values["catalog"]["papers"]))
        print("Requêtes :", values["catalog"]["requested_queries"])
        print("Diagnostic :", json.dumps(values["catalog"]["query_log"], ensure_ascii=False, indent=2))
    if values.get("review"):
        print(json.dumps(values["review"], ensure_ascii=False, indent=2))
    if pending_literature_approval(snapshot):
        print("\nEn attente d'une approbation humaine.")


def main():
    parser = argparse.ArgumentParser(description="V8.2 — Literature review")
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("start", "status", "resume", "export"):
        cmd = commands.add_parser(action)
        cmd.add_argument("--thread", required=True)
        if action == "start":
            cmd.add_argument("--mission", type=Path, required=True)
            cmd.add_argument("--max-queries", type=int, default=4, choices=range(1, 5))
            cmd.add_argument("--per-query", type=int, default=5, choices=range(1, 11))
            cmd.add_argument(
                "--query", action="append", dest="search_queries",
                metavar="KEYWORDS",
                help="Requête bibliographique courte (répéter 1 à 4 fois). "
                     "N'altère pas mission.json ; les requêtes sont tracées.",
            )
        if action == "resume":
            cmd.add_argument("--decision", required=True, choices=["approve", "reject"])
    args = parser.parse_args()
    cfg = config_for(args.thread)
    with PostgresSaver.from_conn_string(database_url()) as saver:
        saver.setup()
        graph = build_literature_graph(
            saver,
            max_queries=getattr(args, "max_queries", 4),
            per_query=getattr(args, "per_query", 5),
            search_queries=getattr(args, "search_queries", None),
        )
        snapshot = graph.get_state(cfg)
        if args.action == "status":
            show_state(graph, cfg)
            return
        if args.action == "start":
            if snapshot.values:
                raise SystemExit("Review thread already exists; use status or a new thread.")
            document = json.loads(args.mission.read_text(encoding="utf-8-sig"))
            validate_approved_mission(document)  # Fail before any network or LLM call.
            try:
                graph.invoke({"mission_document": document}, config=cfg, durability="sync")
            except OpenAlexRateLimitError as exc:
                raise SystemExit(
                    f"\nOpenAlex a refusé une requête (429).\n{exc}\n"
                    "Aucune revue LLM n'a été générée. Corrige la limite OpenAlex "
                    "avant de relancer avec un NOUVEAU --thread. "
                    "Conserve tes checkpoints PostgreSQL."
                ) from exc
            except InsufficientLiteratureError as exc:
                raise SystemExit(
                    f"\nRecherche bibliographique insuffisante.\n{exc}\n"
                    "Aucune revue LLM n'a été générée. "
                    "Relance avec un NOUVEAU --thread et des --query plus courtes. "
                    "Ne supprime pas les checkpoints PostgreSQL."
                ) from exc
        elif args.action == "resume":
            pending = pending_literature_approval(snapshot)
            if pending is None:
                raise SystemExit("No pending literature approval")
            if args.decision == "approve":
                print(json.dumps(pending["review"], ensure_ascii=False, indent=2))
                print("Catalog fingerprint:", pending["catalog_sha256"])
                print("Review fingerprint:", pending["review_sha256"])
                if input("Tape OUI après vérification scientifique : ").strip() != "OUI":
                    print("Aucune décision envoyée.")
                    return
            graph.invoke(Command(resume={
                "decision": args.decision,
                "mission_sha256": pending["mission_sha256"],
                "catalog_sha256": pending["catalog_sha256"],
                "review_sha256": pending["review_sha256"],
            }), config=cfg, durability="sync")
        if args.action == "export" or args.action == "resume":
            values = graph.get_state(cfg).values
            if values.get("status") == "approved":
                paths = export_approved_literature(values, args.thread)
                print("Artefacts exportés :", *(str(p) for p in paths), sep="\n- ")
        show_state(graph, cfg)


if __name__ == "__main__":
    main()
