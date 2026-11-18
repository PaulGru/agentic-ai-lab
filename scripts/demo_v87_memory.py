"""V8.7 research memory CLI: explicit imports, retrieval, history and lifecycle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.memory.contracts import MemoryBinding, SourceOrigin
from research.memory.integration import open_session


def local(path):
    if path.startswith(("\\\\", "//")) or "://" in path:
        raise ValueError("Use explicit local artifact paths")
    return str(Path(path).resolve())


def main(argv=None):
    parser = argparse.ArgumentParser(description="V8.7 — mémoire de recherche durable et vérifiable")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--db", default=None)
    group.add_argument("--postgres", action="store_true", help="Use RESEARCH_MEMORY_DATABASE_URL from the environment")
    parser.add_argument("--schema", default="research_memory_v87")
    parser.add_argument("--project", required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("init")
    for action in ("import-mission", "import-context"):
        command = sub.add_parser(action)
        command.add_argument("--path", required=True)
    command = sub.add_parser("import-literature")
    command.add_argument("--mission", required=True)
    command.add_argument("--literature", required=True)
    command = sub.add_parser("import-design")
    for field in ("mission", "literature", "design", "matrix"):
        command.add_argument("--" + field, required=True)
    command = sub.add_parser("import-review")
    command.add_argument("--store", required=True)
    command.add_argument("--review", required=True)
    command = sub.add_parser("search")
    command.add_argument("--query", required=True)
    command.add_argument("--limit", type=int, default=5)
    command.add_argument("--max-bytes", type=int, default=12000)
    command.add_argument("--kind", action="append", choices=["hypothesis", "project_plan", "literature", "observation", "decision", "lesson"])
    command.add_argument("--scope", action="append", choices=["planning", "abstract_only", "pedagogical", "engineering", "human_note", "controlled_pilot"])
    command = sub.add_parser("history")
    command.add_argument("--memory-id", required=True)
    command.add_argument("--verify-source", action="store_true")
    command = sub.add_parser("note")
    command.add_argument("--title", required=True)
    command.add_argument("--text", required=True)
    command.add_argument("--key", required=True)
    command.add_argument("--actor", required=True)
    for action in ("confirm", "invalidate", "supersede"):
        command = sub.add_parser(action)
        command.add_argument("--memory-id", required=True)
        command.add_argument("--target-sha256", required=True)
        command.add_argument("--actor", required=True)
        command.add_argument("--reason", required=True)
        if action == "supersede":
            command.add_argument("--replacement-id", required=True)
    args = parser.parse_args(argv)
    if args.postgres:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    binding = MemoryBinding(project_id=args.project, backend="postgres" if args.postgres else "sqlite",
                            path=None if args.postgres else local(args.db or "runs_control/v87_memory.sqlite3"),
                            schema_name=args.schema)
    session = open_session(binding, read_only=args.action == "history", initialize=args.action == "init")
    memory = session.memory
    if args.action == "init":
        print("Mémoire initialisée ; aucune source importée, aucun appel LLM.")
        return
    if args.action == "search":
        context = memory.search(args.project, args.query, role="memory_cli", limit=args.limit, max_bytes=args.max_bytes,
                                 kinds=args.kind, scopes=args.scope)
        print(json.dumps(context.model_dump(mode="json"), ensure_ascii=False, indent=2))
        print("Context SHA-256:", context.fingerprint())
        return
    if args.action == "history":
        print(json.dumps(memory.get(args.project, args.memory_id, verify_source=args.verify_source), ensure_ascii=False, indent=2))
        return
    if args.action == "note":
        item = memory.note(args.project, title=args.title, text=args.text, logical_key=args.key, actor=args.actor)
        print(json.dumps(item.model_dump(mode="json"), ensure_ascii=False, indent=2))
        print("Statut tentative ; exclue du contexte courant jusqu'à relecture explicite.")
        print("Item SHA-256:", item.fingerprint())
        return
    if args.action in {"confirm", "invalidate", "supersede"}:
        item = memory.store.item(args.project, args.memory_id)
        if item is None:
            raise ValueError("Souvenir absent de ce projet")
        print(json.dumps({"item": item.model_dump(mode="json"), "action": args.action,
                          "reason": args.reason, "replacement_id": getattr(args, "replacement_id", None)},
                         ensure_ascii=False, indent=2))
        if input("Valider cette décision de mémoire ? Saisir APPROVE : ").strip() != "APPROVE":
            raise PermissionError("Aucune décision enregistrée")
        event = memory.change(args.project, args.memory_id, action=args.action, target_sha256=args.target_sha256,
                              actor=args.actor, reason=args.reason, replacement_id=getattr(args, "replacement_id", None))
        print(event.model_dump_json())
        return
    if args.action == "import-review":
        items = memory.ingest_review(args.project, local(args.store), args.review)
    else:
        if args.action == "import-mission":
            origin = SourceOrigin(kind="mission", path=local(args.path))
        elif args.action == "import-context":
            origin = SourceOrigin(kind="context", path=local(args.path))
        elif args.action == "import-design":
            origin = SourceOrigin(kind="context", path=local(args.design), parent_path=local(args.mission),
                                  literature_path=local(args.literature), matrix_path=local(args.matrix))
        else:
            origin = SourceOrigin(kind="literature", path=local(args.literature), parent_path=local(args.mission))
        items = memory.ingest(args.project, origin)
    print(json.dumps({"indexed_items": len(items), "memory_ids": [item.memory_id for item in items]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, PermissionError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit(str(error)) from None
