"""V8.1 — CLI persistante : start / status / resume."""

import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command

from research.mission import parse_brief
from research.mission_graph import build_mission_graph, pending_approval
from research.memory.integration import add_memory_options, session_from_options


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def database_url() -> str:
    password = os.environ["POSTGRES_PASSWORD"]
    return (
        f"postgresql://lab:{quote(password, safe='')}"
        "@127.0.0.1:55432/agentic_lab"
    )


def config_for(thread: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", thread):
        raise ValueError("Invalid thread ID")
    return {"configurable": {"thread_id": f"v81-mission:{thread}"}}


def show_state(graph, cfg):
    snapshot = graph.get_state(cfg)
    if not snapshot.values:
        print("Mission introuvable.")
        return
    values = snapshot.values
    print("\n=== RESEARCH MISSION ===")
    print("Statut :", values.get("status"))
    print("Brief SHA-256 :", values.get("brief_sha256"))
    print("Mission SHA-256 :", values.get("mission_sha256"))
    if values.get("mission"):
        print(json.dumps(values["mission"], ensure_ascii=False, indent=2))
    if pending_approval(snapshot):
        print("\nEn attente d'une décision humaine (approve / reject).")


def export_approved_mission(values: dict, thread: str):
    if values.get("status") != "approved":
        return None
    destination = Path("missions") / thread / "mission.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": "8.1",
        "brief_sha256": values["brief_sha256"],
        "mission_sha256": values["mission_sha256"],
        "brief_sections": values["brief_sections"],
        "mission": values["mission"],
    }
    # Pas d'écrasement silencieux d'un artefact scientifique approuvé.
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description="V8.1 Research Mission")
    add_memory_options(parser)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("start", "status", "resume"):
        cmd = commands.add_parser(action)
        cmd.add_argument("--thread", required=True)
        if action == "start":
            cmd.add_argument("--brief", type=Path, required=True)
        if action == "resume":
            cmd.add_argument("--decision", required=True,
                             choices=["approve", "reject"])

    args = parser.parse_args()
    cfg = config_for(args.thread)

    with PostgresSaver.from_conn_string(database_url()) as saver:
        saver.setup()
        graph = build_mission_graph(saver, memory=session_from_options(args, read_only=args.action == "status"))
        snapshot = graph.get_state(cfg)

        if args.action == "status":
            show_state(graph, cfg)
            return

        if args.action == "start":
            if snapshot.values:
                raise SystemExit("Mission exists; use status or a new thread.")
            text = args.brief.read_text(encoding="utf-8-sig")
            parse_brief(text)  # Échec avant tout appel LLM si brief invalide.
            graph.invoke({"brief_text": text}, config=cfg, durability="sync")

        elif args.action == "resume":
            pending = pending_approval(snapshot)
            if pending is None:
                raise SystemExit("No pending mission approval.")
            if args.decision == "approve":
                print(json.dumps(pending["mission"], ensure_ascii=False, indent=2))
                confirmation = input("\nTape OUI pour approuver ce plan : ").strip()
                if confirmation != "OUI":
                    print("Aucune décision envoyée.")
                    return
            graph.invoke(
                Command(resume={
                    "decision": args.decision,
                    "brief_sha256": pending["brief_sha256"],
                    "mission_sha256": pending["mission_sha256"],
                    **({"memory_context_sha256": pending["memory_context_sha256"]} if "memory_context_sha256" in pending else {}),
                }),
                config=cfg,
                durability="sync",
            )
            values = graph.get_state(cfg).values
            if values.get("status") == "approved":
                destination = export_approved_mission(values, args.thread)
                print(f"Mission exportée vers {destination}")

        show_state(graph, cfg)


if __name__ == "__main__":
    main()
