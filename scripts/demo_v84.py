"""V8.4 — opt-in engineering canary; PowerShell + PostgreSQL LangGraph."""

import argparse
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command

from research.execution_contract import (
    binding_template, prepare_canary_plan, verify_approved_design,
    pretty_json,
)
from research.execution_graph import build_execution_graph, pending_execution_approval
from scripts.demo_v81 import database_url


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def config_for(thread: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", thread):
        raise ValueError("Invalid thread ID")
    return {"configurable": {"thread_id": f"v84-canary:{thread}"}}


def load_document(path: Path) -> dict:
    if path.stat().st_size > 1_000_000:
        raise ValueError(f"File is too large: {path}")
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError("Expected JSON object")
    return obj


def show_state(graph, cfg):
    snapshot = graph.get_state(cfg)
    if not snapshot.values:
        print("Aucun pilote pour ce thread.")
        return
    v = snapshot.values
    print("\n=== V8.4 — CANARY (PAS DES RÉSULTATS SCIENTIFIQUES) ===")
    print("Statut :", v.get("status"))
    print("Plan SHA256 :", v.get("digest"))
    plan = v.get("plan", {})
    if plan:
        print("Cellule :", plan.get("cell", {}).get("cell_id"))
        print("Traitement :", pretty_json(plan.get("treatment")))
        print("Évaluations prévues :", len(plan.get("task_seed_pairs", [])))
        print("Appels LLM maximum estimés :", plan.get("worst_case_calls"))
        print("Coût conservateur estimé (USD) :", plan.get("conservative_estimated_usd"))
    if pending_execution_approval(snapshot):
        print("\nApprobation humaine en attente. Aucun appel LLM expérimental effectué.")
    if v.get("report"):
        report = v["report"]
        print("Appels enregistrés :", report["total_calls"])
        print("Tokens enregistrés :", report["total_tokens"])
        print("Coût estimé (USD) :", report["estimated_cost_usd"])
        print("Tâches réussies :", sum(x["task_success"] for x in report["observations"]),
              "/", len(report["observations"]))
        print("Reçu :", v.get("receipt_path"))
        if report.get("error"):
            print("Erreur :", report["error"])


def main():
    parser = argparse.ArgumentParser(description="V8.4 — first REAL LLM canary (not paper results)")
    subs = parser.add_subparsers(dest="action", required=True)
    template = subs.add_parser("template")
    template.add_argument("--design", required=True, type=Path)
    template.add_argument("--matrix", required=True, type=Path)
    template.add_argument("--out", required=True, type=Path)
    for action in ("start", "status", "resume"):
        p = subs.add_parser(action)
        p.add_argument("--thread", required=True)
        if action == "start":
            p.add_argument("--design", type=Path, required=True)
            p.add_argument("--matrix", type=Path, required=True)
            p.add_argument("--bindings", type=Path, required=True)
            p.add_argument("--cell", required=True)
            p.add_argument("--evaluations", type=int, default=2)
        if action == "resume":
            p.add_argument("--decision", choices=["approve", "reject"], required=True)
    args = parser.parse_args()

    if args.action == "template":
        design = verify_approved_design(load_document(args.design), load_document(args.matrix))
        output = pretty_json(binding_template(design)) + "\n"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as h:
            h.write(output)
        print("Template généré :", args.out)
        print("Renseigne explicitement les noms de modèles, les traitements et les prix ; les placeholders bloquent l'exécution.")
        return

    cfg = config_for(args.thread)
    # A separate PostgreSQL namespace keeps existing V8.1, V8.2, V8.3 and V7.4 checkpoints intact.
    with PostgresSaver.from_conn_string(database_url()) as saver:
        saver.setup()
        graph = build_execution_graph(saver, ROOT / "missions")
        snapshot = graph.get_state(cfg)
        if args.action == "status":
            show_state(graph, cfg)
            return
        if args.action == "start":
            if snapshot.values:
                raise SystemExit("Existing canary thread; do not overwrite. Use status or a NEW thread.")
            plan = prepare_canary_plan(
                load_document(args.design), load_document(args.matrix), load_document(args.bindings),
                cell_id=args.cell, evaluations=args.evaluations,
            )
            print("\nIMPORTANT: seules des tâches synthétiques arithmétiques seront exécutées.")
            print("Ce pilote n'est PAS un résultat scientifique du plan V8.3.")
            graph.invoke({"thread": args.thread, "plan": plan}, config=cfg, durability="sync")
        elif args.action == "resume":
            pending = pending_execution_approval(snapshot)
            if pending is None:
                raise SystemExit("No pending V8.4 approval for this thread")
            if args.decision == "approve":
                print("\nCANARY sur API payante :", pretty_json(pending))
                if input("Tape OUI pour autoriser ces appels LLM : ").strip() != "OUI":
                    print("Aucune décision envoyée.")
                    return
            graph.invoke(Command(resume={
                "decision": args.decision,
                "plan_sha256": pending["plan_sha256"],
            }), config=cfg, durability="sync")
        show_state(graph, cfg)


if __name__ == "__main__":
    main()
