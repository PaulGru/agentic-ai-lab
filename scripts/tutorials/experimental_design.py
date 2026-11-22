"""V8.3 — PowerShell-compatible experimental design CLI (no execution)."""

import argparse
from research.memory.integration import add_memory_options, session_from_options
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command

from research.design_graph import build_design_graph, pending_design_approval
from research.experimental_design import (
    ExperimentalDesign, render_design_markdown,
    validate_approved_literature, validate_design_against_inputs,
)
from research.literature import digest_json
from scripts.tutorials.research_mission import database_url


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def config_for(thread: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", thread):
        raise ValueError("Invalid thread ID")
    return {"configurable": {"thread_id": f"v83-design:{thread}"}}


def export_approved_design(values: dict, thread: str, base: Path | None = None):
    if values.get("status") != "approved":
        raise ValueError("Experimental design is not approved")
    mission, _, review = validate_approved_literature(
        values["mission_document"], values["literature_document"]
    )
    design = ExperimentalDesign.model_validate(values["design"])
    if design.model_dump() != values["design"]:
        raise ValueError("Design is not canonical")
    validate_design_against_inputs(design, mission, review)
    if values["mission_document"]["mission_sha256"] != values["mission_sha256"]:
        raise ValueError("Mission fingerprint mismatch")
    if digest_json(values["literature_document"]) != values["literature_input_sha256"]:
        raise ValueError("Literature input was modified")
    if digest_json(values["design"]) != values["design_sha256"]:
        raise ValueError("Design was modified")
    document = {
        "schema_version": "8.3",
        "status": "approved",
        "execution_status": "not_executed",
        "mission_sha256": values["mission_sha256"],
        "literature_sha256": values["literature_input_sha256"],
        "design_sha256": values["design_sha256"],
        "design": design.model_dump(),
        "planned_cells": design.design_cells,
        "planned_evaluations": design.total_evaluations,
        "planned_worst_case_llm_calls": design.worst_case_llm_calls,
    }
    output_dir = (base or ROOT / "missions") / thread
    json_path = output_dir / "experimental_design.json"
    md_path = output_dir / "experimental_design.md"
    matrix_path = output_dir / "design_matrix.json"
    matrix = {
        "schema_version": "8.3",
        "design_sha256": values["design_sha256"],
        "execution_status": "not_executed",
        "cells": design.factor_matrix(),
        "tasks_per_cell": design.resource_limits.tasks_per_cell,
        "seeds": design.resource_limits.seeds,
    }
    outputs = {
        json_path: json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        md_path: render_design_markdown(design, mission, values["literature_document"]),
        matrix_path: json.dumps(matrix, ensure_ascii=False, indent=2) + "\n",
    }
    # Never overwrite another approved artifact, even when exporting twice.
    for path, content in outputs.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to overwrite a different protocol: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, content in outputs.items():
        if not path.exists():
            with path.open("x", encoding="utf-8") as handle:
                handle.write(content)
    return json_path, md_path, matrix_path


def show_state(graph, cfg):
    snapshot = graph.get_state(cfg)
    if not snapshot.values:
        print("Protocole introuvable.")
        return
    values = snapshot.values
    print("\n=== V8.3 EXPERIMENTAL DESIGN ===")
    print("Statut :", values.get("status"))
    print("Mission SHA256 :", values.get("mission_sha256"))
    print("Literature SHA256 :", values.get("literature_input_sha256"))
    print("Design SHA256 :", values.get("design_sha256"))
    if values.get("design"):
        design = ExperimentalDesign.model_validate(values["design"])
        print("Cellules :", design.design_cells)
        print("Evaluations :", design.total_evaluations)
        print("Appels LLM au pire :", design.worst_case_llm_calls)
        print(json.dumps(values["design"], ensure_ascii=False, indent=2))
    if pending_design_approval(snapshot):
        print("\nApprobation humaine en attente.")


def main():
    parser = argparse.ArgumentParser(description="V8.3 — Experimental design / preregistration")
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("start", "status", "resume", "export"):
        command = commands.add_parser(action)
        command.add_argument("--thread", required=True)
        if action == "start":
            command.add_argument("--mission", type=Path, required=True)
            command.add_argument("--literature", type=Path, required=True)
        if action == "resume":
            command.add_argument("--decision", choices=["approve", "reject"], required=True)

    add_memory_options(parser)
    args = parser.parse_args()
    cfg = config_for(args.thread)
    with PostgresSaver.from_conn_string(database_url()) as saver:
        saver.setup()
        graph = build_design_graph(saver, memory=session_from_options(args, read_only=args.action in {"status", "export"}))
        snapshot = graph.get_state(cfg)
        if args.action == "status":
            show_state(graph, cfg)
            return
        if args.action == "start":
            if snapshot.values:
                raise SystemExit("Design thread already exists; use status or a new thread.")
            mission = json.loads(args.mission.read_text(encoding="utf-8-sig"))
            literature = json.loads(args.literature.read_text(encoding="utf-8-sig"))
            # No LLM call until both approved artifacts pass validation.
            validate_approved_literature(mission, literature)
            graph.invoke({"mission_document": mission, "literature_document": literature},
                         config=cfg, durability="sync")
        elif args.action == "resume":
            pending = pending_design_approval(snapshot)
            if pending is None:
                raise SystemExit("No pending design approval")
            if args.decision == "approve":
                print(json.dumps(pending["design"], ensure_ascii=False, indent=2))
                print("Design SHA256:", pending["design_sha256"])
                if input("Tape OUI après relecture scientifique : ").strip() != "OUI":
                    print("Aucune décision envoyée.")
                    return
            graph.invoke(Command(resume={
                "decision": args.decision,
                "mission_sha256": pending["mission_sha256"],
                "literature_input_sha256": pending["literature_input_sha256"],
                "design_sha256": pending["design_sha256"],
                **({"memory_context_sha256": pending["memory_context_sha256"]} if "memory_context_sha256" in pending else {}),
            }), config=cfg, durability="sync")
        if args.action in ("resume", "export"):
            values = graph.get_state(cfg).values
            if values.get("status") == "approved":
                paths = export_approved_design(values, args.thread)
                print("Artefacts exportés :", *(str(x) for x in paths), sep="\n- ")
        show_state(graph, cfg)


if __name__ == "__main__":
    main()
