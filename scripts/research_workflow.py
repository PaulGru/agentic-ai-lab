"""Start, inspect, approve and revise a research run from a Markdown brief."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.brief_autonomy import AutonomyBudget, authorize_autonomy, autonomy_plan, run_autonomously
from research.brief_workflow import (
    BRIEF_WORKLOADS, PROGRAM_STAGE, GPU_STAGE, OOD_STAGE, RETRY_GPU_STAGE, ROOT, SYNTHETIC_STAGE,
    BriefWorkflow, Decision, Revision,
)
from research.critique.contracts import LLMSettings
from research.critique.evidence import local_path, read_document
from scripts.research_progress import watch_progress


def read_brief(path):
    with local_path(path).resolve().open("rb") as handle:
        body = handle.read(48_004)
    if len(body) > 48_003:
        raise ValueError("Brief is too large")
    return body.decode("utf-8-sig")


def confirm(gate):
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    try:
        return input(f"{gate['prompt']} Saisir APPROVE pour continuer ou STOP pour suspendre : ").strip() == "APPROVE"
    except EOFError:
        return False


def apply_gate(workflow, gate, actor):
    if gate["kind"] == "authorization" and gate["stage"] not in {
        SYNTHETIC_STAGE, OOD_STAGE, GPU_STAGE, RETRY_GPU_STAGE, PROGRAM_STAGE,
    }:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    return workflow.approve(Decision(actor=actor, target_sha256=gate["target_sha256"]))


def guide(workflow, actor):
    status = workflow.status()
    while status["pending"]:
        gate = status["pending"]
        if not confirm(gate):
            break
        status = apply_gate(workflow, gate, actor)
    if status.get("attention"):
        print(status["attention"])
    return status


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
    start.add_argument("--workload", choices=BRIEF_WORKLOADS)
    start.add_argument("--corpus")
    start.add_argument("--evaluation", help="Corpus d'évaluation indépendant pour un protocole inédit")
    start.add_argument("--auto-resources", action="store_true", default=None,
                       help="Choisir les ressources dans le catalogue local compatible")
    start.add_argument("--gpu-config")
    commands.add_parser("status")
    commands.add_parser("history")
    watch = commands.add_parser("watch", help="Suivre les étapes enregistrées sans exécuter la mission")
    watch.add_argument("--once", action="store_true", help="Afficher les étapes déjà enregistrées")
    dashboard = commands.add_parser("dashboard", help="Ouvrir le suivi web local et archiver les PDF ACL")
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.add_argument("--pdflatex", help="Chemin local vers pdflatex")
    dashboard.add_argument("--acl-style", help="Chemin local vers le fichier officiel acl.sty")
    dashboard.add_argument("--archive-only", action="store_true", help="Compiler les versions enregistrées puis quitter")
    telemetry = commands.add_parser("telemetry", help="Exporter les métriques et traces techniques vérifiées")
    trackers = telemetry.add_mutually_exclusive_group()
    trackers.add_argument("--wandb-offline", action="store_true", help="Créer une exécution W&B locale")
    trackers.add_argument("--weave-project", help="Prévisualiser et confirmer un transfert vers équipe/projet")
    guided = commands.add_parser("guide")
    guided.add_argument("--actor", required=True)
    commands.add_parser("resume-autonomy")
    for action in ("autonomy-plan", "autonomy"):
        command = commands.add_parser(action)
        command.add_argument("--llm-calls", type=int)
        command.add_argument("--openalex-queries", type=int, default=8)
        command.add_argument("--revisions", type=int, default=2)
        if action == "autonomy":
            command.add_argument("--actor", required=True)
            command.add_argument("--target-sha256", required=True)
    for action in ("approve", "revise", "recover"):
        command = commands.add_parser(action)
        command.add_argument("--target-sha256", required=True)
        command.add_argument("--actor", required=True)
        if action == "revise":
            command.add_argument("--stage", required=True,
                                 choices=["mission", "catalog", "literature", "resource_selection", "protocol", "implementation", "draft"])
            command.add_argument("--reason", required=True)
            command.add_argument("--brief")
            command.add_argument("--query", action="append")
            command.add_argument("--workload", choices=BRIEF_WORKLOADS)
            command.add_argument("--corpus")
            command.add_argument("--evaluation")
            command.add_argument("--gpu-config")
    args = parser.parse_args(argv)
    workflow = BriefWorkflow(local_path(args.dir))
    if args.action == "start":
        settings = LLMSettings(provider=args.provider, model=args.model, max_output_tokens=args.max_output_tokens)
        result = workflow.start(args.id, read_brief(args.brief), settings, args.workload,
                                read_document(args.corpus) if args.corpus else None,
                                gpu_config=read_document(local_path(args.gpu_config).resolve()) if args.gpu_config else None,
                                evaluation=read_document(local_path(args.evaluation).resolve()) if args.evaluation else None,
                                auto_resources=args.auto_resources)
    elif args.action == "status":
        result = workflow.status()
    elif args.action == "history":
        result = workflow.store.history()
    elif args.action == "watch":
        watch_progress(workflow.store, once=args.once)
        return
    elif args.action == "dashboard":
        from scripts.research_dashboard import Dashboard, serve_dashboard

        dashboard = Dashboard(workflow, compiler=args.pdflatex, acl_style=args.acl_style)
        if not args.archive_only:
            serve_dashboard(dashboard, args.port)
            return
        dashboard.archive_pending()
        result = dashboard.snapshot()
        if any(item["status"] == "error" for item in result["editions"]):
            print(json.dumps(result["editions"], ensure_ascii=False, indent=2))
            raise SystemExit(1)
        result = result["editions"]
    elif args.action == "telemetry":
        from observability.research_telemetry import export_telemetry, publish_weave, record_wandb_offline, weave_plan

        payload, path = export_telemetry(workflow)
        result = {"metrics": str(path), "agent_calls": len(payload["agent_calls"])}
        if args.wandb_offline:
            result["wandb_offline"] = record_wandb_offline(payload, path.parent.parent / "wandb-offline")
        elif args.weave_project:
            plan = weave_plan(payload, args.weave_project)
            if not confirm(plan):
                raise PermissionError("Transfert Weave non approuvé")
            result["weave"] = publish_weave(payload, args.weave_project, approved_sha256=plan["target_sha256"])
    elif args.action == "guide":
        result = guide(workflow, args.actor)
    elif args.action in {"autonomy-plan", "autonomy", "resume-autonomy"}:
        from dotenv import load_dotenv

        if args.action != "resume-autonomy":
            default_calls = 6 if workflow.store.current()["state"].get("auto_resources") else 4
            budget = AutonomyBudget(llm_calls=default_calls if args.llm_calls is None else args.llm_calls,
                                    openalex_queries=args.openalex_queries,
                                    revisions=args.revisions)
            result = autonomy_plan(workflow, budget)
            if args.action == "autonomy":
                if args.target_sha256 != result["target_sha256"] or not confirm(result):
                    raise PermissionError("Mission autonome non autorisée")
                authorize_autonomy(workflow, budget, Decision(actor=args.actor, target_sha256=args.target_sha256))
        if args.action != "autonomy-plan":
            load_dotenv(ROOT / ".env")
            result = run_autonomously(workflow)
    else:
        details = {"actor": args.actor, "target_sha256": args.target_sha256}
        if args.action == "revise":
            result = workflow.revise(Revision(**details, stage=args.stage, reason=args.reason,
                                              brief=read_brief(args.brief) if args.brief else None,
                                              queries=args.query, workload=args.workload,
                                              corpus=read_document(args.corpus) if args.corpus else None,
                                              gpu_config=read_document(local_path(args.gpu_config).resolve()) if args.gpu_config else None,
                                              evaluation=read_document(local_path(args.evaluation).resolve()) if args.evaluation else None))
        elif args.action == "recover":
            result = workflow.recover(Decision(**details))
        else:
            status = workflow.status()
            gate = status["pending"]
            if not gate or gate["target_sha256"] != args.target_sha256:
                raise PermissionError("The approval does not target the displayed action")
            if not confirm(gate):
                raise PermissionError("Étape non autorisée")
            result = apply_gate(workflow, gate, args.actor)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
