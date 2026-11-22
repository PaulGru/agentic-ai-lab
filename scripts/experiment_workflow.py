
"""V7.4 - Démonstration persistante exécutée depuis Windows."""

import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command

from experiments.graph import ExperimentWorkflow
from experiments.proposal import ExperimentProposal
from experiments.sandbox import SandboxRunner
from research.factory import create_provider


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def database_url() -> str:
    password = os.environ["POSTGRES_PASSWORD"]
    return (
        f"postgresql://lab:{quote(password, safe='')}"
        "@127.0.0.1:55432/agentic_lab"
    )


def config_for(thread: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", thread):
        raise ValueError("Invalid thread ID")

    # Nouveau namespace, sans toucher aux checkpoints V4-V6.
    return {
        "configurable": {
            "thread_id": f"v74-experiment:{thread}"
        },
        "recursion_limit": 60,
    }


def demo_proposal() -> ExperimentProposal:
    return ExperimentProposal(
        hypothesis="La moyenne de 1 à 5 vaut 3.",
        protocol=[
            "Créer la liste.",
            "Calculer la moyenne.",
            "Afficher le résultat.",
        ],
        seed=42,
        dependencies=[],
        code=(
            "numbers = [1, 2, 3, 4, 5]\n"
            "print(sum(numbers) / len(number))\n"
        ),
        expected_output="3.0",
    )


def pending_approval(snapshot):
    for task in snapshot.tasks:
        for item in task.interrupts:
            payload = item.value
            if (
                isinstance(payload, dict)
                and payload.get("kind") == "experiment_approval"
            ):
                return payload
    return None


def show_state(graph, config):
    snapshot = graph.get_state(config)

    if not snapshot.values:
        print("Expérience introuvable.")
        return

    values = snapshot.values

    print("\n=== ETAT DU WORKFLOW ===")
    print("Statut :", values.get("status"))
    print("Message :", values.get("message", ""))
    print("Tentatives :", len(values.get("attempts", [])))

    for item in values.get("attempts", []):
        execution = item["execution"]
        print(f"\n--- Tentative {item['number']} ---")
        print("Statut :", execution["status"])
        print("stdout :", execution["stdout"][:2000])
        print("stderr :", execution["stderr"][:2000])

    pending = pending_approval(snapshot)

    if pending:
        print("\n=== APPROBATION EN ATTENTE ===")
        print("Tentative :", pending["attempt"])
        print("Hypothèse :", pending["hypothesis"])
        print("SHA-256 :", pending["sha256"])
        print("\nCode proposé :\n")
        print(pending["code"])
    elif not snapshot.next:
        print("\nWorkflow terminé.")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(
        dest="action",
        required=True,
    )

    for name in ("start", "status", "resume", "telemetry"):
        command = commands.add_parser(name)
        command.add_argument("--thread", required=True)

        if name == "telemetry":
            command.add_argument("--wandb-offline", action="store_true")

        if name == "start":
            source = command.add_mutually_exclusive_group(
                required=True
            )
            source.add_argument(
                "--demo", action="store_true"
            )
            source.add_argument("--question")

        if name == "resume":
            command.add_argument(
                "--decision",
                required=True,
                choices=["approve", "reject"],
            )

    args = parser.parse_args()
    config = config_for(args.thread)

    with PostgresSaver.from_conn_string(
        database_url()
    ) as saver:
        if args.action != "telemetry":
            saver.setup()

        workflow = ExperimentWorkflow(
            provider_factory=lambda: create_provider(
                native_web_search=False
            ),
            sandbox=SandboxRunner(),
        )
        graph = workflow.build(saver)

        snapshot = graph.get_state(config)

        if args.action == "telemetry":
            from observability.research_telemetry import export_code_telemetry, record_wandb_offline

            if not snapshot.values:
                raise SystemExit("Expérience introuvable.")
            directory = Path(__file__).resolve().parents[1] / "runs_control" / "v74-telemetry"
            payload, path = export_code_telemetry(snapshot.values, directory)
            result = {"metrics": str(path)}
            if args.wandb_offline:
                result["wandb_offline"] = record_wandb_offline(payload, directory / "wandb-offline")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        if args.action == "status":
            show_state(graph, config)
            return

        if args.action == "start":
            if snapshot.values:
                raise SystemExit(
                    "Thread already exists; use status or resume."
                )

            if args.demo:
                initial = {
                    "initial_proposal": (
                        demo_proposal().model_dump()
                    )
                }
            else:
                initial = {
                    "question": args.question
                }

            graph.invoke(
                initial,
                config=config,
                durability="sync",
            )

        elif args.action == "resume":
            pending = pending_approval(snapshot)

            if pending is None:
                raise SystemExit(
                    "No pending approval for this thread."
                )

            decision = args.decision

            if decision == "approve":
                print("\nCode à autoriser :\n")
                print(pending["code"])
                print("SHA-256 :", pending["sha256"])

                confirmation = input(
                    "\nTape OUI pour confirmer : "
                ).strip()

                if confirmation != "OUI":
                    print("Aucune décision envoyée.")
                    return

            graph.invoke(
                Command(resume={
                    "decision": decision,
                    "sha256": pending["sha256"],
                }),
                config=config,
                durability="sync",
            )

        show_state(graph, config)


if __name__ == "__main__":
    main()
