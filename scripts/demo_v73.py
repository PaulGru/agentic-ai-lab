
"""Démonstration V7.3 : erreur -> correction -> nouvelle exécution."""

from pathlib import Path
from dotenv import load_dotenv

# Charge le fichier .env situé à la racine du projet.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from experiments.loop import ExperimentLoop
from experiments.proposal import ExperimentProposal
from experiments.reviser import ExperimentReviser
from experiments.sandbox import SandboxRunner
from research.factory import create_provider


INITIAL_CODE = """
numbers = [1, 2, 3, 4, 5]

mean = sum(numbers) / len(number)

print("Moyenne :", mean)
"""


def ask_approval(
    proposal: ExperimentProposal,
    attempt: int,
) -> bool:
    print(f"\n=== TENTATIVE {attempt} ===")
    print("\nCode proposé :\n")
    print(proposal.code)

    choice = input(
        "\nTape OUI pour autoriser CE code : "
    ).strip()

    return choice == "OUI"


def main():
    proposal = ExperimentProposal(
        hypothesis=(
            "La moyenne des nombres 1 à 5 vaut 3."
        ),
        protocol=[
            "Créer une liste de nombres.",
            "Calculer leur moyenne.",
            "Afficher le résultat.",
        ],
        seed=42,
        dependencies=[],
        code=INITIAL_CODE,
        expected_output="Moyenne : 3.0",
    )

    # Réutilise la factory multi-provider existante.
    provider = create_provider(
        native_web_search=False
    )

    reviser = ExperimentReviser(provider)

    loop = ExperimentLoop(
        sandbox=SandboxRunner(),
        reviser=reviser,
        max_attempts=3,
    )

    result = loop.run(
        proposal,
        approve=ask_approval,
    )

    print("\n=== BILAN EXPERIMENTAL ===")
    print("Statut :", result.status)
    print("Message :", result.message)
    print("Tentatives :", len(result.attempts))

    for attempt in result.attempts:
        execution = attempt.result

        print(f"\n--- Tentative {attempt.number} ---")
        print("Statut :", execution.status)
        print("Durée :", execution.duration_seconds)

        if execution.stdout:
            print("stdout :\n", execution.stdout)

        if execution.stderr:
            print("stderr :\n", execution.stderr)

    print(
        "\nNote : un succès technique ne valide "
        "pas automatiquement l'hypothèse."
    )


if __name__ == "__main__":
    main()
