
from experiments.agent import ExperimentAgent
from research.factory import create_provider


def main():
    provider = create_provider(native_web_search=False)

    agent = ExperimentAgent(provider)

    proposal = agent.propose(
        "Simule 10 000 lancers de deux dés équilibrés. "
        "Vérifie si la somme 7 est la plus fréquente. "
        "Utilise uniquement la bibliothèque standard Python."
    )

    print("\n=== EXPERIENCE PROPOSEE ===\n")
    print(proposal.model_dump_json(indent=2))

    print("\nCode généré mais NON EXECUTE.")


if __name__ == "__main__":
    main()
