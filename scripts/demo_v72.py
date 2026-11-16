
"""Démonstration locale du sandbox V7.2."""

from experiments.sandbox import SandboxRunner


CODE = """
import random
from collections import Counter

random.seed(42)

results = Counter(
    random.randint(1, 6) + random.randint(1, 6)
    for _ in range(10000)
)

for value in range(2, 13):
    print(value, results[value])

print(
    "Plus fréquente :",
    results.most_common(1)[0][0]
)
"""


def main():
    print("=== CODE A EXECUTER ===\n")
    print(CODE)

    choice = input(
        "Pour autoriser cette exécution, tape OUI : "
    ).strip()

    if choice != "OUI":
        print("Exécution annulée.")
        return

    result = SandboxRunner().run(
        CODE,
        approved=True,
    )

    print("\n=== RESULTAT ===")
    print("Statut :", result.status)
    print("Code de sortie :", result.exit_code)
    print("Durée (s) :", result.duration_seconds)
    print("stdout :\n", result.stdout)
    print("stderr :\n", result.stderr)


if __name__ == "__main__":
    main()
