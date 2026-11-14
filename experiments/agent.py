
from agent.runner import AgentRunner
from providers.base import LLMProvider
from tools.registry import ToolRegistry

from experiments.proposal import (
    ExperimentProposal,
    parse_proposal,
)


EXPERIMENT_INSTRUCTIONS = """
Tu es un Experiment Agent spécialisé dans la conception
d'expériences scientifiques et mathématiques en Python.

Ton rôle est de PROPOSER une expérience, pas de l'exécuter.

Pour chaque demande :
- Formule une hypothèse testable.
- Décris un protocole reproductible.
- Choisis une graine aléatoire explicite.
- Génère du code Python 3.12 complet.
- Déclare les dépendances externes nécessaires.
- Précise les résultats attendus.

Contraintes :
- Privilégie la bibliothèque standard Python.
- Utilise la seed dans le code si de l'aléatoire intervient.
- Fixe les versions des dépendances avec ==.
- Utilise [] si aucune dépendance externe n'est nécessaire.
- N'inclus aucun accès réseau, secret ou commande système.
- Ne prétends jamais avoir exécuté le code.

Réponds UNIQUEMENT avec un objet JSON valide.
Aucun Markdown, aucune balise de code.

Clés exactes :
{
  "hypothesis": "hypothèse testable",
  "protocol": ["étape 1", "étape 2"],
  "seed": 42,
  "dependencies": [],
  "code": "programme Python complet",
  "expected_output": "résultats attendus"
}

Le code doit être une chaîne JSON correctement échappée.
"""


class ExperimentAgent:
    def __init__(self, provider: LLMProvider):
        self.runner = AgentRunner(
            name="experiment",
            description=(
                "Conçoit des expériences Python reproductibles"
            ),
            instructions=EXPERIMENT_INSTRUCTIONS,
            provider=provider,
            registry=ToolRegistry(),
        )

    def propose(self, request: str) -> ExperimentProposal:
        response = self.runner.propose(request)

        if response.tool_calls:
            raise ValueError(
                "Experiment Agent must not call tools"
            )

        if not response.text:
            raise ValueError(
                "Experiment Agent returned no proposal"
            )

        return parse_proposal(response.text)
