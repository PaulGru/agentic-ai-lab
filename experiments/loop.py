
"""Boucle d'expérience bornée, avec approbation à chaque tentative."""

from dataclasses import dataclass
from typing import Callable, Literal

from experiments.proposal import ExperimentProposal
from experiments.reviser import ExperimentReviser
from experiments.sandbox import SandboxResult, SandboxRunner


@dataclass(frozen=True)
class AttemptRecord:
    number: int
    code: str
    result: SandboxResult


@dataclass(frozen=True)
class LoopReport:
    status: Literal[
        "completed",
        "rejected",
        "max_attempts",
        "revision_failed",
        "unchanged",
        "stopped",
    ]
    attempts: tuple[AttemptRecord, ...]
    last_proposal: ExperimentProposal
    message: str = ""


class ExperimentLoop:
    def __init__(
        self,
        sandbox: SandboxRunner,
        reviser: ExperimentReviser,
        max_attempts: int = 3,
    ):
        if (
            type(max_attempts) is not int
            or not 1 <= max_attempts <= 3
        ):
            raise ValueError(
                "max_attempts must be between 1 and 3"
            )

        self.sandbox = sandbox
        self.reviser = reviser
        self.max_attempts = max_attempts

    def run(
        self,
        proposal: ExperimentProposal,
        approve: Callable[
            [ExperimentProposal, int], bool
        ],
    ) -> LoopReport:
        if proposal.dependencies:
            raise ValueError(
                "V7.3 allows standard-library-only experiments"
            )

        current = proposal
        attempts: list[AttemptRecord] = []

        def report(
            status: str,
            message: str = "",
        ) -> LoopReport:
            return LoopReport(
                status,
                tuple(attempts),
                current,
                message,
            )

        for number in range(1, self.max_attempts + 1):

            # Une approbation est requise pour CHAQUE code.
            if approve(current, number) is not True:
                return report(
                    "rejected",
                    "Execution refused by operator",
                )

            result = self.sandbox.run(
                current.code,
                approved=True,
                dependencies=current.dependencies,
            )

            attempts.append(
                AttemptRecord(
                    number=number,
                    code=current.code,
                    result=result,
                )
            )

            # Succès technique : aucune correction nécessaire.
            if result.status == "success":
                return report(
                    "completed",
                    "Technical execution succeeded",
                )

            # Ne pas relancer sur un statut inconnu.
            if result.status not in {
                "error",
                "timeout",
                "output_limit",
            }:
                return report(
                    "stopped",
                    "Unknown sandbox execution status",
                )

            # Limite stricte du nombre de tentatives.
            if number == self.max_attempts:
                return report(
                    "max_attempts",
                    "Maximum attempts reached",
                )

            # Le LLM PROPOSE une correction.
            # Aucune exécution à cette étape.
            try:
                revised = self.reviser.revise(
                    current,
                    result,
                )
            except ValueError as error:
                return report(
                    "revision_failed",
                    str(error)[:300],
                )

            # Le modèle ne peut changer que le code.
            if (
                revised.model_dump(exclude={"code"})
                != current.model_dump(exclude={"code"})
            ):
                return report(
                    "revision_failed",
                    "Reviser changed experiment contract",
                )

            if revised.code == current.code:
                return report(
                    "unchanged",
                    "Reviser proposed no change",
                )

            # Nouvelle proposition, non encore approuvée.
            current = revised

        raise AssertionError(
            "Loop exit should be unreachable"
        )
