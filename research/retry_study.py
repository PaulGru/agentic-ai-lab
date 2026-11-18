"""Bind an approved mission and its literature to the supported V9 workload."""
from __future__ import annotations

from analysis.contracts import fingerprint
from analysis.retry_pilot import QUESTION
from experiments.retry_pilot import PilotCorpus
from experiments.retry_pilot_runner import final_plan
from research.experimental_design import validate_approved_literature

WORKLOAD = "retry_catalog"


def protocol_for(mission_document, literature_document, corpus):
    mission, _, _ = validate_approved_literature(mission_document, literature_document)
    plan = final_plan(corpus)
    return {
        "workload": WORKLOAD,
        "requested_question": mission.research_question,
        "measured_question": QUESTION,
        "scope": "Catalogue fictif, modèle unique, lecture sans effet de bord, erreur transitoire injectée.",
        "mission_sha256": fingerprint(mission_document),
        "literature_sha256": fingerprint(literature_document),
        "pilot_plan_sha256": plan["plan_sha256"],
        "budget": plan["budget"],
    }


def check_study(source) -> PilotCorpus:
    if set(source) != {"kind", "corpus", "mission", "literature", "protocol", "protocol_sha256"}:
        raise ValueError("Invalid V9 study source envelope")
    if source["kind"] != "brief_retry_study":
        raise ValueError("Unknown V9 study source envelope")
    corpus = PilotCorpus.model_validate(source["corpus"])
    protocol = protocol_for(source["mission"], source["literature"], corpus)
    if source["protocol"] != protocol or source["protocol_sha256"] != fingerprint(protocol):
        raise ValueError("Approved workflow protocol differs from the corpus and literature")
    return corpus
