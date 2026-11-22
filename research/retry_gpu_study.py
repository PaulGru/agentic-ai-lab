"""Bind the GPU catalog experiment to the approved mission, literature and corpus."""
from analysis.contracts import fingerprint
from analysis.retry_pilot import LIMITS, PROCEDURE, QUESTION
from execution.retry_gpu import PROTOCOL_VERSION, WORKLOAD, RetryGPUConfig, action_plan
from execution.retry_gpu_worker import MODEL, REVISION, worker_sha256
from experiments.retry_pilot import PilotCorpus, prepare_plan
from research.experimental_design import validate_approved_literature

STUDY_KIND = "brief_retry_gpu_study"


def protocol_for(mission_document, literature_document, corpus: PilotCorpus, config: RetryGPUConfig):
    mission, _, _ = validate_approved_literature(mission_document, literature_document)
    schedule = prepare_plan(corpus)
    return {
        "schema_version": PROTOCOL_VERSION,
        "workload": WORKLOAD,
        "requested_question": mission.research_question,
        "measured_question": QUESTION,
        "scope": "Catalogue fictif en lecture seule et comparaison à modèle fixé.",
        "procedure": list(PROCEDURE),
        "estimand": "Écart apparié d'exactitude entre les politiques sous erreur transitoire.",
        "control": "Le même écart est mesuré en condition normale.",
        "limitations": list(LIMITS),
        "mission_sha256": fingerprint(mission_document),
        "literature_sha256": fingerprint(literature_document),
        "corpus_sha256": schedule["corpus_sha256"],
        "schedule_sha256": schedule["plan_sha256"],
        "model": MODEL,
        "revision": REVISION,
        "worker_sha256": worker_sha256(),
        "config": config.model_dump(mode="json"),
    }


def check_study(source):
    if (set(source) != {"kind", "corpus", "mission", "literature", "protocol", "protocol_sha256"}
            or source["kind"] != STUDY_KIND):
        raise ValueError("Invalid catalog GPU study source")
    corpus = PilotCorpus.model_validate(source["corpus"])
    approved = source["protocol"]
    config = RetryGPUConfig.model_validate(approved["config"])
    expected = protocol_for(source["mission"], source["literature"], corpus, config)
    # A stored run remains readable after changes to the local worker. Submission
    # still checks the current source against the historical approved digest.
    expected["worker_sha256"] = approved["worker_sha256"]
    if approved != expected or source["protocol_sha256"] != fingerprint(expected):
        raise ValueError("Catalog GPU study differs from the approved protocol")
    return corpus, action_plan(expected, corpus)
