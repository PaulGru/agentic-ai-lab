"""Read-only terminal view of a research workflow's committed checkpoints."""
from time import sleep


STAGE_NAMES = {
    "mission": "planification de la mission",
    "review_mission": "revue de la mission",
    "catalog": "recherche bibliographique OpenAlex",
    "literature": "synthèse bibliographique",
    "review_literature": "revue de la synthèse",
    "resource_selection": "sélection et préparation des ressources",
    "capability_resolution": "diagnostic et résolution d'une capacité manquante",
    "protocol": "préparation du protocole",
    "review_protocol": "revue du protocole",
    "implementation": "préparation et vérification du programme",
    "protocol_execution": "exécution du plan expérimental",
    "synthetic_execution": "calcul synthétique",
    "ood_execution": "répétition OOD simulée",
    "ood_gpu_execution": "calibration OOD sur GPU",
    "retry_gpu_execution": "exécution du plan apparié sur GPU",
    "calibration": "calibration du pilote",
    "measurement": "mesure du pilote",
    "draft": "analyse, critique et rédaction du brouillon",
    "review_manuscript": "revue et export du manuscrit",
}

BLOCK_REASONS = {
    "InsufficientLiteratureError": "catalogue bibliographique insuffisant",
    "invalid_model_output": "sortie du modèle invalide",
    "OpenAlexRateLimitError": "limite OpenAlex atteinte",
    "calibration_did_not_pass": "critère de calibration non satisfait",
    "no_compatible_executor": "exécuteur compatible introuvable",
    "capability_unresolved": "prérequis manquant",
    "invalid_capability_resolution": "proposition de résolution invalide",
}


def _message(event):
    action, state = event["action"], event["state"]
    detail = event["detail"] or {}
    stage = STAGE_NAMES.get(detail.get("stage", state["stage"]), state["stage"])
    if action == "start":
        return f"Mission {state['id']} initialisée."
    if action == "authorize_autonomy":
        return "Autorisation de mission autonome enregistrée."
    if action == "reserve":
        return f"Étape réservée · {stage}."
    if action == "autonomy_pause":
        return f"Autonomie suspendue · {detail['reason']}"
    if state.get("block"):
        block = state["block"]
        reason = ("état de l’opération à réconcilier" if block["uncertain"]
                  else BLOCK_REASONS.get(block["reason"], "contrôle en échec"))
        return f"Incident · {stage} · {reason}."
    if state["status"] == "complete":
        if state.get("workload") == "retry_gpu" and state.get("report"):
            return f"Essais GPU terminés · rapport {state['report']['path']}."
        export = state.get("export")
        return f"Mission terminée · export {export}" if export else "Mission terminée."
    if action == "result":
        if detail["stage"] == "catalog":
            count = len(state["artifacts"]["catalog"]["papers"])
            return f"Recherche OpenAlex terminée · {count} notices."
        if detail["stage"] == "ood_gpu_execution":
            return "Lot GPU terminé · reçu enregistré."
        if detail["stage"] == "retry_gpu_execution":
            return "Essais GPU terminés · rapport vérifié."
        return f"Étape terminée · {stage}."
    if action in {"revise", "approve_revision"}:
        return f"Révision · retour à l’étape « {stage} »."
    if action == "prepare_protocol":
        return "Protocole prêt."
    if action == "prepare_draft":
        return "Analyse et critique terminées · brouillon prêt."
    if action.startswith("recover_"):
        return "Reprise depuis les résultats enregistrés."
    return f"Point de reprise enregistré · {stage}."


def watch_progress(store, *, once=False):
    seen = 0
    last_digest = None
    try:
        while True:
            history = store.history()
            if not history:
                raise FileNotFoundError("Aucune mission initialisée dans ce dossier.")
            if seen and (len(history) < seen or history[seen - 1]["sha256"] != last_digest):
                raise ValueError("Le journal a changé pendant le suivi.")
            for index, event in enumerate(history[seen:], start=seen + 1):
                print(f"[{index:02d} · v{event['state']['version']}] {_message(event)}", flush=True)
            if once or history[-1]["state"]["status"] == "complete":
                return
            seen = len(history)
            last_digest = history[-1]["sha256"]
            sleep(1)
    except KeyboardInterrupt:
        return
