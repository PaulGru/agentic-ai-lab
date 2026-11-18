"""Deterministic findings first; model opinions cannot remove or upgrade them."""
from __future__ import annotations

import math

from analysis.retry_pilot import RetryPilotAnalysis
from research.critique.contracts import (
    ClaimAssessment, CriticSuggestions, CritiqueIssue, EvidenceBundle, PILOT_CONCLUSION, ReviewReport,
)
from research.critique.evidence import EvidenceLedger, resolve_ref, verify_evidence


def _review(bundle: EvidenceBundle) -> ReviewReport:
    issues = []

    def issue(code, category, severity, finding, recommendation, refs, human=False):
        issues.append(CritiqueIssue(
            issue_id=code, origin="deterministic", category=category, severity=severity,
            basis="verified_check", finding=finding, recommendation=recommendation,
            evidence_refs=refs, requires_human_review=human))

    observation_refs = []
    retry_pilot = isinstance(bundle.analysis, RetryPilotAnalysis)
    if retry_pilot:
        for index, code in enumerate(("pilot_sample", "fault_profile", "descriptive_scope")):
            recommendations = (
                "Élargir le corpus et répéter les essais sur plusieurs versions de modèles.",
                "Ajouter un profil de panne persistante et mesurer les tentatives inutiles.",
                "Fixer le plan d'analyse de la future campagne avant d'examiner ses données.",
            )
            issue(code, "methodology", "major", bundle.analysis.limitations[index], recommendations[index],
                  [f"/analysis/limitations/{index}"], True)
        for profile in ("clean", "transient"):
            for policy in ("no_retry", "retry_once"):
                observation_refs.append(f"/analysis/comparison/{profile}/arms/{policy}")
    else:
        issue("engineering_scope", "scope", "major",
              "Les sources actuelles sont synthétiques ; elles ne justifient aucune conclusion scientifique.",
              "Préserver cette limite et définir une véritable campagne avant toute généralisation.",
              ["/scientific_evidence", "/analysis/validity/ready_for_scientific_inference"])
        for group in sorted(bundle.analysis.descriptive.groups):
            safe_group = group.replace("~", "~0").replace("/", "~1")
            observation_refs.append(f"/analysis/descriptive/groups/{safe_group}")
    if bundle.source_kind == "verified_pilot_aggregates":
        issue("no_replication", "statistics", "major",
              "Chaque job fournit un agrégat ; le nombre de valeurs entrées ne mesure pas la réplication indépendante.",
              "Définir les unités expérimentales et les répétitions avant de prévoir une inférence.",
              ["/dataset_document/unit_of_observation", "/dataset_document/observations"])
        if any(obs["simulated"] for obs in bundle.dataset_document["observations"]):
            issue("simulated_transport", "provenance", "major",
                  "Certaines observations proviennent d'un transport de test simulé.",
                  "Présenter ces sorties comme des tests de l'infrastructure.",
                  ["/dataset_document/observations"])
        issue("runtime_not_pinned", "provenance", "minor",
              "Le contrat du pilote ne conserve pas un digest immuable de l'image Docker exécutée.",
              "Documenter cette limite de reproductibilité ; vérifier l'environnement avant une campagne réelle.",
              ["/source_document/schema_version"], True)
    elif not retry_pilot:
        issue("assumed_independence", "methodology", "major",
              "L'indépendance et la symétrie des différences ne sont pas démontrées par une fixture JSON.",
              "Justifier ces hypothèses sur un futur processus de collecte ; conserver ici le statut pédagogique.",
              ["/source_document/sign_flip_symmetry_assumed", "/analysis/validity/cautions"], True)
        if not bundle.analysis.validity.ready_for_pedagogical_inference:
            issue("pedagogical_inference_blocked", "statistics", "major",
                  "La porte de validité refuse même l'inférence pédagogique sur cette fixture.",
                  "Examiner les blocages sans traiter le seuil pédagogique comme une analyse de puissance.",
                  ["/analysis/validity/blockers"])
    hypothesis_refs = []
    if bundle.context:
        issue("unlinked_design", "provenance", "major",
              "Aucun lien vérifié ne rattache ces résultats au design V8.3 fourni.",
              "Limiter les révisions à une revue documentaire du design ; ne pas attribuer les résultats à ses traitements.",
              ["/design_execution_link", "/context/design_document/design_sha256"])
        issue("abstract_only", "bibliography", "minor",
              "La littérature fournie contrôle des extraits de résumés, sans vérification des articles intégraux.",
              "Lire et vérifier les méthodes pertinentes avant d'affirmer la nouveauté ou la validité du protocole.",
              ["/context/literature_document/verification_level"], True)
        issue("feasibility_unverified", "methodology", "minor",
              "Les accès et licences du jeu de tâches ne sont pas déclarés vérifiés dans le design.",
              "Conserver les vérifications humaines de faisabilité avant toute exécution.",
              ["/context/design_document/design/task_suite"])
        hypothesis_refs = [f"/context/mission_document/mission/hypotheses/{i}"
                           for i in range(len(bundle.context.mission_document["mission"]["hypotheses"]))]
    assessments = []
    for index, claim in enumerate(bundle.claims):
        status, reason = "unsupported", "Aucune observation numérique vérifiable ne soutient cette assertion."
        if claim.kind == "hypothesis":
            status, reason = "hypothesis_only", "Une hypothèse reste à tester ; elle n'est pas un résultat."
        elif claim.kind == "scientific_conclusion":
            status = "outside_scientific_scope"
            reason = ("Ce pilote contrôlé ne permet pas cette conclusion scientifique." if retry_pilot else
                      "Les sources V8.5 actuelles interdisent les conclusions scientifiques.")
        elif claim.claimed_value is not None and len(claim.evidence_refs) == 1:
            try:
                actual = resolve_ref(bundle, claim.evidence_refs[0])
                # Only source measurements / computed statistics count as observations.
                allowed = claim.evidence_refs[0].startswith(("/analysis/descriptive/", "/analysis/inference/",
                                                            "/dataset_document/observations/", "/source_document/pairs/"))
                if retry_pilot:
                    allowed = claim.evidence_refs[0].startswith("/analysis/comparison/")
                if (allowed and type(actual) in (int, float) and math.isfinite(actual)
                        and actual == claim.claimed_value):
                    status, reason = "supported_numeric", "Valeur numérique concordante ; l'interprétation du texte reste à relire."
            except ValueError:
                pass
        assessments.append(ClaimAssessment(claim_id=claim.claim_id, status=status, reason=reason))
        if status in {"unsupported", "outside_scientific_scope"}:
            issue(f"claim_{index}_unsupported", "claims", "major", reason,
                  "Retirer ou reformuler cette assertion dans le périmètre des preuves disponibles.",
                  [f"/claims/{index}"])
    return ReviewReport(
        bundle_sha256=bundle.fingerprint(), analysis_sha256=bundle.analysis.fingerprint(),
        observation_refs=observation_refs, hypothesis_refs=hypothesis_refs,
        issues=issues, claim_assessments=assessments,
        **({"conclusion": PILOT_CONCLUSION} if retry_pilot else {}))


def deterministic_review(bundle: EvidenceBundle, manager: EvidenceLedger | None = None) -> ReviewReport:
    verify_evidence(bundle, manager)
    return _review(bundle)


def with_suggestions(bundle: EvidenceBundle, report: ReviewReport,
                     suggestions: CriticSuggestions) -> ReviewReport:
    check_report_bindings(bundle, report)
    if any(item.origin == "llm" for item in report.issues):
        raise ValueError("LLM critique was already attached to this review")
    extra = []
    for i, item in enumerate(suggestions.issues):
        for ref in item.evidence_refs:
            resolve_ref(bundle, ref)
        extra.append(CritiqueIssue(
            issue_id=f"llm_{i + 1}", origin="llm", basis="potential_concern",
            requires_human_review=True, **item.model_dump(mode="json")))
    result = ReviewReport.model_validate({**report.model_dump(mode="json"),
                                         "issues": [x.model_dump(mode="json") for x in report.issues + extra]})
    check_report_bindings(bundle, result)
    return result


def check_report_bindings(bundle: EvidenceBundle, report: ReviewReport) -> None:
    """Snapshot consistency only; caller revalidates original sources separately."""
    baseline = _review(bundle)
    if report.model_dump(exclude={"issues"}) != baseline.model_dump(exclude={"issues"}):
        raise ValueError("Review facts, hypotheses, scope or source binding were changed")
    deterministic = [x for x in report.issues if x.origin == "deterministic"]
    if deterministic != baseline.issues:
        raise ValueError("Deterministic findings may not be removed or modified")
    for issue in report.issues:
        for ref in issue.evidence_refs:
            resolve_ref(bundle, ref)


def verify_review(bundle: EvidenceBundle, report: ReviewReport,
                  manager: EvidenceLedger | None = None) -> None:
    verify_evidence(bundle, manager)
    check_report_bindings(bundle, report)
