"""Readable snapshots and exclusive exports; SQLite remains authoritative."""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path

from analysis.contracts import canonical_json, fingerprint
from research.critique.contracts import EvidenceBundle, ReviewReport
from research.critique.evidence import resolve_ref
from research.critique.rules import check_report_bindings
from research.critique.store import ReviewStore


def render_review(bundle: EvidenceBundle, report: ReviewReport) -> str:
    check_report_bindings(bundle, report)

    def safe(value):
        return html.escape(str(value)).replace("\n", " ").replace("|", "\\|")

    lines = ["# V8.6 — Critique et révision", "",
             "**Sources synthétiques — aucune conclusion scientifique justifiée.**", "",
             "## Observations calculées", ""]
    for ref in report.observation_refs:
        lines.append(f"- {safe(ref)} : {safe(canonical_json(resolve_ref(bundle, ref)))}")
    lines.extend(["", "## Hypothèses à tester", ""])
    for ref in report.hypothesis_refs:
        lines.append(f"- {safe(canonical_json(resolve_ref(bundle, ref)))}")
    if not report.hypothesis_refs:
        lines.append("Aucune mission fournie ; aucune hypothèse scientifique n'est reconstruite.")
    lines.extend(["", "## Constats et remarques", ""])
    for issue in report.issues:
        lines.extend([
            f"### {safe(issue.issue_id)} — {safe(issue.severity)}", "",
            f"Origine : {safe(issue.origin)} ; statut : {safe(issue.basis)}.", "",
            safe(issue.finding), "", f"Recommandation : {safe(issue.recommendation)}", "",
            "Références : " + ", ".join(safe(ref) for ref in issue.evidence_refs), "",
        ])
    if report.claim_assessments:
        lines.extend(["## Assertions examinées", ""])
        for assessment in report.claim_assessments:
            lines.append(f"- {safe(assessment.claim_id)} : {safe(assessment.status)} — {safe(assessment.reason)}")
    lines.extend(["", "## Conclusion et traçabilité", "", safe(report.conclusion), "",
                  f"Dossier SHA-256 : {bundle.fingerprint()}", f"Rapport SHA-256 : {report.fingerprint()}", "",
                  "Un checksum vérifie la cohérence du contenu, pas l'identité de l'auteur.",
                  "L'export est un instantané ; la provenance doit être revérifiée contre les sources originales.", ""])
    return "\n".join(lines)


def export_review(store: ReviewStore, review_id: str, destination: str | Path) -> Path:
    bundle = EvidenceBundle.model_validate(store.require("bundle", review_id))
    report = ReviewReport.model_validate(store.require("review", review_id))
    check_report_bindings(bundle, report)
    package = {"schema_version": "8.6", "kind": "review_snapshot", "review_id": review_id,
               "artifacts": store.artifacts(review_id), "calls": store.call_summaries(review_id),
               "source_provenance_rechecked": False}
    outputs = {
        "review_package.json": (json.dumps(package, sort_keys=True, ensure_ascii=False,
                                          allow_nan=False, indent=2) + "\n").encode("utf-8"),
        "review.md": render_review(bundle, report).encode("utf-8"),
    }
    context = store.get("memory_context", review_id)
    if context is not None:
        from research.memory.contracts import MemoryContext
        context = MemoryContext.model_validate(context)
        lines = ["", "## Mémoire de recherche consultée", "",
                 "Instantané historique ; ces souvenirs ne constituent aucune autorisation ni preuve scientifique.", ""]
        for hit in context.hits:
            lines.append(f"- {html.escape(hit.title)} — {hit.memory_id} ({hit.scope}, {hit.status})")
            lines.append("  " + html.escape(hit.excerpt).replace("\n", " "))
        outputs["review.md"] += ("\n".join(lines) + "\n").encode("utf-8")
    manifest = {"schema_version": "8.6", "complete": True,
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in outputs.items()}}
    outputs["manifest.json"] = (canonical_json(manifest) + "\n").encode("utf-8")
    root = Path(destination)
    # Preflight every output. A divergent/partial file is never silently repaired.
    for name, data in outputs.items():
        path = root / name
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError("Export differs or is incomplete; choose a new destination")
    root.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        path = root / name
        if not path.exists():
            with path.open("xb") as target:
                target.write(data)
                target.flush()
                os.fsync(target.fileno())
    return root / "manifest.json"  # written last, only after all payloads


def inspect_export(destination: str | Path) -> dict:
    root = Path(destination)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "8.6" or manifest.get("complete") is not True
            or set(manifest.get("files", {})) != {"review_package.json", "review.md"}):
        raise ValueError("Invalid review export manifest")
    for name, digest in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Review export checksum mismatch")
    package = json.loads((root / "review_package.json").read_text(encoding="utf-8"))
    if package.get("schema_version") != "8.6" or package.get("kind") != "review_snapshot":
        raise ValueError("Invalid review snapshot")
    for artifact in package["artifacts"]:
        if fingerprint(artifact["document"]) != artifact["sha256"]:
            raise ValueError("Review snapshot artifact checksum mismatch")
    return package
