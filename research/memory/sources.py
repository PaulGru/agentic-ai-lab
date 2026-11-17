"""Deterministic projections of checked artifacts; no LLM writes to memory."""
from __future__ import annotations

from analysis.contracts import canonical_json, fingerprint
from research.critique.contracts import DesignContext, RevisionProposal, ReviewInput, ReviewReport
from research.critique.evidence import load_input, read_document, validate_context
from research.critique.revision import verify_revision
from research.critique.store import ReviewStore
from research.experimental_design import validate_approved_literature
from research.literature import validate_approved_mission
from research.memory.contracts import MemoryItem, SourceOrigin, SourceSnapshot
from research.memory.stores import terms


def pointer(document, path: str):
    import re
    if not path.startswith("/") or len(path) > 400:
        raise ValueError("Invalid memory source pointer")
    current = document
    try:
        for raw in path[1:].split("/"):
            if re.search(r"~(?![01])", raw):
                raise ValueError("Malformed pointer escape")
            key = raw.replace("~1", "/").replace("~0", "~")
            if isinstance(current, list):
                if not key.isascii() or not key.isdigit() or (len(key) > 1 and key.startswith("0")):
                    raise ValueError("Invalid list reference")
                current = current[int(key)]
            elif isinstance(current, dict):
                current = current[key]
            else:
                raise ValueError("Reference traverses a scalar")
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Memory reference does not resolve") from error
    return current


def checked_review(path: str, review_id: str):
    store = ReviewStore(path, read_only=True)
    request = ReviewInput.model_validate(store.require("session", review_id))
    bundle = load_input(request)
    if bundle.model_dump(mode="json") != store.require("bundle", review_id):
        raise ValueError("Original review evidence changed")
    # Reconstruct opinions from their durable receipts, not from an asserted report.
    from research.critique.graph import ReviewWorkflow
    report = ReviewReport.model_validate(store.require("review", review_id))
    workflow = ReviewWorkflow(store)
    if workflow._report(request, bundle) != report:
        raise ValueError("Review differs from its checked sources and LLM receipts")
    return store, workflow, request, bundle, report


def load_origin(origin: SourceOrigin, *, note_document: dict | None = None) -> dict:
    if origin.kind == "note":
        if (not isinstance(note_document, dict) or set(note_document) != {"title", "text", "logical_key", "actor", "tags"}
                or any(not isinstance(note_document[k], str) or not note_document[k].strip()
                       for k in ("title", "text", "logical_key", "actor"))):
            raise ValueError("A note is an explicit unverified, attributed document")
        return note_document
    if origin.kind == "mission":
        doc = read_document(origin.path)
        validate_approved_mission(doc)
        return doc
    if origin.kind == "literature":
        mission, literature = read_document(origin.parent_path), read_document(origin.path)
        validate_approved_literature(mission, literature)
        return {"mission_document": mission, "literature_document": literature}
    if origin.kind == "context":
        document = ({"mission_document": read_document(origin.parent_path),
                     "literature_document": read_document(origin.literature_path),
                     "design_document": read_document(origin.path), "matrix_document": read_document(origin.matrix_path)}
                    if origin.parent_path else read_document(origin.path))
        context = DesignContext.model_validate(document)
        validate_context(context)
        return context.model_dump(mode="json")
    store, workflow, request, bundle, report = checked_review(origin.path, origin.review_id)
    root = {"bundle": bundle.model_dump(mode="json"), "report": report.model_dump(mode="json")}
    if origin.kind == "decision":
        decision = store.require("decision", origin.artifact_id)
        if decision.get("kind") != "revision_decision":
            raise ValueError("Only historical protocol decisions are indexed, never API authorizations")
        # Locate the exact proposal by digest, not by untrusted text or filenames.
        matches = [a["document"] for a in store.artifacts(origin.review_id)
                   if a["kind"] == "proposal" and a["sha256"] == decision.get("target_sha256")]
        if len(matches) != 1:
            raise ValueError("Decision has no unique matching protocol proposal")
        proposal = RevisionProposal.model_validate(matches[0])
        verify_revision(bundle, report, proposal, workflow._manager(request))
        if proposal.review_id != origin.review_id or decision.get("decision") not in {"approve", "reject", "request_changes"}:
            raise ValueError("Decision/review lineage mismatch")
        root.update(decision=decision, proposal=proposal.model_dump(mode="json"))
    return root


def snapshot(project_id: str, origin: SourceOrigin, document: dict) -> SourceSnapshot:
    digest = fingerprint(document)
    identity = {"project_id": project_id, "origin": origin.model_dump(mode="json"), "document_sha256": digest}
    return SourceSnapshot(source_id=fingerprint(identity), project_id=project_id, origin=origin,
                          document=document, document_sha256=digest)


def project_items(source: SourceSnapshot) -> list[MemoryItem]:
    """Only these checked projections can become source-checked memories."""
    root, origin = source.document, source.origin
    result = []

    def add(kind, key, title, path, scope, verification="source_checked", tags=(), text=None):
        value = pointer(root, path)
        data = {"schema_version": "8.7", "project_id": source.project_id, "logical_key": key,
                "kind": kind, "title": title[:240],
                "text": (text if text is not None else canonical_json(value))[:4000],
                "tags": list(dict.fromkeys(tags)), "source_id": source.source_id,
                "source_pointer": path, "value_sha256": fingerprint(value), "scope": scope,
                "verification": verification,
                "initial_status": "tentative" if verification.startswith("unverified_") else "recorded",
                "scientific_evidence": False, "action_authorization": False}
        result.append(MemoryItem(memory_id=fingerprint(data), **data))

    def mission(prefix, document):
        mission_sha = document["mission_sha256"]
        question = document["mission"]["research_question"]
        add("project_plan", f"mission:{mission_sha}", "Question de recherche — " + question,
            prefix + "/mission", "planning", tags=("mission", "planning"))
        for i, hyp in enumerate(document["mission"]["hypotheses"]):
            add("hypothesis", f"hypothesis:{mission_sha}:{hyp['id']}", "Hypothèse à tester — " + hyp["id"],
                prefix + f"/mission/hypotheses/{i}", "planning", tags=("hypothesis", hyp["id"], *terms(question)[:12]))

    def literature(prefix, document):
        for i, finding in enumerate(document["review"]["findings"]):
            paper = next(p for p in document["catalog"]["papers"] if p["work_id"] == finding["work_id"])
            add("literature", f"abstract:{finding['work_id']}", "Résumé — " + finding["work_id"] + " — " + paper["title"],
                prefix + f"/review/findings/{i}", "abstract_only", "abstract_excerpt_checked",
                tags=("literature", "abstract", finding["work_id"], *terms(paper["title"])[:10]))

    if origin.kind == "note":
        add("lesson", "note:" + root["logical_key"], root["title"], "/text", "human_note",
            "unverified_note", tags=tuple(root["tags"]), text=root["text"])
    elif origin.kind == "mission":
        mission("", root)
    elif origin.kind in {"literature", "context"}:
        mission("/mission_document", root["mission_document"])
        literature("/literature_document", root["literature_document"])
        if origin.kind == "context":
            doc = root["design_document"]
            add("project_plan", "design:" + doc["mission_sha256"], "Protocole approuvé pour conception, non exécuté",
                "/design_document/design/analysis_plan", "planning", tags=("design", "protocol", "analysis"))
    elif origin.kind == "review":
        scope = "pedagogical" if root["bundle"]["source_kind"] == "pedagogical_paired_fixture" else "engineering"
        for ref in root["report"]["observation_refs"]:
            path = "/bundle" + ref
            group = ref.rsplit("/", 1)[-1]
            metric = root["bundle"]["source_document"].get("metric_name", "synthetic.mean")
            add("observation", f"observation:{origin.review_id}:{group}", "Observation synthétique — " + group,
                path, scope, tags=("observation", "synthetic", group, metric))
        for i, issue in enumerate(root["report"]["issues"]):
            verification = "source_checked" if issue["origin"] == "deterministic" else "unverified_model_opinion"
            add("lesson", f"issue:{origin.review_id}:{issue['issue_id']}", issue["issue_id"], f"/report/issues/{i}",
                scope, verification, tags=(issue["category"], "critique", issue["issue_id"]),
                text=issue["finding"] + "\n" + issue["recommendation"])
        context = root["bundle"].get("context")
        if context:
            mission("/bundle/context/mission_document", context["mission_document"])
            literature("/bundle/context/literature_document", context["literature_document"])
            add("project_plan", "design:" + context["mission_document"]["mission_sha256"],
                "Protocole examiné — conception non exécutée", "/bundle/context/design_document/design/analysis_plan",
                "planning", tags=("design", "protocol", "protocole", "analysis"))
    else:
        decision = root["decision"]
        aliases = {"reject": ("rejet", "rejete", "refus", "refuse", "protocole"),
                   "approve": ("approbation", "adoption", "protocole"),
                   "request_changes": ("revision", "modification", "protocole")}[decision["decision"]]
        context = root["bundle"].get("context")
        question = context["mission_document"]["mission"]["research_question"] if context else ""
        add("decision", f"decision:{origin.review_id}:{origin.artifact_id}",
            "Décision historique — " + decision["decision"], "/decision", "planning",
            tags=("decision", decision["decision"], "protocol", *aliases),
            text=question + "\nDécision historique, sans autorisation future : " + canonical_json(decision))
    return result
