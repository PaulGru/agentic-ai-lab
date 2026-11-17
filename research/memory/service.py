"""Project-scoped research memory with source revalidation and retrieval audit."""
from __future__ import annotations

from pathlib import Path
import re

from analysis.contracts import ID, canonical_json, fingerprint
from research.memory.contracts import MemoryContext, MemoryEvent, MemoryHit, MemoryItem, SourceOrigin
from research.memory.sources import load_origin, pointer, project_items, snapshot
from research.memory.stores import SQLMemoryStore, search_text, terms


class ResearchMemory:
    def __init__(self, store: SQLMemoryStore):
        self.store = store

    @staticmethod
    def project(project_id):
        if not isinstance(project_id, str) or not re.fullmatch(ID, project_id):
            raise ValueError("An explicit valid project identifier is required")

    def ingest(self, project_id: str, origin: SourceOrigin) -> list[MemoryItem]:
        self.project(project_id)
        source = snapshot(project_id, origin, load_origin(origin))
        items = project_items(source)
        self.store.ingest(source, items)
        return items

    def ingest_review(self, project_id: str, path: str | Path, review_id: str) -> list[MemoryItem]:
        SourceOrigin(kind="review", path=str(path), review_id=review_id)  # reject UNC/URLs before resolving
        path = str(Path(path).resolve())
        items = self.ingest(project_id, SourceOrigin(kind="review", path=path, review_id=review_id))
        # Append-only decisions can arrive after the report; import each separately.
        from research.critique.store import ReviewStore
        store = ReviewStore(path, read_only=True)
        for artifact in store.artifacts(review_id):
            if artifact["kind"] == "decision" and artifact["document"].get("kind") == "revision_decision":
                items.extend(self.ingest(project_id, SourceOrigin(kind="decision", path=path, review_id=review_id,
                                                                  artifact_id=artifact["artifact_id"])))
        return items

    def note(self, project_id: str, *, title: str, text: str, logical_key: str, actor: str,
             tags: list[str] | None = None) -> MemoryItem:
        self.project(project_id)
        document = {"title": title, "text": text, "logical_key": logical_key, "actor": actor, "tags": tags or []}
        origin = SourceOrigin(kind="note")
        source = snapshot(project_id, origin, load_origin(origin, note_document=document))
        items = project_items(source)
        # The record is bounded without silently truncating a human note.
        if len(text) > 4000 or len(title) > 240:
            raise ValueError("Memory note exceeds the title/text limit")
        self.store.ingest(source, items)
        return items[0]

    def _checked(self, project_id: str, memory_id: str, validated_sources=None, *, verify_external=True):
        item = self.store.item(project_id, memory_id)
        if item is None:
            raise ValueError("Memory not found in this project")
        source = self.store.source(project_id, item.source_id)
        if source is None:
            raise ValueError("Memory source missing")
        if verify_external and (validated_sources is None or source.source_id not in validated_sources):
            current = load_origin(source.origin, note_document=source.document)
            if fingerprint(current) != source.document_sha256:
                raise SourceChanged("Memory source has changed since indexing")
            if validated_sources is not None:
                validated_sources.add(source.source_id)
        if fingerprint(pointer(source.document, item.source_pointer)) != item.value_sha256:
            raise ValueError("Memory reference/value checksum mismatch")
        if item not in project_items(source):
            raise ValueError("Memory is not a deterministic projection of its source")
        events = self.store.events(project_id, memory_id)
        status = item.initial_status
        for event in events:
            if event.target_sha256 != item.fingerprint():
                raise ValueError("Memory lifecycle targets another record version")
            status = {"confirm": "human_reviewed", "invalidate": "invalidated", "supersede": "superseded"}[event.action]
        return item, source, status, fingerprint([e.model_dump(mode="json") for e in events])

    def get(self, project_id: str, memory_id: str, *, verify_source: bool = True) -> dict:
        self.project(project_id)
        item, source, status, lifecycle = self._checked(project_id, memory_id, verify_external=verify_source)
        return {"item": item.model_dump(mode="json"), "status": status, "lifecycle_sha256": lifecycle,
                "source": source.model_dump(mode="json"), "source_provenance_rechecked": verify_source,
                "events": [event.model_dump(mode="json") for event in self.store.events(project_id, memory_id)]}

    def change(self, project_id: str, memory_id: str, *, action: str, target_sha256: str,
               actor: str, reason: str, replacement_id: str | None = None) -> MemoryEvent:
        self.project(project_id)
        item = self.store.item(project_id, memory_id)
        if item is None or item.fingerprint() != target_sha256:
            raise PermissionError("Human decision does not target this project/memory version")
        # Invalidation remains possible even if a source has disappeared or is corrupt.
        if action == "confirm":
            self._checked(project_id, memory_id)
        if replacement_id:
            _, _, replacement_status, _ = self._checked(project_id, replacement_id)
            if replacement_status not in {"recorded", "human_reviewed"}:
                raise ValueError("Replacement must be active and reviewed/source-checked")
        data = {"project_id": project_id, "memory_id": memory_id, "target_sha256": target_sha256,
                "action": action, "replacement_id": replacement_id, "actor": actor, "reason": reason}
        event = MemoryEvent(event_id=fingerprint(data), **data)
        self.store.append_event(event)
        return event

    def search(self, project_id: str, query: str, *, role: str = "research", limit: int = 5,
               max_bytes: int = 12_000, audit: bool = True, exclude_review_id: str | None = None,
               kinds: list[str] | None = None, scopes: list[str] | None = None) -> MemoryContext:
        self.project(project_id)
        if (not isinstance(query, str) or not 1 <= len(query.strip()) <= 500
                or type(limit) is not int or not 1 <= limit <= 10
                or type(max_bytes) is not int or not 2000 <= max_bytes <= 24_000):
            raise ValueError("Invalid or unbounded research memory query")
        query = query.strip()
        query_terms = terms(query)[:12]
        checked_query = MemoryContext(project_id=project_id, role=role, query=query, terms=query_terms,
                                      hits=[], kinds=kinds or [], scopes=scopes or [])
        candidates = self.store.candidates(project_id, query_terms, kinds=checked_query.kinds, scopes=checked_query.scopes)
        candidates.sort(key=lambda item: (-len(set(query_terms) & set(terms(search_text(item)))), item.memory_id))
        hits, omitted, validated, versions = [], [], set(), set()

        def context():
            return MemoryContext(project_id=project_id, role=role, query=query, terms=query_terms,
                                 hits=hits, omitted=omitted, kinds=checked_query.kinds, scopes=checked_query.scopes)

        for item in candidates:
            source = self.store.source(project_id, item.source_id)
            if source and exclude_review_id and source.origin.review_id == exclude_review_id:
                continue
            # Lifecycle exclusions are independent of whether old source files still exist.
            events = self.store.events(project_id, item.memory_id)
            inactive = events and events[-1].action in {"invalidate", "supersede"}
            tentative = item.initial_status == "tentative" and not events
            if inactive or tentative:
                omitted.append({"memory_id": item.memory_id, "reason": "inactive" if inactive else "tentative"})
                continue
            try:
                item, source, status, lifecycle = self._checked(project_id, item.memory_id, validated)
            except (FileNotFoundError, OSError, KeyError, SourceChanged) as error:
                omitted.append({"memory_id": item.memory_id,
                                "reason": "source_changed" if isinstance(error, SourceChanged) else "source_unavailable"})
                continue
            score = len(set(query_terms) & set(terms(search_text(item)))) / max(1, len(query_terms))
            if not score:
                continue
            version = (item.kind, item.logical_key, item.value_sha256)
            if version in versions:
                continue
            hit = MemoryHit(memory_id=item.memory_id, item_sha256=item.fingerprint(), source_id=source.source_id,
                            source_sha256=source.document_sha256, source_pointer=item.source_pointer,
                            value_sha256=item.value_sha256, title=item.title, excerpt=item.text[:700],
                            kind=item.kind, scope=item.scope, verification=item.verification,
                            status=status, lifecycle_sha256=lifecycle, score=score)
            hits.append(hit)
            if len(canonical_json(context().model_dump(mode="json")).encode("utf-8")) > max_bytes:
                hits.pop()
                omitted.append({"memory_id": item.memory_id, "reason": "context_budget"})
            else:
                versions.add(version)
            if len(hits) >= limit:
                break
        while len(canonical_json(context().model_dump(mode="json")).encode("utf-8")) > max_bytes and omitted:
            omitted.pop()
        result = context()
        if audit:
            self.store.audit(result)
        return result

    def verify_context(self, context: MemoryContext) -> None:
        recorded = self.store.retrieval(context.project_id, context.fingerprint())
        if recorded != context:
            raise ValueError("Retrieved memory context has no matching immutable audit record")
        if context.terms != terms(context.query)[:12]:
            raise ValueError("Memory query terms changed")
        for hit in context.hits:
            item, source, status, lifecycle = self._checked(context.project_id, hit.memory_id)
            if (status not in {"recorded", "human_reviewed"} or item.fingerprint() != hit.item_sha256
                    or source.document_sha256 != hit.source_sha256 or lifecycle != hit.lifecycle_sha256
                    or item.text[:700] != hit.excerpt or item.scope != hit.scope
                    or item.verification != hit.verification or item.kind != hit.kind
                    or (context.kinds and item.kind not in context.kinds)
                    or (context.scopes and item.scope not in context.scopes)
                    or status != hit.status or item.title != hit.title or item.source_pointer != hit.source_pointer
                    or item.value_sha256 != hit.value_sha256 or source.source_id != hit.source_id):
                raise ValueError("Selected memory/source/lifecycle changed after retrieval")


class SourceChanged(ValueError):
    pass
