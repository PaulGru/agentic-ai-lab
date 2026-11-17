"""SQLite audit log, separate from job execution and LangGraph checkpoints.

Artifacts are append-only. Call reservations precede external effects. A reserved
or uncertain call is never retried automatically, even after process restart.
The local database is a trust boundary, not a cryptographic identity service.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from analysis.contracts import canonical_json, fingerprint
from research.critique.contracts import LLMRequest, ReviewInput


class NeedsReconciliation(RuntimeError):
    pass


class ReviewStore:
    def __init__(self, path: str | Path, *, read_only: bool = False):
        self.path = Path(path)
        self.read_only = read_only
        if read_only:
            if not self.path.is_file():
                raise FileNotFoundError("Existing review database required")
            with self._connect() as db:
                db.execute("SELECT kind FROM artifacts LIMIT 0")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS artifacts (
                kind TEXT NOT NULL, artifact_id TEXT NOT NULL, review_id TEXT NOT NULL,
                document TEXT NOT NULL, sha256 TEXT NOT NULL,
                PRIMARY KEY(kind, artifact_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS calls (
                call_id TEXT PRIMARY KEY, review_id TEXT NOT NULL,
                request_json TEXT NOT NULL, request_sha256 TEXT NOT NULL,
                phase TEXT NOT NULL, result_json TEXT, result_sha256 TEXT,
                error_type TEXT)""")

    @contextmanager
    def _connect(self):
        if self.read_only:
            db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
            db.execute("PRAGMA query_only=ON")
        else:
            db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _write(self):
        if self.read_only:
            raise PermissionError("Review store is read-only")

    @staticmethod
    def _decode(text: str, digest: str) -> dict:
        document = json.loads(text)
        if not isinstance(document, dict) or fingerprint(document) != digest:
            raise ValueError("Persisted review artifact checksum mismatch")
        return document

    def put(self, kind: str, artifact_id: str, review_id: str, document: dict) -> str:
        self._write()
        if kind not in {"session", "bundle", "review", "proposal", "decision", "approved_revision", "memory_binding", "memory_context"}:
            raise ValueError("Unknown review artifact kind")
        text, digest = canonical_json(document), fingerprint(document)
        if len(text.encode("utf-8")) > 1_048_576:
            raise ValueError("Review artifact exceeds 1 MiB")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT review_id, document, sha256 FROM artifacts WHERE kind=? AND artifact_id=?",
                             (kind, artifact_id)).fetchone()
            if row:
                self._decode(row[1], row[2])
                if row != (review_id, text, digest):
                    raise ValueError("Existing review artifact differs; choose a new identifier")
            else:
                db.execute("INSERT INTO artifacts VALUES (?, ?, ?, ?, ?)",
                           (kind, artifact_id, review_id, text, digest))
        return digest

    def get(self, kind: str, artifact_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("SELECT document, sha256 FROM artifacts WHERE kind=? AND artifact_id=?",
                             (kind, artifact_id)).fetchone()
        return self._decode(*row) if row else None

    def require(self, kind: str, artifact_id: str) -> dict:
        value = self.get(kind, artifact_id)
        if value is None:
            raise KeyError(f"Missing {kind} artifact")
        return value

    def artifacts(self, review_id: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT kind, artifact_id, document, sha256 FROM artifacts WHERE review_id=? ORDER BY kind, artifact_id",
                              (review_id,)).fetchall()
        return [{"kind": k, "artifact_id": i, "sha256": s, "document": self._decode(t, s)}
                for k, i, t, s in rows]

    def call(self, call_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute("""SELECT request_json, request_sha256, phase,
                result_json, result_sha256, error_type FROM calls WHERE call_id=?""", (call_id,)).fetchone()
        if row is None:
            return None
        request = self._decode(row[0], row[1])
        result = self._decode(row[3], row[4]) if row[3] is not None else None
        if (row[2] == "completed") != (result is not None):
            raise ValueError("Inconsistent persisted call result")
        return {"request": request, "request_sha256": row[1], "phase": row[2],
                "result": result, "result_sha256": row[4], "error_type": row[5]}

    def reserve_call(self, request: LLMRequest) -> dict | None:
        self._write()
        session = ReviewInput.model_validate(self.require("session", request.review_id))
        enabled = session.assist_critique if request.role == "critique" else session.assist_revision
        if (not enabled or session.llm != request.settings or request.cycle > session.max_cycles
                or fingerprint(self.require("bundle", request.review_id)) != request.bundle_sha256):
            raise PermissionError("Call is outside the recorded review policy/evidence")
        digest = request.fingerprint()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT request_sha256 FROM calls WHERE call_id=?", (request.call_id,)).fetchone()
            if row:
                if row[0] != digest:
                    raise ValueError("Call identifier already reserved for another request")
            else:
                db.execute("INSERT INTO calls (call_id, review_id, request_json, request_sha256, phase) VALUES (?, ?, ?, ?, 'reserved')",
                           (request.call_id, request.review_id, canonical_json(request.model_dump(mode="json")), digest))
                return None
        previous = self.call(request.call_id)
        if previous["phase"] == "completed":
            return previous["result"]
        raise NeedsReconciliation("LLM call is reserved, failed or uncertain; no automatic retry")

    def finish_call(self, request: LLMRequest, result: dict) -> None:
        self._write()
        text, digest = canonical_json(result), fingerprint(result)
        if len(text.encode("utf-8")) > 262_144:
            raise ValueError("Validated LLM result exceeds 256 KiB")
        with self._connect() as db:
            cursor = db.execute("""UPDATE calls SET phase='completed', result_json=?, result_sha256=?
                WHERE call_id=? AND request_sha256=? AND phase='reserved'""",
                (text, digest, request.call_id, request.fingerprint()))
            if cursor.rowcount != 1:
                raise NeedsReconciliation("Call reservation is no longer available")

    def fail_call(self, request: LLMRequest, *, invalid_response: bool, error_type: str) -> None:
        self._write()
        phase = "invalid_response" if invalid_response else "needs_reconciliation"
        with self._connect() as db:
            db.execute("""UPDATE calls SET phase=?, error_type=?
                WHERE call_id=? AND request_sha256=? AND phase='reserved'""",
                (phase, error_type[:100], request.call_id, request.fingerprint()))

    def call_summaries(self, review_id: str) -> list[dict]:
        with self._connect() as db:
            ids = db.execute("SELECT call_id FROM calls WHERE review_id=? ORDER BY call_id", (review_id,)).fetchall()
        summaries = []
        for (call_id,) in ids:
            call = self.call(call_id)
            request = LLMRequest.model_validate(call["request"])
            summaries.append({"call_id": call_id, "request_sha256": call["request_sha256"],
                              "phase": call["phase"], "settings": request.settings.model_dump(mode="json"),
                              "role": request.role, "cycle": request.cycle,
                              "result_sha256": call["result_sha256"], "usage": "not_collected"})
        return summaries
