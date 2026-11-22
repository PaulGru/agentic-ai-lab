"""Small append-only checkpoint store for one local research workflow."""
from __future__ import annotations

from contextlib import closing
import json
import logging
from pathlib import Path
import sqlite3

from analysis.contracts import canonical_json, fingerprint


class WorkflowStore:
    def __init__(self, path: Path):
        self.path = path

    def _read(self, db):
        rows = db.execute("SELECT document, sha256 FROM checkpoints ORDER BY seq").fetchall()
        history = []
        previous = None
        for body, digest in rows:
            event = json.loads(body)
            if event["previous_sha256"] != previous or fingerprint(event) != digest:
                raise ValueError("Workflow checkpoint history checksum mismatch")
            history.append({**event, "sha256": digest})
            previous = digest
        return history

    def history(self):
        if not self.path.is_file():
            return []
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)) as db:
            return self._read(db)

    def current(self):
        history = self.history()
        if not history:
            raise FileNotFoundError("No workflow exists in this directory")
        return history[-1]

    def record_model_trace(self, reservation_sha256, trace):
        document = {"reservation_sha256": reservation_sha256, **trace}
        try:
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute("""CREATE TABLE IF NOT EXISTS model_trace (
                    seq INTEGER PRIMARY KEY, document TEXT NOT NULL, sha256 TEXT NOT NULL)""")
                db.execute("INSERT INTO model_trace(document, sha256) VALUES (?, ?)",
                           (canonical_json(document), fingerprint(document)))
        except sqlite3.Error:
            if trace["status"] == "started":
                raise
            logging.getLogger(__name__).warning("Model trace outcome could not be recorded")

    def model_traces(self):
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)) as db:
            exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_trace'").fetchone()
            if not exists:
                return []
            rows = db.execute("SELECT document, sha256 FROM model_trace ORDER BY seq").fetchall()
        documents = []
        for body, digest in rows:
            document = json.loads(body)
            if fingerprint(document) != digest:
                raise ValueError("Model trace checksum mismatch")
            documents.append(document)
        return documents

    def append(self, previous, state, *, action, detail=None):
        if previous is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        elif not self.path.is_file():
            raise FileNotFoundError("Workflow database is missing")
        body = {"previous_sha256": previous, "action": action, "detail": detail, "state": state}
        digest = fingerprint(body)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS checkpoints (seq INTEGER PRIMARY KEY, document TEXT NOT NULL, sha256 TEXT NOT NULL)")
            db.execute("BEGIN IMMEDIATE")
            history = self._read(db)
            latest = history[-1]["sha256"] if history else None
            if latest != previous:
                raise PermissionError("Workflow advanced concurrently or is already initialized")
            db.execute("INSERT INTO checkpoints(document, sha256) VALUES (?, ?)", (canonical_json(body), digest))
        return {**body, "sha256": digest}
