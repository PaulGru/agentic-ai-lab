"""Two durable backends sharing explicit transactions and immutable JSON records.

PostgreSQL uses built-in full text search; SQLite FTS5 is the offline backend.
No vector service, embedding API, generated SQL or destructive cleanup is used.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import re
import sqlite3
import unicodedata

from analysis.contracts import canonical_json, fingerprint
from research.memory.contracts import MemoryContext, MemoryEvent, MemoryItem, SourceSnapshot


def terms(text: str) -> list[str]:
    plain = "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))
    stop_words = {"de", "des", "du", "la", "le", "les", "un", "une", "dans", "sur", "pour", "par", "avec",
                  "sans", "est", "sont", "au", "aux", "que", "qui", "ce", "ces", "ne", "pas", "and", "the",
                  "of", "to", "for", "with", "from", "this", "that", "or", "an", "in", "on", "as", "be", "is", "are"}
    return list(dict.fromkeys(word for word in re.findall(r"[^\W_]+", plain, flags=re.UNICODE)
                             if 2 <= len(word) <= 40 and word not in stop_words))


def search_text(item: MemoryItem) -> str:
    return " ".join(terms(" ".join([item.title, item.text, *item.tags])))


class SQLMemoryStore:
    postgres = False

    def table(self, name: str) -> str:
        if name not in {"sources", "memories", "events", "retrievals", "memory_fts"}:
            raise ValueError("Unknown research memory table")
        return f"{self.schema_name}.{name}" if self.postgres else name

    def execute(self, db, sql: str, parameters=()):
        return db.execute(sql.replace("?", "%s") if self.postgres else sql, parameters)

    def require_write(self):
        if self.read_only:
            raise PermissionError("Research memory store is read-only")

    def setup(self):
        self.require_write()
        with self.connect() as db:
            if self.postgres:
                self.execute(db, f"CREATE SCHEMA IF NOT EXISTS {self.schema_name}")
            self.execute(db, f"""CREATE TABLE IF NOT EXISTS {self.table('sources')} (
                source_id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
                document TEXT NOT NULL, sha256 TEXT NOT NULL)""")
            self.execute(db, f"""CREATE TABLE IF NOT EXISTS {self.table('memories')} (
                memory_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, source_id TEXT NOT NULL,
                logical_key TEXT NOT NULL, kind TEXT NOT NULL, scope TEXT NOT NULL, document TEXT NOT NULL,
                sha256 TEXT NOT NULL, search_text TEXT NOT NULL,
                FOREIGN KEY(source_id) REFERENCES {self.table('sources')}(source_id))""")
            sequence = "BIGSERIAL PRIMARY KEY" if self.postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"
            self.execute(db, f"""CREATE TABLE IF NOT EXISTS {self.table('events')} (
                sequence {sequence}, event_id TEXT UNIQUE NOT NULL, project_id TEXT NOT NULL,
                memory_id TEXT NOT NULL, document TEXT NOT NULL, sha256 TEXT NOT NULL,
                FOREIGN KEY(memory_id) REFERENCES {self.table('memories')}(memory_id))""")
            self.execute(db, f"""CREATE TABLE IF NOT EXISTS {self.table('retrievals')} (
                context_sha256 TEXT PRIMARY KEY, project_id TEXT NOT NULL,
                document TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
            self.execute(db, f"CREATE INDEX IF NOT EXISTS memories_project ON {self.table('memories')}(project_id, kind, scope)")
            if self.postgres:
                self.execute(db, f"CREATE INDEX IF NOT EXISTS memories_text ON {self.table('memories')} USING GIN (to_tsvector('simple', search_text))")
            else:
                self.execute(db, "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(memory_id UNINDEXED, search_text)")

    @staticmethod
    def decode(model, document, sha256):
        value = model.model_validate_json(document)
        if value.fingerprint() != sha256:
            raise ValueError("Persisted research memory checksum mismatch")
        return value

    def ingest(self, source: SourceSnapshot, items: list[MemoryItem]) -> None:
        self.require_write()
        # Revalidate instances: frozen models can still contain mutable nested lists/dicts.
        source = SourceSnapshot.model_validate(source.model_dump(mode="json"))
        items = [MemoryItem.model_validate(item.model_dump(mode="json")) for item in items]
        if any(item.project_id != source.project_id or item.source_id != source.source_id for item in items):
            raise ValueError("Memory item belongs to another project or source")
        with self.connect() as db:
            if not self.postgres:
                self.execute(db, "BEGIN IMMEDIATE")
            table = self.table("sources")
            self.execute(db, f"INSERT INTO {table} VALUES (?, ?, ?, ?) ON CONFLICT(source_id) DO NOTHING",
                         (source.source_id, source.project_id, canonical_json(source.model_dump(mode="json")), source.fingerprint()))
            saved = self.execute(db, f"SELECT document, sha256 FROM {table} WHERE source_id=?", (source.source_id,)).fetchone()
            if self.decode(SourceSnapshot, *saved) != source:
                raise ValueError("Existing memory source differs")
            table = self.table("memories")
            for item in items:
                values = (item.memory_id, item.project_id, item.source_id, item.logical_key, item.kind, item.scope,
                          canonical_json(item.model_dump(mode="json")), item.fingerprint(), search_text(item))
                inserted = self.execute(db, f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(memory_id) DO NOTHING",
                                        values).rowcount == 1
                saved = self.execute(db, f"SELECT document, sha256 FROM {table} WHERE memory_id=?", (item.memory_id,)).fetchone()
                if self.decode(MemoryItem, *saved) != item:
                    raise ValueError("Existing memory item differs")
                if inserted and not self.postgres:
                    self.execute(db, "INSERT INTO memory_fts (memory_id, search_text) VALUES (?, ?)", (item.memory_id, values[-1]))

    def source(self, project_id: str, source_id: str) -> SourceSnapshot | None:
        with self.connect() as db:
            row = self.execute(db, f"SELECT document, sha256 FROM {self.table('sources')} WHERE project_id=? AND source_id=?",
                               (project_id, source_id)).fetchone()
        result = self.decode(SourceSnapshot, *row) if row else None
        if result and (result.project_id != project_id or result.source_id != source_id):
            raise ValueError("Memory source project/identity mismatch")
        return result

    def item(self, project_id: str, memory_id: str) -> MemoryItem | None:
        with self.connect() as db:
            row = self.execute(db, f"SELECT document, sha256 FROM {self.table('memories')} WHERE project_id=? AND memory_id=?",
                               (project_id, memory_id)).fetchone()
        result = self.decode(MemoryItem, *row) if row else None
        if result and (result.project_id != project_id or result.memory_id != memory_id):
            raise ValueError("Memory item project/identity mismatch")
        return result

    def candidates(self, project_id: str, query_terms: list[str], *, kinds=(), scopes=()) -> list[MemoryItem]:
        if not query_terms:
            return []
        with self.connect() as db:
            table = self.table("memories")
            filters, parameters = "", []
            for column, values in (("kind", kinds), ("scope", scopes)):
                if values:
                    filters += f" AND m.{column} IN ({','.join('?' for _ in values)})"
                    parameters.extend(values)
            if self.postgres:
                query = " | ".join(query_terms)
                rows = self.execute(db, f"""SELECT m.document, m.sha256, m.search_text FROM {table} m
                    WHERE m.project_id=? AND to_tsvector('simple', m.search_text) @@ to_tsquery('simple', ?){filters}
                    ORDER BY ts_rank(to_tsvector('simple', m.search_text), to_tsquery('simple', ?)) DESC, m.memory_id LIMIT 50""",
                                    (project_id, query, *parameters, query)).fetchall()
            else:
                query = " OR ".join('"' + word + '"' for word in query_terms)
                rows = self.execute(db, f"""SELECT m.document, m.sha256, m.search_text FROM {table} m
                    JOIN memory_fts f ON f.memory_id=m.memory_id WHERE m.project_id=? AND memory_fts MATCH ?{filters}
                    ORDER BY bm25(memory_fts), m.memory_id LIMIT 50""", (project_id, query, *parameters)).fetchall()
        results = []
        for text, digest, indexed_text in rows:
            item = self.decode(MemoryItem, text, digest)
            if (item.project_id != project_id or indexed_text != search_text(item)
                    or (kinds and item.kind not in kinds) or (scopes and item.scope not in scopes)):
                raise ValueError("Memory search index/source mismatch")
            results.append(item)
        return results

    def events(self, project_id: str, memory_id: str) -> list[MemoryEvent]:
        with self.connect() as db:
            rows = self.execute(db, f"SELECT document, sha256 FROM {self.table('events')} WHERE project_id=? AND memory_id=? ORDER BY sequence",
                                (project_id, memory_id)).fetchall()
        events = [self.decode(MemoryEvent, *row) for row in rows]
        if any(e.project_id != project_id or e.memory_id != memory_id for e in events):
            raise ValueError("Memory event project/target mismatch")
        return events

    def append_event(self, event: MemoryEvent) -> None:
        self.require_write()
        event = MemoryEvent.model_validate(event.model_dump(mode="json"))
        with self.connect() as db:
            if not self.postgres:
                self.execute(db, "BEGIN IMMEDIATE")
            else:
                # Serialize lifecycle decisions per project, including opposite
                # A->B / B->A supersessions, without locking another project's data.
                self.execute(db, "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))", (event.project_id,))
            suffix = " FOR UPDATE" if self.postgres else ""
            row = self.execute(db, f"SELECT document, sha256 FROM {self.table('memories')} WHERE project_id=? AND memory_id=?{suffix}",
                               (event.project_id, event.memory_id)).fetchone()
            if row is None:
                raise ValueError("Memory target not found in this project")
            item = self.decode(MemoryItem, *row)
            if item.fingerprint() != event.target_sha256:
                raise PermissionError("Lifecycle decision targets another memory version")
            previous = self.execute(db, f"SELECT document, sha256 FROM {self.table('events')} WHERE event_id=?", (event.event_id,)).fetchone()
            if previous:
                if self.decode(MemoryEvent, *previous) != event:
                    raise ValueError("Conflicting lifecycle event")
                return
            latest = self.execute(db, f"SELECT document, sha256 FROM {self.table('events')} WHERE project_id=? AND memory_id=? ORDER BY sequence DESC LIMIT 1",
                                  (event.project_id, event.memory_id)).fetchone()
            prior = self.decode(MemoryEvent, *latest) if latest else None
            if prior and prior.action in {"invalidate", "supersede"}:
                raise ValueError("Invalidated or superseded memories cannot be reactivated")
            if event.action == "confirm" and (item.initial_status != "tentative" or prior):
                raise ValueError("Only a tentative memory may be confirmed")
            if event.replacement_id:
                replacement = self.execute(db, f"SELECT document, sha256 FROM {self.table('memories')} WHERE project_id=? AND memory_id=?",
                                           (event.project_id, event.replacement_id)).fetchone()
                if replacement is None:
                    raise ValueError("Replacement must exist in the same project")
                other = self.decode(MemoryItem, *replacement)
                if (other.kind, other.logical_key) != (item.kind, item.logical_key):
                    raise ValueError("Replacement must describe the same logical knowledge")
                history = self.execute(db, f"SELECT document, sha256 FROM {self.table('events')} WHERE project_id=? AND memory_id=? ORDER BY sequence DESC LIMIT 1",
                                       (event.project_id, other.memory_id)).fetchone()
                if history and self.decode(MemoryEvent, *history).action in {"invalidate", "supersede"}:
                    raise ValueError("Replacement is inactive")
            self.execute(db, f"INSERT INTO {self.table('events')} (event_id, project_id, memory_id, document, sha256) VALUES (?, ?, ?, ?, ?)",
                         (event.event_id, event.project_id, event.memory_id,
                          canonical_json(event.model_dump(mode="json")), event.fingerprint()))

    def audit(self, context: MemoryContext) -> None:
        self.require_write()
        with self.connect() as db:
            self.execute(db, f"INSERT INTO {self.table('retrievals')} (context_sha256, project_id, document) VALUES (?, ?, ?) ON CONFLICT(context_sha256) DO NOTHING",
                         (context.fingerprint(), context.project_id, canonical_json(context.model_dump(mode="json"))))

    def retrieval(self, project_id: str, context_sha256: str) -> MemoryContext | None:
        with self.connect() as db:
            row = self.execute(db, f"SELECT document FROM {self.table('retrievals')} WHERE project_id=? AND context_sha256=?",
                               (project_id, context_sha256)).fetchone()
        return self.decode(MemoryContext, row[0], context_sha256) if row else None


class SQLiteMemoryStore(SQLMemoryStore):
    def __init__(self, path: str | Path, *, read_only: bool = False):
        self.path = Path(path)
        self.read_only = read_only
        if read_only and not self.path.is_file():
            raise FileNotFoundError("Existing research memory database required")

    def setup(self):
        self.require_write()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        super().setup()

    @contextmanager
    def connect(self):
        if self.read_only:
            db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
            db.execute("PRAGMA query_only=ON")
        else:
            db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()


class PostgresMemoryStore(SQLMemoryStore):
    postgres = True

    def __init__(self, dsn: str, *, schema_name: str = "research_memory_v87", read_only: bool = False):
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,60}", schema_name):
            raise ValueError("Invalid PostgreSQL memory schema")
        self.dsn = dsn  # never serialize credentials into bindings, prompts or records
        self.schema_name = schema_name
        self.read_only = read_only

    @contextmanager
    def connect(self):
        import psycopg
        with psycopg.connect(self.dsn, connect_timeout=5) as db:
            db.read_only = self.read_only
            db.execute("SET LOCAL statement_timeout = '15000ms'")
            yield db
