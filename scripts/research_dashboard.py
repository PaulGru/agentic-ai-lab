"""Serve a local workflow timeline and compile immutable manuscript editions."""
from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from shutil import which
from threading import Event, Thread

from research.acl_pdf import archive_acl_pdf, archived_pdf
from research.brief_workflow import ROOT
from scripts.research_progress import STAGE_NAMES, _message


ASSETS = Path(__file__).with_name("dashboard_assets")
LOCAL_TOOLS = ROOT / "runs_control" / "acl-tools"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/dashboard.css": ("dashboard.css", "text/css; charset=utf-8"),
    "/dashboard.js": ("dashboard.js", "text/javascript; charset=utf-8"),
}


@dataclass(frozen=True)
class Edition:
    step: int
    version: int
    checkpoint: str
    review_id: str
    manuscript_sha256: str
    source_file: str
    approved: bool

    @property
    def identity(self):
        return dict(checkpoint=self.checkpoint, manuscript_sha256=self.manuscript_sha256, approved=self.approved)


def timeline(history):
    steps, editions, seen = [], {}, {}
    for step, event in enumerate(history, start=1):
        state = event["state"]
        artifacts = state.get("artifacts", {})
        draft = artifacts.get("draft")
        archive = None
        if draft:
            approved = bool(state.get("export"))
            source = state.get("export") if approved else draft["draft"]
            key = (source, draft["manuscript_sha256"], approved)
            if key not in seen:
                seen[key] = step
                editions[step] = Edition(step, state["version"], event["sha256"], draft["review_id"],
                                         draft["manuscript_sha256"], source, approved)
            archive = seen[key]
        mission = artifacts.get("mission", {}).get("mission", {})
        literature = artifacts.get("literature", {}).get("review", {})
        protocol = artifacts.get("protocol", {}).get("document", {})
        highlights = [("Question de recherche", mission.get("research_question")),
                      ("Synthèse bibliographique", literature.get("overview")),
                      ("Périmètre du protocole", protocol.get("scope"))]
        message = "Manuscrit approuvé et exporté." if state.get("export") else _message(event)
        steps.append({"step": step, "version": state["version"], "stage": STAGE_NAMES.get(state["stage"], state["stage"]),
                      "message": message, "status": state["status"], "action": event["action"],
                      "highlights": [{"label": label, "text": value} for label, value in highlights if isinstance(value, str)],
                      "edition": archive})
    return steps, editions


class Dashboard:
    def __init__(self, workflow, *, compiler=None, acl_style=None):
        self.root = workflow.root
        self.store = workflow.store
        self.archive_root = self.root / "dashboard" / "pdfs"
        portable = next((LOCAL_TOOLS / "tinytex").glob("*/bin/windows/pdflatex.exe"), None)
        self.compiler = str(Path(compiler).resolve()) if compiler else which("pdflatex") or portable
        self.acl_style = Path(acl_style).resolve() if acl_style else LOCAL_TOOLS / "acl.sty"
        self.pdf_states = {}

    def _destination(self, edition):
        path = self.archive_root / f"{edition.step:04d}-{edition.checkpoint[:12]}"
        if not path.resolve().is_relative_to(self.root) or path.is_symlink():
            raise ValueError("Le dossier d'archive doit rester dans la mission locale.")
        return path

    def archive_pending(self):
        _, editions = timeline(self.store.history())
        for step, edition in editions.items():
            if edition.checkpoint in self.pdf_states:
                continue
            try:
                source = Path(edition.source_file).resolve()
                if not source.is_relative_to(self.root):
                    raise ValueError("Le manuscrit doit rester dans la mission locale.")
                archive_acl_pdf(source.parent, self._destination(edition), review_id=edition.review_id,
                                compiler=self.compiler, acl_style=self.acl_style, **edition.identity)
                result = {"status": "ready", "url": f"/pdf/{step}"}
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                result = {"status": "error", "message": str(error)[:400]}
            self.pdf_states[edition.checkpoint] = result

    def snapshot(self):
        history = self.store.history()
        steps, editions = timeline(history)
        return {
            "run": history[-1]["state"]["id"] if history else self.root.name,
            "steps": steps,
            "editions": [{"step": edition.step, "version": edition.version, "approved": edition.approved,
                          **self.pdf_states.get(edition.checkpoint, {"status": "pending"})}
                         for edition in editions.values()],
        }

    def pdf(self, step):
        _, editions = timeline(self.store.history())
        edition = editions.get(step)
        if edition:
            return archived_pdf(self._destination(edition), **edition.identity)
        return None


def serve_dashboard(dashboard, port):
    stopped = Event()

    def archive_loop():
        while not stopped.is_set():
            try:
                dashboard.archive_pending()
            except (OSError, ValueError):
                pass  # Each read of the page reports a journal error until it can be read again.
            stopped.wait(1)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.headers.get("Host") not in {f"127.0.0.1:{server.server_port}", f"localhost:{server.server_port}"}:
                self.reply(403, b"Forbidden", "text/plain")
                return
            try:
                if self.path in STATIC_FILES:
                    filename, content_type = STATIC_FILES[self.path]
                    self.reply(200, (ASSETS / filename).read_bytes(), content_type)
                    return
                if self.path == "/api/progress":
                    payload = json.dumps(dashboard.snapshot(), ensure_ascii=False).encode("utf-8")
                    self.reply(200, payload, "application/json; charset=utf-8")
                    return
                match = re.fullmatch(r"/pdf/([1-9][0-9]{0,8})", self.path)
                if match and (pdf := dashboard.pdf(int(match[1]))):
                    self.reply(200, pdf.read_bytes(), "application/pdf")
                    return
            except (OSError, ValueError, KeyError):
                self.reply(503, b"Journal ou archive indisponible", "text/plain")
                return
            self.reply(404, b"Introuvable", "text/plain")

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        worker = Thread(target=archive_loop, daemon=True)
        worker.start()
        print(f"Suivi local disponible sur http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            stopped.set()
            worker.join()
