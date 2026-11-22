"""Compile versioned ACL PDFs from verified manuscript exports."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import sleep
import unicodedata

from analysis.contracts import canonical_json
from research.critique.store import ReviewStore
from research.manuscript import SECTIONS, verify_manuscript_draft, verify_manuscript_export


_ESCAPES = str.maketrans({
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&",
    "#": r"\#", "%": r"\%", "_": r"\_", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}",
    "–": "-", "—": "-", "‑": "-", "−": "-", "\u00a0": " ", "\u202f": " ", "\u200b": "",
})

TITLES = {
    "9.0": "Pilote de relance sur catalogue",
    "9.0-ood-replay": "Répétition OOD hors ligne",
    "9.0-ood-gpu": "Calibration OOD sur GPU",
    "9.0-retry-gpu": "Effet d'une relance après une erreur d'outil : pilote sur GPU",
    "9.0-protocol-cpu": "Protocole expérimental : pilote numérique sur CPU",
    "9.0-protocol-gpu": "Protocole expérimental : pilote numérique sur GPU",
    "9.0-agentic-gpu": "Robustesse agentique : pilote comparatif sur GPU",
    "8.8": "Démonstration du processus de recherche",
}


def _text(value):
    return " ".join(unicodedata.normalize("NFC", str(value)).split()).translate(_ESCAPES)


def render_acl(manuscript, *, approved):
    title = TITLES[manuscript.schema_version]
    edition = "Version approuvée" if approved else "Brouillon de travail"
    lines = [r"\documentclass[11pt]{article}", r"\usepackage[preprint]{acl}", r"\usepackage{times}",
             r"\usepackage[T1]{fontenc}", r"\usepackage[utf8]{inputenc}", r"\usepackage{microtype}",
             r"\renewcommand{\abstractname}{Résumé}", rf"\title{{{title}}}",
             rf"\author{{Agentic AI Lab \\ {edition}}}", r"\begin{document}", r"\maketitle",
             r"\begin{abstract}", "Ce document rend compte d'un pilote pédagogique. Les observations portent sur le protocole et les tâches décrits ci-dessous.",
             r"\end{abstract}"]
    for section, heading in SECTIONS.items():
        statements = [statement for statement in manuscript.statements if statement.section == section]
        if not statements:
            continue
        lines.append(rf"\section{{{_text(heading)}}}")
        for statement in statements:
            lines.extend([f"{_text(statement.text)} [{_text(statement.statement_id)}]", ""])
        if section == "results" and manuscript.figure:
            lines.extend([r"\begin{table}[h]", r"\centering", r"\begin{tabular}{p{0.65\linewidth}r}",
                          r"\textbf{Condition} & \textbf{Valeur} \\ \hline"])
            percent = r"\,\%" if manuscript.schema_version.startswith("9.") else ""
            for row in manuscript.figure:
                lines.append(rf"{_text(row.group)} & {row.mean:g}{percent} \\")
            lines.extend([r"\end{tabular}", r"\caption{Résultats agrégés issus du rapport vérifié.}", r"\end{table}"])
    if manuscript.bibliography:
        lines.append(r"\section*{Références}")
        for item in manuscript.bibliography:
            citation = " · ".join(str(part) for part in (", ".join(item.authors), item.year, item.title, item.doi) if part)
            lines.extend([f"[{_text(item.work_id)}] {_text(citation)}", ""])
    lines.extend([r"\section*{Traçabilité}",
                  "Les identifiants entre crochets renvoient aux affirmations du manuscrit JSON et à leurs références dans le dossier de preuves.",
                  r"\end{document}", ""])
    return "\n".join(lines)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archived_pdf(destination, *, checkpoint, manuscript_sha256, approved):
    if not destination.is_dir():
        return None
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    expected = (checkpoint, manuscript_sha256, approved)
    if (manifest["checkpoint"], manifest["manuscript_sha256"], manifest["approved"]) != expected:
        raise ValueError("L'archive PDF ne correspond pas à ce jalon.")
    for name in ("manuscript.pdf", "manuscript.tex", "acl.sty"):
        path = destination / name
        if path.is_symlink() or _hash(path) != manifest["files"][name]:
            raise ValueError("L'intégrité d'une pièce de l'archive PDF a échoué.")
    return destination / "manuscript.pdf"


def archive_acl_pdf(source, destination, *, checkpoint, review_id, manuscript_sha256, approved, compiler, acl_style):
    identity = dict(checkpoint=checkpoint, manuscript_sha256=manuscript_sha256, approved=approved)
    existing = archived_pdf(destination, **identity)
    if existing:
        return existing
    store = ReviewStore(source.parent / "review.sqlite3", read_only=True)
    if approved:
        manuscript = verify_manuscript_export(store, review_id, manuscript_sha256, source)
    else:
        manuscript = verify_manuscript_draft(store, review_id, source)
    if manuscript.fingerprint() != manuscript_sha256:
        raise ValueError("L'empreinte du manuscrit ne correspond pas au jalon.")
    if not compiler or not acl_style or not acl_style.is_file():
        raise FileNotFoundError("Configurer pdflatex et le fichier officiel acl.sty pour compiler le PDF.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="acl-", dir=destination.parent, ignore_cleanup_errors=True) as directory:
        working = Path(directory)
        output = working / "export"
        output.mkdir()
        (working / "acl.sty").write_bytes(acl_style.read_bytes())
        (working / "manuscript.tex").write_text(render_acl(manuscript, approved=approved), encoding="utf-8", newline="\n")
        try:
            completed = subprocess.run([str(compiler), "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error",
                                        "-output-directory=export", "manuscript.tex"], cwd=working,
                                       capture_output=True, timeout=60, check=False)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("La compilation ACL a dépassé 60 secondes.") from error
        pdf = output / "manuscript.pdf"
        if completed.returncode or not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
            details = completed.stdout.decode("utf-8", errors="replace").splitlines()
            error = next((line for line in details if line.startswith("!")), "Vérifier les paquets TeX locaux.")
            raise RuntimeError(f"Compilation ACL en échec. {error}")

        archive = working / "archive"
        archive.mkdir()
        for path in (pdf, working / "manuscript.tex", working / "acl.sty"):
            (archive / path.name).write_bytes(path.read_bytes())
        manifest = {**identity, "files": {path.name: _hash(path) for path in archive.iterdir()}}
        (archive / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8", newline="\n")
        for attempt in range(5):
            try:
                os.rename(archive, destination)
                break
            except OSError as error:
                if destination.is_dir():
                    break
                if not isinstance(error, PermissionError) or attempt == 4:
                    raise
                sleep(0.2 * (attempt + 1))
    return archived_pdf(destination, **identity)
