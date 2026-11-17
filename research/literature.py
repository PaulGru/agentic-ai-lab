"""V8.2 — Typed bibliographic artifacts and conservative evidence checks."""

import hashlib
import json
import re
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from research.mission import ResearchMission, mission_digest


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=10, max_length=1200)]
Short = Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=400)]
HEX256 = re.compile(r"[a-f0-9]{64}\Z")


def digest_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_approved_mission(document: dict[str, Any]) -> ResearchMission:
    """A content digest detects modifications; it is not a digital signature."""
    if not isinstance(document, dict) or document.get("schema_version") != "8.1":
        raise ValueError("Expected a V8.1 mission.json export")
    if not isinstance(document.get("brief_sha256"), str) or not HEX256.fullmatch(document["brief_sha256"]):
        raise ValueError("Missing V8.1 brief SHA-256")
    raw = document.get("mission")
    mission = ResearchMission.model_validate(raw)
    if mission.model_dump() != raw:
        raise ValueError("Mission content is not canonical")
    expected = mission_digest(raw)
    if document.get("mission_sha256") != expected:
        raise ValueError("Mission fingerprint mismatch; do not start review")
    return mission


class Paper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_id: str = Field(pattern=r"^W[0-9]+$")
    title: Short
    authors: list[str] = Field(max_length=10)
    year: int = Field(ge=1800, le=2100)
    doi: str | None = Field(default=None, max_length=300)
    venue: str | None = Field(default=None, max_length=300)
    work_type: str = Field(max_length=100)
    abstract: str | None = Field(default=None, max_length=2500)
    openalex_url: str
    cited_by_count: int = Field(ge=0)
    matched_queries: list[str] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def check_identifiers(self):
        if self.openalex_url != f"https://openalex.org/{self.work_id}":
            raise ValueError("Unexpected OpenAlex URL")
        if self.doi is not None and not re.fullmatch(r"10\.\d{4,9}/[^\s<>]+", self.doi):
            raise ValueError("Invalid DOI syntax")
        return self


class LiteratureCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(pattern="^openalex$")
    retrieved_at: str
    search_mode: str = Field(pattern=r"^title_and_abstract$")
    requested_queries: list[str] = Field(min_length=1, max_length=4)
    results_per_query: int = Field(ge=1, le=10)
    query_log: list[dict[str, Any]] = Field(min_length=1, max_length=4)
    papers: list[Paper] = Field(min_length=3, max_length=40)

    @model_validator(mode="after")
    def check_catalog(self):
        ids = [p.work_id for p in self.papers]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate OpenAlex work IDs")
        dois = [p.doi.lower() for p in self.papers if p.doi]
        if len(dois) != len(set(dois)):
            raise ValueError("Duplicate DOI")
        if sum(p.abstract is not None for p in self.papers) < 2:
            raise ValueError("Insufficient abstracts for evidence-based review")
        return self


class EvidenceFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    work_id: str = Field(pattern=r"^W[0-9]+$")
    claim: Text
    abstract_excerpt: Short = Field(max_length=260)

    @model_validator(mode="after")
    def short_quote(self):
        if len(self.abstract_excerpt.split()) > 25:
            raise ValueError("Use at most 25 words from an abstract")
        return self


class LiteratureReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overview: Text
    findings: list[EvidenceFinding] = Field(min_length=1, max_length=8)
    possible_gaps: list[Text] = Field(min_length=1, max_length=6)
    methodological_risks: list[Text] = Field(min_length=1, max_length=6)
    next_queries: list[Short] = Field(min_length=1, max_length=6)
    questions_for_v83: list[Text] = Field(min_length=1, max_length=6)
    coverage_limits: list[Text] = Field(min_length=1, max_length=6)


def validate_evidence(review: LiteratureReview, catalog: LiteratureCatalog) -> None:
    """Checks quote presence, not semantic correctness of the paraphrase."""
    known = {p.work_id: p for p in catalog.papers}
    seen: set[str] = set()
    for item in review.findings:
        if item.work_id in seen:
            raise ValueError("Only one quoted finding per paper")
        seen.add(item.work_id)
        paper = known.get(item.work_id)
        if paper is None:
            raise ValueError(f"Unretrieved citation: {item.work_id}")
        if not paper.abstract:
            raise ValueError(f"No abstract available for {item.work_id}")
        normalized_abstract = " ".join(paper.abstract.casefold().split())
        normalized_excerpt = " ".join(item.abstract_excerpt.casefold().split())
        if normalized_excerpt not in normalized_abstract:
            raise ValueError(f"Quote not found in retrieved abstract: {item.work_id}")


def _markdown_text(value: str) -> str:
    """Remove raw HTML and table metacharacters from untrusted metadata."""
    return value.replace("<", "&lt;").replace(">", "&gt;").replace("|", "\\|").replace("\n", " ")


def render_markdown(catalog: LiteratureCatalog, review: LiteratureReview, mission: ResearchMission) -> str:
    references = {paper.work_id: paper for paper in catalog.papers}
    lines = [
        "# V8.2 — Revue bibliographique préliminaire",
        "",
        f"**Mission :** {_markdown_text(mission.title)}",
        f"**Recherche :** OpenAlex (titres et résumés), {catalog.retrieved_at}",
        f"**Périmètre :** {len(catalog.papers)} notices retrouvées, {len(review.findings)} extraits cités.",
        "",
        "> ATTENTION — Synthèse exploratoire de métadonnées et résumés : pas de vérification des articles intégraux, pas de preuve de nouveauté.",
        "",
        "## Vue d'ensemble (proposition du LLM, à vérifier)",
        "",
        _markdown_text(review.overview),
        "",
        "## Éléments rattachés à des résumés",
        "",
    ]
    for item in review.findings:
        paper = references[item.work_id]
        lines.extend([
            f"### {item.work_id} — {_markdown_text(paper.title)} ({paper.year})",
            "",
            f"**Interprétation proposée :** {_markdown_text(item.claim)}",
            "",
            f"**Extrait exact du résumé :** « {_markdown_text(item.abstract_excerpt)} »",
            "",
            f"**Notice :** {paper.openalex_url}",
            "",
        ])
    for heading, values in [
        ("Lacunes possibles (hypothèses, non démontrées)", review.possible_gaps),
        ("Risques méthodologiques", review.methodological_risks),
        ("Recherches complémentaires", review.next_queries),
        ("Questions pour V8.3", review.questions_for_v83),
        ("Limites de couverture", review.coverage_limits),
    ]:
        lines.extend([f"## {heading}", ""])
        lines.extend(f"- {_markdown_text(x)}" for x in values)
        lines.append("")
    lines.extend(["## Catalogue des références retrouvées", "", "| ID | Année | Titre | DOI |", "|---|---:|---|---|"])
    for paper in catalog.papers:
        doi = f"https://doi.org/{paper.doi}" if paper.doi else "Non fourni"
        lines.append(f"| {paper.work_id} | {paper.year} | {_markdown_text(paper.title)} | {doi} |")
    lines.extend(["", "## Requêtes utilisées", ""])
    lines.extend(f"- `{_markdown_text(q)}`" for q in catalog.requested_queries)
    lines.extend(["", "**À faire avant toute affirmation scientifique forte :** lire les textes intégraux, compléter les requêtes, vérifier méthodes et résultats, documenter les exclusions.", ""])
    return "\n".join(lines)
