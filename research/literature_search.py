"""V8.2 — Bounded, metadata-only scientific retrieval using OpenAlex."""

import json
import os
import re
from time import sleep
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import ValidationError

from research.literature import LiteratureCatalog, Paper
from research.mission import ResearchMission


OPENALEX_ENDPOINT = "https://api.openalex.org/works"
FIELDS = "id,display_name,publication_year,authorships,doi,abstract_inverted_index,type,primary_location,cited_by_count"
MAX_ABSTRACT_CHARACTERS = 2500


class InsufficientLiteratureError(ValueError):
    """The retrieved corpus cannot support a minimally grounded review."""


class OpenAlexRateLimitError(RuntimeError):
    """OpenAlex returned HTTP 429 and the request cannot be retried safely now."""



def reconstruct_abstract(index: Any) -> str | None:
    """OpenAlex provides some abstracts as token-position inverted indexes."""
    if not isinstance(index, dict) or not index or len(index) > 1500:
        return None
    positions: dict[int, str] = {}
    for word, offsets in index.items():
        if not isinstance(word, str) or len(word) > 80 or not isinstance(offsets, list):
            return None
        for offset in offsets:
            if type(offset) is not int or not 0 <= offset < 5000 or offset in positions:
                return None
            positions[offset] = word
        if len(positions) > 5000:
            return None
    if not positions or set(positions) != set(range(max(positions) + 1)):
        return None
    abstract = " ".join(positions[i] for i in range(len(positions)))
    return abstract[:MAX_ABSTRACT_CHARACTERS].strip() or None


def normalize_doi(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value.strip(), flags=re.I)
    if not re.fullmatch(r"10\.\d{4,9}/[^\s<>]+", text, flags=re.I):
        return None
    return text.lower()[:300]


def parse_paper(raw: Any, query: str) -> Paper | None:
    if not isinstance(raw, dict):
        return None
    identifier = raw.get("id")
    if not isinstance(identifier, str):
        return None
    match = re.fullmatch(r"https://openalex\.org/(W[0-9]+)", identifier)
    if not match:
        return None
    work_id = match.group(1)
    authors = []
    raw_authorships = raw.get("authorships")
    if not isinstance(raw_authorships, list):
        raw_authorships = []
    for item in raw_authorships[:10]:
        if not isinstance(item, dict):
            continue
        author = item.get("author")
        name = author.get("display_name") if isinstance(author, dict) else None
        if isinstance(name, str) and name.strip():
            authors.append(name.strip()[:160])
    location = raw.get("primary_location")
    source = location.get("source") if isinstance(location, dict) else None
    venue = source.get("display_name") if isinstance(source, dict) else None
    if not isinstance(venue, str):
        venue = None
    try:
        return Paper(
            work_id=work_id,
            title=str(raw.get("display_name") or "")[:400],
            authors=authors,
            year=raw.get("publication_year"),
            doi=normalize_doi(raw.get("doi")),
            venue=venue[:300] if venue else None,
            work_type=str(raw.get("type") or "unknown")[:100],
            abstract=reconstruct_abstract(raw.get("abstract_inverted_index")),
            openalex_url=f"https://openalex.org/{work_id}",
            cited_by_count=raw.get("cited_by_count") or 0,
            matched_queries=[query],
        )
    except (ValidationError, ValueError, TypeError):
        return None


class OpenAlexClient:
    def __init__(self, api_key: str | None = None, transport: Callable[[str, dict], Any] | None = None):
        self.api_key = api_key if api_key is not None else os.getenv("OPENALEX_API_KEY", "").strip()
        self.transport = transport or self._get_json

    def _get_json(self, url: str, headers: dict) -> Any:
        """GET with bounded retries only for short-lived HTTP 429 responses.

        Never retry a known exhausted daily quota or a Retry-After > 5 s.
        API keys stay in Authorization headers, never in diagnostics.
        """
        request = Request(url, headers=headers)
        for attempt in range(3):
            try:
                with urlopen(request, timeout=20) as response:
                    if response.status != 200:
                        raise RuntimeError(f"OpenAlex responded with HTTP {response.status}")
                    body = response.read(4_000_001)
                    if len(body) > 4_000_000:
                        raise RuntimeError("OpenAlex response exceeds size limit")
                    return json.loads(body)
            except HTTPError as exc:
                if exc.code != 429:
                    raise RuntimeError(f"OpenAlex HTTP error {exc.code}") from exc

                info = exc.headers or {}
                remaining = info.get("X-RateLimit-Remaining", "unknown")
                reset = info.get("X-RateLimit-Reset")
                retry_header = info.get("Retry-After")
                # The server may return a number of seconds. Invalid or HTTP-date
                # values are not guessed: use a short exponential fallback.
                retry_after = None
                if retry_header is not None:
                    try:
                        retry_after = float(retry_header)
                        if not 0 <= retry_after <= 86400:
                            retry_after = None
                    except (ValueError, TypeError):
                        pass

                details = f"OpenAlex HTTP 429. Remaining daily budget: {remaining}."
                if reset is not None:
                    details += f" Reset in {reset} second(s)."
                if retry_header is not None:
                    details += f" Retry-After: {retry_header}."

                # Remaining = 0 means retrying now will not help.
                if remaining == "0":
                    raise OpenAlexRateLimitError(
                        details + " Daily budget appears exhausted; retry after UTC reset "
                        "or configure a free OPENALEX_API_KEY."
                    ) from exc

                delay = max(1.0, float(2 ** attempt), retry_after or 0.0)
                if attempt == 2 or delay > 5:
                    raise OpenAlexRateLimitError(
                        details + " Retry later, or use a free OPENALEX_API_KEY. "
                        "Avoid repeatedly starting a new research thread."
                    ) from exc
                sleep(delay)
            except (URLError, TimeoutError) as exc:
                raise RuntimeError("OpenAlex network unavailable") from exc
        raise AssertionError("OpenAlex retry loop should always return or raise")

    def search(self, query: str, per_page: int) -> tuple[list[dict], int]:
        if not 1 <= per_page <= 10 or not 5 <= len(query) <= 500:
            raise ValueError("Unsafe OpenAlex search parameters")
        params = urlencode({
            "search.title_and_abstract": query,
            "per-page": per_page,
            "select": FIELDS,
        })
        headers = {"User-Agent": "AgenticAI-Lab/8.2 scientific-literature-demo", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = self.transport(f"{OPENALEX_ENDPOINT}?{params}", headers)
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError("Malformed OpenAlex response")
        count = payload.get("meta", {}).get("count", 0) if isinstance(payload.get("meta"), dict) else 0
        if not isinstance(count, int):
            count = 0
        return payload["results"][:per_page], max(0, count)


def retrieve_catalog(
    mission: ResearchMission,
    client: OpenAlexClient,
    *,
    max_queries: int = 4,
    per_query: int = 5,
    queries_override: list[str] | None = None,
) -> LiteratureCatalog:
    if not (1 <= max_queries <= 4 and 1 <= per_query <= 10):
        raise ValueError("Retrieval limits are bounded to 4 queries x 10 results")
    # A literature search can refine queries without changing the approved
    # research mission. Record the effective queries in the catalog.
    if queries_override is None:
        queries = list(dict.fromkeys(mission.literature_queries))[:max_queries]
    else:
        if (
            not isinstance(queries_override, list)
            or not 1 <= len(queries_override) <= max_queries
            or any(
                not isinstance(q, str)
                or not 5 <= len(q.strip()) <= 500
                or "\n" in q or "\r" in q
                for q in queries_override
            )
        ):
            raise ValueError("Provide 1-4 distinct, single-line search queries (5-500 chars)")
        queries = [q.strip() for q in queries_override]
        if len(set(queries)) != len(queries):
            raise ValueError("Duplicate search queries are not allowed")
    catalog: dict[str, Paper] = {}
    doi_to_id: dict[str, str] = {}
    query_log: list[dict[str, Any]] = []
    for query in queries:
        raw_works, count = client.search(query, per_query)
        accepted = 0
        rejected = 0
        duplicates = 0
        new_abstracts = 0
        for item in raw_works:
            paper = parse_paper(item, query)
            if paper is None:
                rejected += 1
                continue
            existing_id = paper.work_id
            if paper.doi and paper.doi in doi_to_id:
                existing_id = doi_to_id[paper.doi]
            if existing_id in catalog:
                old = catalog[existing_id]
                duplicates += 1
                if query not in old.matched_queries:
                    catalog[existing_id] = old.model_copy(update={"matched_queries": [*old.matched_queries, query]})
                continue
            catalog[paper.work_id] = paper
            if paper.abstract is not None:
                new_abstracts += 1
            if paper.doi:
                doi_to_id[paper.doi] = paper.work_id
            accepted += 1
        query_log.append({
            "query": query,
            "reported_matches": count,
            "returned": len(raw_works),
            "rejected_metadata": rejected,
            "duplicates": duplicates,
            "unique_new_papers": accepted,
            "new_abstracts": new_abstracts,
        })

    # Do not lower the catalog's evidence threshold to hide retrieval failures.
    abstract_count = sum(p.abstract is not None for p in catalog.values())
    if len(catalog) < 3 or abstract_count < 2:
        details = "; ".join(
            f"{row['query']!r}: matches={row['reported_matches']}, "
            f"returned={row['returned']}, accepted={row['unique_new_papers']}, "
            f"rejected={row['rejected_metadata']}, duplicates={row['duplicates']}"
            for row in query_log
        )
        raise InsufficientLiteratureError(
            f"Corpus insufficient: {len(catalog)} unique paper(s), "
            f"{abstract_count} usable abstract(s). "
            f"Minimum: 3 papers, 2 abstracts. Details: {details}. "
            "Try shorter, domain-relevant search queries with --query."
        )

    # If coverage is weak, explicitly stop instead of generating confident prose.
    return LiteratureCatalog(
        provider="openalex",
        retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        search_mode="title_and_abstract",
        requested_queries=queries,
        results_per_query=per_query,
        query_log=query_log,
        papers=list(catalog.values()),
    )
