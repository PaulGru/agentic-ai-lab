
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlsplit,
    urlunsplit,
)


def inspect_url(url: str):
    """Validate URL syntax and return a deduplication key."""

    if (
        not isinstance(url, str)
        or not url
        or len(url) > 2048
        or url != url.strip()
        or any(c.isspace() or ord(c) < 32 for c in url)
    ):
        return None

    try:
        parts = urlsplit(url)
        host = parts.hostname
        port = parts.port
    except ValueError:
        return None

    scheme = parts.scheme.lower()

    if (
        scheme not in ("http", "https")
        or not host
        or parts.username is not None
        or parts.password is not None
    ):
        return None

    host = host.lower().rstrip(".")

    netloc = f"[{host}]" if ":" in host else host
    default_port = 443 if scheme == "https" else 80

    if port is not None and port != default_port:
        netloc += f":{port}"

    # Ignore tracking parameters when detecting duplicates.
    query = [
        (key, value)
        for key, value in parse_qsl(
            parts.query,
            keep_blank_values=True,
        )
        if not key.lower().startswith("utm_")
        and key.lower() not in ("fbclid", "gclid")
    ]

    path = parts.path.rstrip("/") or "/"

    normalized = urlunsplit((
        scheme,
        netloc,
        path,
        urlencode(query),
        "",
    ))

    return normalized, host


def audit_sources(
    sources: list[dict[str, str]],
    web_search_used: bool,
) -> tuple[list[dict[str, str]], dict]:
    """Audit citation metadata, without fetching web pages."""

    cleaned_sources = []
    seen_urls = set()
    domains = set()

    invalid_urls = 0
    duplicate_urls = 0
    non_https_urls = 0

    for source in sources:
        if not isinstance(source, dict):
            invalid_urls += 1
            continue

        url = source.get("url")
        inspected = inspect_url(url)

        if inspected is None:
            invalid_urls += 1
            continue

        normalized_url, domain = inspected

        if normalized_url in seen_urls:
            duplicate_urls += 1
            continue

        seen_urls.add(normalized_url)
        domains.add(domain)

        if urlsplit(url).scheme.lower() != "https":
            non_https_urls += 1

        title = source.get("title")

        if not isinstance(title, str) or not title.strip():
            title = url

        cleaned_sources.append({
            "title": title[:300],
            "url": url,
            "domain": domain,
        })

    warnings = []

    if web_search_used:
        status = (
            "citations_available"
            if cleaned_sources
            else "search_without_citations"
        )

        if not cleaned_sources:
            warnings.append("web_search_without_citations")
    else:
        status = (
            "inconsistent_metadata"
            if cleaned_sources
            else "not_searched"
        )

        if cleaned_sources:
            warnings.append("citations_without_web_search")

    if invalid_urls:
        warnings.append("invalid_source_urls")

    if duplicate_urls:
        warnings.append("duplicate_source_urls")

    if non_https_urls:
        warnings.append("non_https_source_urls")

    report = {
        "status": status,
        "source_count": len(cleaned_sources),
        "distinct_domains": len(domains),
        "domains": sorted(domains),
        "invalid_urls": invalid_urls,
        "duplicate_urls": duplicate_urls,
        "content_verified": False,
        "warnings": warnings,
    }

    return cleaned_sources, report
