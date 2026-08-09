"""arXiv source. The API, not scraping. Categories come from
config/sources.yaml. The rate limit is real: one request per three seconds,
batched by category — never parallelized.

Identifier extraction (arxiv_id, doi, github_repo, ...) happens centrally in
ingest.py during normalization, not here — every source's raw text goes
through the same extractor so a paper linked from an HN comment or a post
body is caught the same way as one linked directly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx
import yaml

from src.models import RawItem
from src.sources.base import RateLimiter

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "sources.yaml"

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["arxiv"]


def _parse_dt(text: str | None) -> datetime | None:
    if not text:
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def parse_atom_feed(xml_text: str, fetched_at: datetime | None = None) -> list[RawItem]:
    """Pure parser: raw Atom XML in, RawItems out. No network — this is what
    tests exercise against tests/fixtures/arxiv_sample_response.xml."""
    fetched_at = fetched_at or datetime.now(UTC)
    root = ET.fromstring(xml_text)
    items: list[RawItem] = []

    for entry in root.findall(f"{_ATOM_NS}entry"):
        entry_id = entry.findtext(f"{_ATOM_NS}id") or ""
        title = (entry.findtext(f"{_ATOM_NS}title") or "").strip().replace("\n", " ")
        summary = (entry.findtext(f"{_ATOM_NS}summary") or "").strip()
        published_at = _parse_dt(entry.findtext(f"{_ATOM_NS}published"))
        authors = [
            (a.findtext(f"{_ATOM_NS}name") or "").strip()
            for a in entry.findall(f"{_ATOM_NS}author")
        ]
        authors = [a for a in authors if a]

        abs_link = entry_id  # arXiv's <id> is the abs page URL, e.g. .../abs/2401.12345v2
        raw_id = abs_link.rsplit("/", 1)[-1]

        comment = entry.findtext(f"{_ARXIV_NS}comment") or ""
        doi = entry.findtext(f"{_ARXIV_NS}doi") or ""
        journal_ref = entry.findtext(f"{_ARXIV_NS}journal_ref") or ""
        links = [link.get("href", "") for link in entry.findall(f"{_ATOM_NS}link")]
        categories = [c.get("term", "") for c in entry.findall(f"{_ATOM_NS}category")]

        items.append(
            RawItem(
                source="arxiv",
                raw_id=raw_id,
                url=abs_link,
                fetched_at=fetched_at,
                title=title,
                abstract=summary,
                authors=authors,
                published_at=published_at,
                raw_payload={
                    "comment": comment,
                    "doi": doi,
                    "journal_ref": journal_ref,
                    "links": links,
                    "categories": categories,
                },
            )
        )
    return items


def fetch() -> list[RawItem]:
    """Network fetch: one request per configured category, rate-limited."""
    config = _load_config()
    if not config.get("enabled", False):
        return []

    limiter = RateLimiter(config["rate_limit_seconds"])
    all_items: list[RawItem] = []
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for category in config["categories"]:
            limiter.wait()
            try:
                resp = client.get(
                    config["base_url"],
                    params={
                        "search_query": f"cat:{category}",
                        "start": 0,
                        "max_results": config["max_results_per_category"],
                        "sortBy": "submittedDate",
                        "sortOrder": "descending",
                    },
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                # A dead/erroring source logs a warning and moves on — the
                # pipeline must produce a digest even if a source is down.
                print(f"[arxiv] category={category} fetch failed: {exc}")
                continue
            all_items.extend(parse_atom_feed(resp.text))
    return all_items
