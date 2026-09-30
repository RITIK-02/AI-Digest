"""HN source. Algolia's HN Search API, filtered by a points threshold — the
API, not scraping. No rate limit like arXiv's; Algolia's public search API
has generous limits at this volume.

When an HN story links directly to an arXiv paper, its arxiv_id gets
extracted during ingest.normalize() the same as any other item, so dedupe's
strong-key clustering attaches it to the paper's story as a discussion item
automatically — no HN-specific logic needed there. Points/comment counts are
kept in raw_payload for the `rising` section (never invented — always the
real Algolia number, or the section stays empty for that story).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

from src.models import RawItem

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "sources.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["hn"]


def _hn_discussion_url(object_id: str) -> str:
    return f"https://news.ycombinator.com/item?id={object_id}"


def _parse_dt(text: str | None) -> datetime | None:
    if not text:
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def parse_hits(hits: list[dict], fetched_at: datetime | None = None) -> list[RawItem]:
    """Pure parser: Algolia hit dicts in, RawItems out. No network — this is
    what tests exercise against tests/fixtures/hn_sample_response.json."""
    fetched_at = fetched_at or datetime.now(UTC)
    items: list[RawItem] = []

    for hit in hits:
        object_id = str(hit.get("objectID") or "")
        if not object_id:
            continue
        discussion_url = _hn_discussion_url(object_id)
        # External link if there is one (this is what lets an arXiv-linking
        # HN story get picked up as a discussion item for that paper);
        # otherwise it's a self-post and the discussion IS the item.
        url = hit.get("url") or discussion_url
        title = hit.get("title") or hit.get("story_title") or ""
        abstract = hit.get("story_text") or hit.get("text") or ""
        author = hit.get("author")

        items.append(
            RawItem(
                source="hn",
                raw_id=object_id,
                url=url,
                fetched_at=fetched_at,
                title=title,
                abstract=abstract,
                authors=[author] if author else [],
                published_at=_parse_dt(hit.get("created_at")),
                raw_payload={
                    "points": hit.get("points"),
                    "num_comments": hit.get("num_comments"),
                    "hn_url": discussion_url,
                },
            )
        )
    return items


def fetch() -> list[RawItem]:
    config = _load_config()
    if not config.get("enabled", False):
        return []

    cutoff = int(datetime.now(UTC).timestamp()) - config["lookback_hours"] * 3600
    params = {
        "tags": "story",
        "numericFilters": f"points>={config['points_threshold']},created_at_i>={cutoff}",
        "hitsPerPage": config["max_results"],
    }
    try:
        resp = httpx.get(config["base_url"], params=params, timeout=30.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        # A dead/erroring source logs a warning and moves on — the pipeline
        # must produce a digest even if a source is down.
        print(f"[hn] fetch failed: {exc}")
        return []

    return parse_hits(resp.json().get("hits", []))
