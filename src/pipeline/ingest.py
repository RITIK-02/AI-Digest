"""Stage 0: fetch all enabled sources, normalize into Item, write to db.

Raw payloads are stored alongside the normalized row so a parser bug never
costs a refetch. Every source module handles its own failure — a dead feed
logs a warning and returns an empty list; the pipeline must still produce a
digest from whatever sources are up.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import yaml

from src.models import Item, RawItem
from src.sources import arxiv
from src.sources.base import extract_identifiers

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "sources.yaml"

# New sources register here as they're added — see CLAUDE.md "Source notes".
SOURCE_MODULES = {
    "arxiv": arxiv,
}


def enabled_sources() -> list[str]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return [name for name, cfg in config.items() if cfg.get("enabled")]


def fetch_all() -> list[RawItem]:
    raw_items: list[RawItem] = []
    for name in enabled_sources():
        module = SOURCE_MODULES.get(name)
        if module is None:
            print(f"[ingest] source '{name}' enabled in config but has no module — skipping")
            continue
        try:
            fetched = module.fetch()
        except Exception as exc:  # noqa: BLE001 - one dead source must not kill the run
            print(f"[ingest] source '{name}' failed entirely: {exc}")
            continue
        raw_items.extend(fetched)
    return raw_items


def normalize(raw: RawItem) -> Item:
    """Identifier extraction runs here, centrally, over every text field a
    source gives us — not per-source — so a paper linked only in a post body
    is caught the same way as one linked directly."""
    ids = extract_identifiers(raw.url, raw.title, raw.abstract, json.dumps(raw.raw_payload))
    return Item(
        source=raw.source,
        raw_id=raw.raw_id,
        url=raw.url,
        title=raw.title or "(untitled)",
        abstract=raw.abstract or "",
        authors=raw.authors,
        published_at=raw.published_at,
        fetched_at=raw.fetched_at,
        arxiv_id=ids["arxiv_id"],
        doi=ids["doi"],
        openreview_id=ids["openreview_id"],
        github_repo=ids["github_repo"],
    )


def write_item(conn: sqlite3.Connection, raw: RawItem, item: Item) -> int:
    content_hash = item.content_hash()
    conn.execute(
        """
        INSERT INTO items (
            source, raw_id, url, title, abstract, authors_json, published_at,
            fetched_at, arxiv_id, doi, openreview_id, github_repo,
            content_hash, raw_payload_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(source, raw_id) DO UPDATE SET
            title=excluded.title,
            abstract=excluded.abstract,
            authors_json=excluded.authors_json,
            published_at=excluded.published_at,
            arxiv_id=excluded.arxiv_id,
            doi=excluded.doi,
            openreview_id=excluded.openreview_id,
            github_repo=excluded.github_repo,
            content_hash=excluded.content_hash,
            raw_payload_json=excluded.raw_payload_json
        """,
        (
            item.source,
            item.raw_id,
            item.url,
            item.title,
            item.abstract,
            json.dumps(item.authors),
            item.published_at.isoformat() if item.published_at else None,
            item.fetched_at.isoformat(),
            item.arxiv_id,
            item.doi,
            item.openreview_id,
            item.github_repo,
            content_hash,
            json.dumps(raw.raw_payload, default=str),
        ),
    )
    row = conn.execute(
        "SELECT id FROM items WHERE source = ? AND raw_id = ?", (item.source, item.raw_id)
    ).fetchone()
    return row["id"]


def run(conn: sqlite3.Connection) -> list[int]:
    raw_items = fetch_all()
    item_ids = []
    for raw in raw_items:
        item = normalize(raw)
        item_ids.append(write_item(conn, raw, item))
    conn.commit()
    print(f"[ingest] fetched {len(raw_items)} raw items, {len(set(item_ids))} unique in db")
    return item_ids
