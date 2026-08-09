"""Dedupe logic against an in-memory db and fixture Items — no network, and
the embedding model is monkeypatched out so this stays a fast unit test
rather than something that downloads sentence-transformers weights."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import numpy as np
import pytest

from src.models import Item, RawItem
from src.pipeline import db, dedupe, ingest


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    db.init_schema(connection)
    yield connection
    connection.close()


@pytest.fixture(autouse=True)
def _no_real_embeddings(monkeypatch):
    # Zero vectors -> cosine similarity is always 0, well under the 0.78
    # adjudication floor, so these tests only exercise the strong-key and
    # title-normalization merge paths, not the embedding path.
    monkeypatch.setattr(
        dedupe, "embed_texts", lambda texts: np.zeros((len(texts), 8), dtype="float32")
    )


def _make_item(raw_id: str, title: str, arxiv_id: str | None = None) -> Item:
    return Item(
        source="arxiv",
        raw_id=raw_id,
        url=f"http://arxiv.org/abs/{raw_id}",
        title=title,
        abstract="",
        authors=[],
        published_at=None,
        fetched_at=datetime.now(UTC),
        arxiv_id=arxiv_id,
    )


def _insert(conn: sqlite3.Connection, item: Item) -> int:
    raw = RawItem(source=item.source, raw_id=item.raw_id, url=item.url, fetched_at=item.fetched_at)
    return ingest.write_item(conn, raw, item)


def _story_membership(conn: sqlite3.Connection) -> dict[int, set[str]]:
    rows = conn.execute(
        "SELECT si.story_id, i.raw_id FROM story_items si JOIN items i ON i.id = si.item_id"
    ).fetchall()
    out: dict[int, set[str]] = {}
    for row in rows:
        out.setdefault(row["story_id"], set()).add(row["raw_id"])
    return out


def test_normalize_title_strips_punctuation_and_case():
    assert dedupe.normalize_title("Jailbreaking LLMs: A Survey!") == "jailbreaking llms a survey"
    assert dedupe.normalize_title("  Extra   Spaces  ") == "extra spaces"


def test_union_find_merges_transitively():
    uf = dedupe.UnionFind([1, 2, 3, 4])
    uf.union(1, 2)
    uf.union(3, 4)
    uf.union(2, 3)
    groups = uf.groups()
    assert len(groups) == 1
    assert set(next(iter(groups.values()))) == {1, 2, 3, 4}


def test_run_merges_items_sharing_arxiv_id(conn):
    a = _make_item("2401.00001v1", "Jailbreak Attacks on Aligned LLMs", arxiv_id="2401.00001")
    b = _make_item(
        "2401.00001v2", "Jailbreak Attacks on Aligned LLMs (Revised)", arxiv_id="2401.00001"
    )
    c = _make_item("2402.00002v1", "Unrelated Time Series Paper", arxiv_id="2402.00002")
    for item in (a, b, c):
        _insert(conn, item)

    dedupe.run(conn)

    story_count = conn.execute("SELECT COUNT(*) FROM stories").fetchone()[0]
    assert story_count == 2  # a+b merged by shared arxiv_id, c stands alone

    membership = _story_membership(conn)
    assert {"2401.00001v1", "2401.00001v2"} in membership.values()


def test_run_merges_items_sharing_normalized_title(conn):
    a = _make_item("3001.00001v1", "Same Title Here!")
    b = _make_item("3001.00002v1", "same title here")
    for item in (a, b):
        _insert(conn, item)

    dedupe.run(conn)

    assert conn.execute("SELECT COUNT(*) FROM stories").fetchone()[0] == 1


def test_run_leaves_unrelated_items_in_separate_stories(conn):
    a = _make_item("4001.00001v1", "First Distinct Paper")
    b = _make_item("4001.00002v1", "Second Distinct Paper")
    for item in (a, b):
        _insert(conn, item)

    dedupe.run(conn)

    assert conn.execute("SELECT COUNT(*) FROM stories").fetchone()[0] == 2


def test_run_attaches_new_version_to_existing_story(conn):
    a = _make_item("5001.00001v1", "Original Version", arxiv_id="5001.00001")
    _insert(conn, a)
    dedupe.run(conn)
    assert conn.execute("SELECT COUNT(*) FROM stories").fetchone()[0] == 1

    b = _make_item("5001.00001v2", "Original Version (v2 title tweak)", arxiv_id="5001.00001")
    _insert(conn, b)
    dedupe.run(conn)

    # Still one story — v2 attached to the existing one rather than starting a new one.
    assert conn.execute("SELECT COUNT(*) FROM stories").fetchone()[0] == 1
    membership = _story_membership(conn)
    assert set(next(iter(membership.values()))) == {"5001.00001v1", "5001.00001v2"}
