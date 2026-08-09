"""Stage 2: embed title + abstract locally, score against the interest
vector, keep the top ~80 candidates.

This stage is what keeps the bill under $5/month. Do not remove it, and do
not "temporarily" bypass it for debugging — pass `limit=` instead.

Embeddings run locally in the runner (CPU sentence-transformers). The full
candidate pool never goes to an API.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
PREFILTER_TOP_N = 80
INTERESTS_PATH = Path(__file__).resolve().parents[2] / "config" / "interests.md"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Shared by embed.py (interest scoring) and dedupe.py (pairwise
    similarity) so the model loads once per process and both stages agree
    on what "the embedding" means."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return np.asarray(_model().encode(texts, show_progress_bar=False), dtype=np.float32)


def _get_or_build_interest_vector(conn: sqlite3.Connection) -> np.ndarray:
    row = conn.execute("SELECT model, vector FROM interest_vector WHERE id = 1").fetchone()
    if row is not None and row["model"] == EMBED_MODEL_NAME:
        return np.frombuffer(row["vector"], dtype=np.float32)

    interests_text = INTERESTS_PATH.read_text(encoding="utf-8")
    vector = embed_texts([interests_text])[0]
    conn.execute(
        """
        INSERT INTO interest_vector (id, model, vector, updated_at) VALUES (1, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            model=excluded.model, vector=excluded.vector, updated_at=excluded.updated_at
        """,
        (EMBED_MODEL_NAME, vector.tobytes(), datetime.now(UTC).isoformat()),
    )
    conn.commit()
    return vector


def _story_canonical_items(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT s.id AS story_id, i.* FROM stories s
        JOIN items i ON i.id = s.canonical_item_id
        """
    ).fetchall()


def run(conn: sqlite3.Connection, limit: int | None = None) -> list[int]:
    """Embed every story's canonical item, score against the interest
    vector, return the top-N story ids. `limit` is for local debugging only —
    it shrinks the prefilter, it does not bypass it."""
    interest_vec = _get_or_build_interest_vector(conn)
    interest_unit = interest_vec / (np.linalg.norm(interest_vec) or 1.0)

    stories = _story_canonical_items(conn)
    if not stories:
        return []

    texts = [f"{row['title']} {row['abstract']}" for row in stories]
    vectors = embed_texts(texts)

    run_date = datetime.now(UTC).date().isoformat()
    scored = []
    for row, vec in zip(stories, vectors, strict=True):
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (item_id, model, vector) VALUES (?, ?, ?)",
            (row["id"], EMBED_MODEL_NAME, vec.tobytes()),
        )
        unit = vec / (np.linalg.norm(vec) or 1.0)
        score = float(unit @ interest_unit)
        scored.append((row["story_id"], row["id"], score))

    scored.sort(key=lambda t: t[2], reverse=True)
    top_n = PREFILTER_TOP_N if limit is None else limit
    kept_story_ids = []
    for rank, (story_id, item_id, score) in enumerate(scored):
        shown = rank < top_n
        if shown:
            kept_story_ids.append(story_id)
        conn.execute(
            """
            INSERT INTO considered_items (run_date, item_id, score, shown, reason)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(run_date, item_id) DO UPDATE SET score=excluded.score, shown=excluded.shown
            """,
            (run_date, item_id, score, int(shown), "interest-vector prefilter"),
        )
    conn.commit()
    print(f"[embed] scored {len(scored)} stories, kept top {len(kept_story_ids)}")
    return kept_story_ids


def todays_story_ids(conn: sqlite3.Connection, run_date: str | None = None) -> list[int]:
    """Recover the set of story ids that survived today's prefilter — the
    single source of truth for "today's candidates," used by every later
    stage (triage, summarize, brief, publish) so a high scorer from last
    week doesn't keep winning a slot forever."""
    run_date = run_date or datetime.now(UTC).date().isoformat()
    rows = conn.execute(
        """
        SELECT s.id FROM considered_items ci
        JOIN stories s ON s.canonical_item_id = ci.item_id
        WHERE ci.run_date = ? AND ci.shown = 1
        """,
        (run_date,),
    ).fetchall()
    return [row["id"] for row in rows]
