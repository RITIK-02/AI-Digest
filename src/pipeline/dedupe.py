"""Stage 1: cluster items into stories.

Strong keys first (arxiv_id, doi, normalized URL). Then title normalization.
Then embedding cosine above 0.85. Borderline (0.78-0.85) goes to a small LLM
adjudication batch, capped at 15 calls/run.

Two phases: first, candidate items get attached to an *existing* story if
they share a strong key with something already clustered (a new version of a
paper joins its existing story, it doesn't start a new one). Second, whatever
is left gets clustered among itself via union-find and turned into new
stories.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime

import numpy as np

from src.llm.routing import get_client_for_batch, submit_stage_batch
from src.models import BatchHandle, DedupeAdjudication, LLMRequest, Pending
from src.pipeline.embed import embed_texts

PROMPT_VERSION = "1"
STRONG_MERGE_THRESHOLD = 0.85
ADJUDICATION_LOW = 0.78
MAX_ADJUDICATION_CALLS = 15

_PROMPT_PATH = (
    __import__("pathlib").Path(__file__).resolve().parents[2]
    / "src"
    / "llm"
    / "prompts"
    / "dedupe_adjudication.md"
)


def normalize_title(title: str) -> str:
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


class UnionFind:
    def __init__(self, ids: list[int]):
        self.parent = {i: i for i in ids}

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for x in self.parent:
            out.setdefault(self.find(x), []).append(x)
        return out


def _candidate_items(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT i.* FROM items i
        LEFT JOIN story_items si ON si.item_id = i.id
        WHERE si.item_id IS NULL
        """
    ).fetchall()


def _attach_to_existing_story(conn: sqlite3.Connection, item: sqlite3.Row) -> int | None:
    """If this item shares a strong key with something already in a story,
    attach it there instead of clustering it fresh. Returns the story id if attached."""
    for key_col in ("arxiv_id", "doi"):
        key_val = item[key_col]
        if not key_val:
            continue
        row = conn.execute(
            f"""
            SELECT si.story_id FROM story_items si
            JOIN items other ON other.id = si.item_id
            WHERE other.{key_col} = ? AND other.id != ?
            LIMIT 1
            """,
            (key_val, item["id"]),
        ).fetchone()
        if row:
            conn.execute(
                "INSERT OR IGNORE INTO story_items (story_id, item_id, role) "
                "VALUES (?, ?, 'discussion')",
                (row["story_id"], item["id"]),
            )
            return row["story_id"]
    return None


def _cluster_remaining(items: list[sqlite3.Row]) -> tuple[UnionFind, list[tuple[int, int, float]]]:
    ids = [row["id"] for row in items]
    uf = UnionFind(ids)

    # Strong keys among the remaining candidates themselves.
    for key_col in ("arxiv_id", "doi"):
        seen: dict[str, int] = {}
        for row in items:
            val = row[key_col]
            if not val:
                continue
            if val in seen:
                uf.union(seen[val], row["id"])
            else:
                seen[val] = row["id"]

    # Title normalization.
    seen_titles: dict[str, int] = {}
    for row in items:
        norm = normalize_title(row["title"])
        if not norm:
            continue
        if norm in seen_titles:
            uf.union(seen_titles[norm], row["id"])
        else:
            seen_titles[norm] = row["id"]

    # Embedding cosine for whatever strong keys/titles didn't already merge.
    borderline_pairs: list[tuple[int, int, float]] = []
    if len(items) > 1:
        texts = [f"{row['title']} {row['abstract']}" for row in items]
        vectors = embed_texts(texts)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = vectors / norms
        sim = unit @ unit.T

        n = len(ids)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = ids[i], ids[j]
                if uf.find(a) == uf.find(b):
                    continue
                score = float(sim[i, j])
                if score >= STRONG_MERGE_THRESHOLD:
                    uf.union(a, b)
                elif score >= ADJUDICATION_LOW:
                    borderline_pairs.append((a, b, score))

    return uf, borderline_pairs


def _adjudicate(
    conn: sqlite3.Connection, pairs: list[tuple[int, int, float]], by_id: dict[int, sqlite3.Row]
) -> list[tuple[int, int]]:
    """Send up to MAX_ADJUDICATION_CALLS borderline pairs to a cheap LLM.
    Returns the (a, b) pairs judged to be the same story."""
    if not pairs:
        return []
    pairs = pairs[:MAX_ADJUDICATION_CALLS]
    template = _PROMPT_PATH.read_text(encoding="utf-8")

    requests = []
    for a, b, _score in pairs:
        item_a, item_b = by_id[a], by_id[b]
        prompt = template.format(
            item_a_title=item_a["title"],
            item_a_abstract=item_a["abstract"][:500],
            item_a_source=item_a["source"],
            item_b_title=item_b["title"],
            item_b_abstract=item_b["abstract"][:500],
            item_b_source=item_b["source"],
        )
        requests.append(
            LLMRequest(
                custom_id=f"{a}:{b}",
                prompt=prompt,
                json_schema=DedupeAdjudication.model_json_schema(),
            )
        )

    handle = submit_stage_batch("dedupe_adjudication", requests, conn)
    print(
        f"[dedupe] submitted {len(requests)} adjudication requests via "
        f"{handle.provider} (batch {handle.external_id}) — collect.py will resolve these"
    )
    # Batch results aren't available synchronously; nothing to merge yet this run.
    return []


def run(conn: sqlite3.Connection) -> list[int]:
    candidates = _candidate_items(conn)
    remaining = []
    story_ids: list[int] = []

    for item in candidates:
        story_id = _attach_to_existing_story(conn, item)
        if story_id is not None:
            story_ids.append(story_id)
        else:
            remaining.append(item)

    if remaining:
        by_id = {row["id"]: row for row in remaining}
        uf, borderline = _cluster_remaining(remaining)
        if borderline:
            try:
                _adjudicate(conn, borderline, by_id)  # async: resolved by collect.py
            except Exception as exc:  # noqa: BLE001 - adjudication is a refinement, not a
                # blocker: losing it just leaves borderline pairs as separate stories for
                # now (collect_adjudications retries next run) rather than failing the
                # whole dedupe stage — a partial digest beats a failed workflow.
                print(f"[dedupe] adjudication submission failed, skipping this run: {exc}")

        now = datetime.now(UTC).isoformat()
        for _root, member_ids in uf.groups().items():
            members = [by_id[i] for i in member_ids]
            # Canonical = the one with an arxiv_id if any, else earliest fetched.
            canonical = next((m for m in members if m["arxiv_id"]), None) or min(
                members, key=lambda m: m["fetched_at"]
            )
            cur = conn.execute(
                "INSERT INTO stories (canonical_item_id, created_at) VALUES (?, ?)",
                (canonical["id"], now),
            )
            story_id = cur.lastrowid
            for m in members:
                role = "canonical" if m["id"] == canonical["id"] else "discussion"
                conn.execute(
                    "INSERT OR IGNORE INTO story_items (story_id, item_id, role) VALUES (?, ?, ?)",
                    (story_id, m["id"], role),
                )
            story_ids.append(story_id)

    conn.commit()
    print(f"[dedupe] {len(candidates)} candidate items -> {len(set(story_ids))} stories touched")
    return story_ids


def collect_adjudications(conn: sqlite3.Connection) -> int:
    """Resolve any pending dedupe_adjudication batches submitted by a prior
    run() call, merging story pairs judged the same story.

    This resolves too late to affect the run that submitted it (embed/triage
    for that day already ran against the unmerged stories) — it corrects the
    story graph for every run after. Called from collect.py.
    """
    pending = conn.execute(
        "SELECT provider, external_id, submitted_at FROM batches "
        "WHERE stage = 'dedupe_adjudication' AND status = 'submitted'"
    ).fetchall()
    if not pending:
        return 0

    now = datetime.now(UTC).isoformat()
    merged = 0
    for row in pending:
        handle = BatchHandle(
            provider=row["provider"],
            stage="dedupe_adjudication",
            external_id=row["external_id"],
            submitted_at=row["submitted_at"],
        )
        client = get_client_for_batch(handle)
        result = client.poll_batch(handle)
        if isinstance(result, Pending):
            continue

        for response in result:
            a_id, b_id = (int(x) for x in response.custom_id.split(":"))
            if not (response.parsed or {}).get("same_story"):
                continue
            story_a = conn.execute(
                "SELECT story_id FROM story_items WHERE item_id = ?", (a_id,)
            ).fetchone()
            story_b = conn.execute(
                "SELECT story_id FROM story_items WHERE item_id = ?", (b_id,)
            ).fetchone()
            if not story_a or not story_b or story_a["story_id"] == story_b["story_id"]:
                continue
            conn.execute(
                "UPDATE story_items SET story_id = ? WHERE story_id = ?",
                (story_a["story_id"], story_b["story_id"]),
            )
            conn.execute("DELETE FROM stories WHERE id = ?", (story_b["story_id"],))
            merged += 1

        conn.execute(
            "UPDATE batches SET status='collected', collected_at=? "
            "WHERE stage='dedupe_adjudication' AND external_id=?",
            (now, handle.external_id),
        )

    conn.commit()
    if merged:
        print(f"[dedupe] merged {merged} story pair(s) from adjudication")
    return merged
