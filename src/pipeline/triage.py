"""Stage 3: batched LLM scoring, ~10 items per call. Returns per item: a
relevance score 0-10, section tags (multiple allowed — tags, not hard
filters), one-line reason.

Split into submit()/collect() to match the two-workflow batch mechanic:
submit.py calls submit() at 02:00 UTC, collect.py calls collect() at 05:00
UTC once the batch has had time to process.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import yaml

from src.llm.cost import record_and_check
from src.llm.routing import get_client_for_batch, poll_until_done, submit_stage_batch
from src.models import BatchHandle, LLMRequest, Section, TriageResult

PROMPT_VERSION = "1"
BATCH_SIZE = 10
_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = _ROOT / "src" / "llm" / "prompts" / "triage.md"
INTERESTS_PATH = _ROOT / "config" / "interests.md"
TAXONOMY_PATH = _ROOT / "config" / "taxonomy.yaml"

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "score": {"type": "number"},
                    # Enum, not free string: strict schema mode can only
                    # keep the model inside the taxonomy if it's told what
                    # the taxonomy is.
                    "sections": {
                        "type": "array",
                        "items": {"type": "string", "enum": [s.value for s in Section]},
                    },
                    "reason": {"type": "string"},
                },
                "required": ["score", "sections", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


_B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def _to_b36(n: int) -> str:
    out = ""
    while True:
        n, r = divmod(n, 36)
        out = _B36[r] + out
        if n == 0:
            return out


def encode_custom_id(item_ids: list[int]) -> str:
    """custom_id must fit [A-Za-z0-9_-]{1,64} (see LLMRequest). Ten decimal
    ids overflow 64 chars once ids pass 5 digits; base36 keeps ten ids
    under 64 until ids pass ~1.7M."""
    return "triage-" + "_".join(_to_b36(i) for i in item_ids)


def decode_custom_id(custom_id: str) -> list[int]:
    if custom_id.startswith("triage:"):  # pre-2026-10-06 format, batches still in flight
        return [int(x) for x in custom_id.split(":", 1)[1].split(",")]
    return [int(x, 36) for x in custom_id.split("-", 1)[1].split("_")]


def _taxonomy_text() -> str:
    with open(TAXONOMY_PATH, encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)
    lines = []
    for group, sections in taxonomy.items():
        for tag, meta in sections.items():
            lines.append(f"- {tag} ({group}): {meta['description'].strip()}")
    return "\n".join(lines)


def _items_needing_triage(conn: sqlite3.Connection, story_ids: list[int]) -> list[sqlite3.Row]:
    if not story_ids:
        return []
    placeholders = ",".join("?" for _ in story_ids)
    return conn.execute(
        f"""
        SELECT s.id AS story_id, i.* FROM stories s
        JOIN items i ON i.id = s.canonical_item_id
        WHERE s.id IN ({placeholders})
        AND i.content_hash NOT IN (
            SELECT content_hash FROM triage_results WHERE prompt_version = ?
        )
        """,
        (*story_ids, PROMPT_VERSION),
    ).fetchall()


def submit(conn: sqlite3.Connection, story_ids: list[int]) -> BatchHandle | None:
    items = _items_needing_triage(conn, story_ids)
    if not items:
        print("[triage] nothing new to triage (all cached)")
        return None

    template = PROMPT_PATH.read_text(encoding="utf-8")
    interest_profile = INTERESTS_PATH.read_text(encoding="utf-8")
    taxonomy = _taxonomy_text()

    requests = []
    for batch_start in range(0, len(items), BATCH_SIZE):
        batch = items[batch_start : batch_start + BATCH_SIZE]
        items_text = "\n\n".join(
            f"[{idx}] Title: {row['title']}\nAbstract: {row['abstract'][:600]}"
            for idx, row in enumerate(batch)
        )
        prompt = template.format(
            interest_profile=interest_profile, taxonomy=taxonomy, items=items_text
        )
        custom_id = encode_custom_id([row["id"] for row in batch])
        requests.append(LLMRequest(custom_id=custom_id, prompt=prompt, json_schema=RESULT_SCHEMA))

    handle = submit_stage_batch("triage", requests, conn)
    print(f"[triage] submitted {len(requests)} batch calls covering {len(items)} items")
    return handle


def collect(
    conn: sqlite3.Connection, handle: BatchHandle, max_wait_s: float = 1200, interval_s: float = 20
) -> list[TriageResult] | None:
    """Returns None if the batch still hasn't finished after max_wait_s.

    By the time collect.py runs (3 hours after submit.py), the batch has
    usually had plenty of time — this bounded wait just absorbs the tail
    where it hasn't quite finished, rather than giving up on one poll.
    """
    client = get_client_for_batch(handle)
    result = poll_until_done(client, handle, max_wait_s, interval_s)
    if result is None:
        print(f"[triage] batch {handle.external_id} still not done after {max_wait_s}s")
        return None

    now = datetime.now(UTC).isoformat()
    triage_results: list[TriageResult] = []
    for response in result:
        record_and_check(conn, response.cost)

        item_ids = decode_custom_id(response.custom_id)
        entries = (response.parsed or {}).get("results", [])
        per_item_cost = response.cost.usd / max(len(entries), 1)

        for item_id, entry in zip(item_ids, entries, strict=False):
            row = conn.execute(
                "SELECT content_hash FROM items WHERE id = ?", (item_id,)
            ).fetchone()
            if row is None:
                continue
            # An off-taxonomy tag ("security", "privacy") fails the tag, not
            # the item — the score and the valid tags are still usable.
            raw_sections = entry.get("sections", [])
            sections = [s for s in raw_sections if s in Section._value2member_map_]
            if dropped := [s for s in raw_sections if s not in Section._value2member_map_]:
                print(f"[triage] item {item_id}: dropped unknown section tags {dropped}")
            try:
                tr = TriageResult(
                    item_hash=row["content_hash"],
                    prompt_version=PROMPT_VERSION,
                    provider=response.cost.provider,
                    model=response.cost.model,
                    score=float(entry.get("score", 0)),
                    sections=sections,
                    reason=entry.get("reason", ""),
                    cost_usd=per_item_cost,
                )
            except Exception as exc:  # noqa: BLE001 - malformed LLM output fails the item, not the run
                print(f"[triage] item {item_id} failed schema validation, skipping: {exc}")
                continue
            conn.execute(
                """
                INSERT INTO triage_results (
                    content_hash, prompt_version, model, provider, score,
                    sections_json, reason, cost_usd, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(content_hash, prompt_version, model) DO UPDATE SET
                    score=excluded.score, sections_json=excluded.sections_json,
                    reason=excluded.reason, cost_usd=excluded.cost_usd
                """,
                (
                    tr.item_hash,
                    tr.prompt_version,
                    tr.model,
                    tr.provider,
                    tr.score,
                    json.dumps([s.value if hasattr(s, "value") else s for s in tr.sections]),
                    tr.reason,
                    tr.cost_usd,
                    now,
                ),
            )
            triage_results.append(tr)

    conn.execute(
        "UPDATE batches SET status='collected', collected_at=? "
        "WHERE stage='triage' AND external_id=?",
        (now, handle.external_id),
    )
    conn.commit()
    print(f"[triage] collected {len(triage_results)} results from batch {handle.external_id}")
    return triage_results
