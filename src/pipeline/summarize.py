"""Stage 4: LLM summaries for the top ~35 stories (ranked by triage score).
Three sentences plus one "why this matters" line. Feed the abstract, never
the PDF — full text is a click-to-expand action, not part of the nightly run.

Same submit()/collect() split as triage.py, for the same batch-mechanics reason.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from src.llm.cost import record_and_check
from src.llm.routing import get_client_for_batch, poll_until_done, submit_stage_batch
from src.models import BatchHandle, LLMRequest, SummaryResult
from src.pipeline import triage

PROMPT_VERSION = "1"
# Front page shows a flat top ~8 (see publish.TOTAL_STORY_CAP) — this stays a
# bit above that so there's a real pool to pick the best from, not exactly 8.
TOP_N = 15
_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = _ROOT / "src" / "llm" / "prompts" / "summarize.md"

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "why_it_matters": {"type": "string"},
    },
    "required": ["summary", "why_it_matters"],
    "additionalProperties": False,
}


def _stories_needing_summary(conn: sqlite3.Connection, story_ids: list[int]) -> list[sqlite3.Row]:
    if not story_ids:
        return []
    placeholders = ",".join("?" for _ in story_ids)
    return conn.execute(
        f"""
        SELECT s.id AS story_id, i.*, tr.score AS triage_score
        FROM stories s
        JOIN items i ON i.id = s.canonical_item_id
        JOIN triage_results tr ON tr.content_hash = i.content_hash AND tr.prompt_version = ?
        WHERE s.id IN ({placeholders})
        AND i.content_hash NOT IN (
            SELECT content_hash FROM summaries WHERE prompt_version = ?
        )
        ORDER BY tr.score DESC
        LIMIT ?
        """,
        (triage.PROMPT_VERSION, *story_ids, PROMPT_VERSION, TOP_N),
    ).fetchall()


def submit(conn: sqlite3.Connection, story_ids: list[int]) -> BatchHandle | None:
    items = _stories_needing_summary(conn, story_ids)
    if not items:
        print("[summarize] nothing new to summarize (all cached or none triaged yet)")
        return None

    template = PROMPT_PATH.read_text(encoding="utf-8")
    requests = [
        LLMRequest(
            custom_id=f"summarize:{row['id']}",
            prompt=template.format(title=row["title"], abstract=row["abstract"][:4000]),
            json_schema=RESULT_SCHEMA,
        )
        for row in items
    ]
    handle = submit_stage_batch("summarize", requests, conn)
    print(f"[summarize] submitted {len(requests)} requests")
    return handle


def collect(
    conn: sqlite3.Connection, handle: BatchHandle, max_wait_s: float = 600, interval_s: float = 15
) -> list[SummaryResult] | None:
    """Returns None if the batch still hasn't finished after max_wait_s.

    Unlike triage, this stage is submitted fresh inside collect.py (it needs
    triage's scores first, so it can't go out with submit.py at 02:00) — so
    the whole wait happens here, bounded, rather than across the workflow gap.
    """
    client = get_client_for_batch(handle)
    result = poll_until_done(client, handle, max_wait_s, interval_s)
    if result is None:
        print(f"[summarize] batch {handle.external_id} still not done after {max_wait_s}s")
        return None

    now = datetime.now(UTC).isoformat()
    summaries: list[SummaryResult] = []
    for response in result:
        record_and_check(conn, response.cost)

        item_id = int(response.custom_id.split(":", 1)[1])
        row = conn.execute("SELECT content_hash FROM items WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            continue

        parsed = response.parsed or {}
        try:
            sr = SummaryResult(
                item_hash=row["content_hash"],
                prompt_version=PROMPT_VERSION,
                provider=response.cost.provider,
                model=response.cost.model,
                summary=parsed["summary"],
                why_it_matters=parsed["why_it_matters"],
                cost_usd=response.cost.usd,
            )
        except Exception as exc:  # noqa: BLE001 - malformed LLM output fails the item, not the run
            print(f"[summarize] item {item_id} failed schema validation, skipping: {exc}")
            continue

        conn.execute(
            """
            INSERT INTO summaries (
                content_hash, prompt_version, model, provider, summary,
                why_it_matters, cost_usd, created_at
            ) VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(content_hash, prompt_version, model) DO UPDATE SET
                summary=excluded.summary, why_it_matters=excluded.why_it_matters,
                cost_usd=excluded.cost_usd
            """,
            (
                sr.item_hash,
                sr.prompt_version,
                sr.model,
                sr.provider,
                sr.summary,
                sr.why_it_matters,
                sr.cost_usd,
                now,
            ),
        )
        summaries.append(sr)

    conn.execute(
        "UPDATE batches SET status='collected', collected_at=? "
        "WHERE stage='summarize' AND external_id=?",
        (now, handle.external_id),
    )
    conn.commit()
    print(f"[summarize] collected {len(summaries)} summaries from batch {handle.external_id}")
    return summaries
