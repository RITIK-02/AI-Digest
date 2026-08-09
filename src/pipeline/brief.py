"""Stage 5: the daily prose brief. ~8 items, written for a researcher
skimming over coffee. Must include one item that cuts against the apparent
interest profile, flagged as such — or say so in one line if none qualify.
The brief is the product: if quality has to go somewhere, it goes here.

Unlike triage/summarize, this runs synchronously inside collect.py — it
needs the just-collected summaries and publish needs its output immediately
after, so there's no later step to collect an async batch from. It still
goes through the batch API for consistency and the discount, but collect.py
polls it in a bounded loop (poll_until_done) instead of across the
submit/collect workflow split.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from src.llm.cost import record_and_check
from src.llm.routing import get_client_for_batch, poll_until_done, submit_stage_batch
from src.models import Brief, LLMRequest
from src.pipeline import summarize

PROMPT_VERSION = "1"
_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = _ROOT / "src" / "llm" / "prompts" / "brief.md"
MAX_WAIT_SECONDS = 600
POLL_INTERVAL_SECONDS = 15


def _candidate_stories(conn: sqlite3.Connection, story_ids: list[int]) -> list[sqlite3.Row]:
    if not story_ids:
        return []
    placeholders = ",".join("?" for _ in story_ids)
    return conn.execute(
        f"""
        SELECT s.id AS story_id, i.title, i.url, sm.summary, sm.why_it_matters,
               tr.score, tr.sections_json, tr.reason
        FROM stories s
        JOIN items i ON i.id = s.canonical_item_id
        JOIN summaries sm ON sm.content_hash = i.content_hash AND sm.prompt_version = ?
        JOIN triage_results tr ON tr.content_hash = i.content_hash
        WHERE s.id IN ({placeholders})
        ORDER BY tr.score DESC
        """,
        (summarize.PROMPT_VERSION, *story_ids),
    ).fetchall()


def generate(conn: sqlite3.Connection, story_ids: list[int]) -> Brief | None:
    candidates = _candidate_stories(conn, story_ids)
    if not candidates:
        print("[brief] no summarized stories to work from")
        return None

    template = PROMPT_PATH.read_text(encoding="utf-8")
    stories_text = "\n\n".join(
        f"- {row['title']} ({row['url']})\n"
        f"  Summary: {row['summary']}\n"
        f"  Why it matters: {row['why_it_matters']}\n"
        f"  Sections: {row['sections_json']} | score: {row['score']:.1f} | {row['reason']}"
        for row in candidates
    )
    prompt = template.format(stories=stories_text)

    handle = submit_stage_batch("brief", [LLMRequest(custom_id="brief:today", prompt=prompt)], conn)
    client = get_client_for_batch(handle)
    result = poll_until_done(client, handle, MAX_WAIT_SECONDS, POLL_INTERVAL_SECONDS)

    now = datetime.now(UTC).isoformat()
    if result is None:
        print(
            f"[brief] batch {handle.external_id} did not finish within {MAX_WAIT_SECONDS}s — "
            "publishing without a brief this run rather than blocking indefinitely"
        )
        return None
    if not result:
        print("[brief] batch finished with no usable response")
        return None

    response = result[0]
    record_and_check(conn, response.cost)

    brief = Brief(
        date=datetime.now(UTC).date().isoformat(),
        provider=response.cost.provider,
        model=response.cost.model,
        content_md=response.raw_text,
        cost_usd=response.cost.usd,
    )
    conn.execute(
        """
        INSERT INTO briefs (run_date, provider, model, content_md, cost_usd, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_date, provider) DO UPDATE SET
            model=excluded.model, content_md=excluded.content_md, cost_usd=excluded.cost_usd
        """,
        (brief.date, brief.provider, brief.model, brief.content_md, brief.cost_usd, now),
    )
    conn.execute(
        "UPDATE batches SET status='collected', collected_at=? "
        "WHERE stage='brief' AND external_id=?",
        (now, handle.external_id),
    )
    conn.commit()
    print(f"[brief] generated via {brief.provider}/{brief.model}, ${brief.cost_usd:.4f}")
    return brief
