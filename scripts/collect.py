"""Entry point for the 05:00 UTC `collect` workflow.

Resolves the triage batch submit.py submitted three hours earlier, then runs
the rest of the pipeline: submit + poll summarize (bounded wait — it can't go
out any earlier, it needs triage's scores), generate the brief, publish site
data and send the email. Also resolves any pending dedupe-adjudication
batches, which corrects the story graph for future runs even though it
arrives too late to affect today's.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()  # local dev reads .env; in CI, real env vars are already set and this is a no-op

from src.models import BatchHandle  # noqa: E402
from src.pipeline import brief, db, dedupe, embed, publish, summarize, triage  # noqa: E402


def _pending_handle(conn, stage: str) -> BatchHandle | None:
    row = conn.execute(
        "SELECT provider, external_id, submitted_at FROM batches "
        "WHERE stage = ? AND status = 'submitted' ORDER BY submitted_at DESC LIMIT 1",
        (stage,),
    ).fetchone()
    if row is None:
        return None
    return BatchHandle(
        provider=row["provider"],
        stage=stage,
        external_id=row["external_id"],
        submitted_at=row["submitted_at"],
    )


def main() -> None:
    conn = db.connect()
    try:
        dedupe.collect_adjudications(conn)

        triage_handle = _pending_handle(conn, "triage")
        if triage_handle is None:
            print("[collect] no pending triage batch found — did submit.py run today?")
        else:
            results = triage.collect(conn, triage_handle)
            if results is None:
                print("[collect] triage batch not finished yet — aborting, retry next collect run")
                return

        story_ids = embed.todays_story_ids(conn)
        if not story_ids:
            print("[collect] no stories from today's prefilter — nothing to summarize or publish")
            return

        summarize_handle = summarize.submit(conn, story_ids)
        if summarize_handle is not None:
            summaries = summarize.collect(conn, summarize_handle)
            if summaries is None:
                print("[collect] summarize batch timed out — publishing with existing summaries")

        brief.generate(conn, story_ids)

        publish.run(conn, story_ids)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
