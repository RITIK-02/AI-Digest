"""Entry point for the 02:00 UTC `submit` workflow.

Fetch sources, dedupe, embed/prefilter, submit the triage batch. Summarize
and brief can't be submitted yet — they depend on triage scores, which won't
exist until collect.py resolves the batch three hours later. Nothing here
touches the frontend or sends email; that is collect.py's job.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()  # local dev reads .env; in CI, real env vars are already set and this is a no-op

from src.pipeline import db, dedupe, embed, ingest, triage  # noqa: E402


def main() -> None:
    conn = db.connect()
    try:
        ingest.run(conn)
        dedupe.run(conn)
        story_ids = embed.run(conn)
        if not story_ids:
            print("[submit] no stories survived the prefilter — nothing to triage today")
            return
        triage.submit(conn, story_ids)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
