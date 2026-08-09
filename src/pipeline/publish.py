"""Stage 6: render the site data + email payload, send the email.

Front page is a single flat top-N list (TOTAL_STORY_CAP), ranked by triage
score across every section at once, each story still carrying its full list
of section tags as metadata. This replaced an earlier section-first design
(up to 5 items independently per section, 15 sections — 70+ possible slots)
that was simply too much to read daily. See _flat_top_stories().

The section-first machinery (_section_candidates, _select_top, per-section
slot budget) is NOT deleted — it still runs and its output is kept in the
JSON under "sections"/"archive" as a full "browse everything, organized by
topic" view (see archive.astro), and it's the natural place to hang a real
per-section front page again later if a flat list turns out too coarse.
Keep it working; just don't feed it to the front page.

Slot budget per section is fixed slots, not score bonuses:
- 70% relevance-ranked against the interest vector
- 20% adjacent: high embedding distance from the centroid *plus* an
  independent quality signal (HN traction, citation velocity, oral/spotlight
  at a top venue). Distance alone is noise — never fill on distance alone.
- 10% wildcard: cited-by adjacency from saved items.
The flat top-N list reuses the same relevance-slot/epsilon-greedy logic,
just with one shared budget instead of one per section.

Every non-relevance item renders a one-line reason — without it, exploration
is indistinguishable from a ranking bug.

With only arXiv as a source this pass, there is no HN/citation signal to
satisfy the adjacent slot's "independent quality signal" requirement, and no
saved-item history for wildcard slots (cold start, no clicks yet) — so those
two slots render empty. That is the mechanism working as specified against a
thin data diet, not a shortcut. `rising`, `deadlines`, and `jobs` are
similarly unfilled: they need sources (HN/social velocity, a curated CFP
list, a jobs feed) this pass doesn't have. `has-code` and `watchlist` ARE
fully implemented — both work off data arXiv already provides, and stay in
the section browse view (they're filters/curated lists, not relevance-ranked,
so they don't belong in the flat top-N).
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = _ROOT / "config" / "taxonomy.yaml"
INTERESTS_PATH = _ROOT / "config" / "interests.md"
SITE_DATA_PATH = _ROOT / "site" / "src" / "data" / "digest.json"

TOTAL_STORY_CAP = 8  # front page: flat top-N across every section combined
SECTION_CAP = 5  # per-section cap in the "browse by section" archive view
RELEVANCE_SHARE = 0.7
ADJACENT_SHARE = 0.2
EPSILON = 0.1  # epsilon-greedy exploration within the relevance slot
MIN_SCORE_FOR_SECTION = 4.0  # below this, an item doesn't belong in the section at all

_NO_SOURCE_SECTIONS = {
    "rising": "needs HN points / social velocity / citation data — no such source yet",
    "deadlines": "needs a curated CFP list plus deadline detection — not built yet",
    "jobs": "needs a jobs feed — not built yet",
}


def _load_taxonomy() -> dict[str, dict]:
    with open(TAXONOMY_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    flat = {}
    for group, sections in data.items():
        for tag, meta in sections.items():
            flat[tag] = {"group": group, **meta}
    return flat


def _flat_candidates(conn: sqlite3.Connection, story_ids: list[int]) -> list[dict]:
    """Every triaged story above threshold, regardless of section — the pool
    the flat front-page top-N is drawn from."""
    if not story_ids:
        return []
    placeholders = ",".join("?" for _ in story_ids)
    rows = conn.execute(
        f"""
        SELECT s.id AS story_id, i.id AS item_id, i.title, i.url,
               tr.score, tr.reason, tr.sections_json,
               sm.summary, sm.why_it_matters
        FROM stories s
        JOIN items i ON i.id = s.canonical_item_id
        JOIN triage_results tr ON tr.content_hash = i.content_hash
        LEFT JOIN summaries sm ON sm.content_hash = i.content_hash
        WHERE s.id IN ({placeholders}) AND tr.score >= ?
        """,
        (*story_ids, MIN_SCORE_FOR_SECTION),
    ).fetchall()
    out = [dict(row) for row in rows]
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


def _section_candidates(conn: sqlite3.Connection, section: str, story_ids: list[int]) -> list[dict]:
    if not story_ids:
        return []
    placeholders = ",".join("?" for _ in story_ids)
    rows = conn.execute(
        f"""
        SELECT s.id AS story_id, i.id AS item_id, i.title, i.url,
               tr.score, tr.reason, tr.sections_json,
               sm.summary, sm.why_it_matters
        FROM stories s
        JOIN items i ON i.id = s.canonical_item_id
        JOIN triage_results tr ON tr.content_hash = i.content_hash
        LEFT JOIN summaries sm ON sm.content_hash = i.content_hash
        WHERE s.id IN ({placeholders})
        """,
        story_ids,
    ).fetchall()
    out = []
    for row in rows:
        if section not in json.loads(row["sections_json"]):
            continue
        if row["score"] < MIN_SCORE_FOR_SECTION:
            continue
        out.append(dict(row))
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


def _select_top(candidates: list[dict], cap: int) -> tuple[list[dict], list[dict]]:
    """Returns (selected, overflow) for a given cap, shared by the flat
    front-page list (cap=TOTAL_STORY_CAP) and each section's browse-view
    slice (cap=SECTION_CAP)."""
    if not candidates:
        return [], []

    n_relevance = round(cap * RELEVANCE_SHARE)
    selected: list[dict] = []
    selected_ids: set[int] = set()

    pool = list(candidates)
    for _ in range(min(n_relevance, len(pool))):
        remaining = [c for c in pool if c["item_id"] not in selected_ids]
        if not remaining:
            break
        explore = random.random() < EPSILON and len(remaining) > 1
        pick = random.choice(remaining) if explore else remaining[0]
        pick = dict(pick)
        pick["slot"] = "relevance"
        selected.append(pick)
        selected_ids.add(pick["item_id"])

    # Adjacent slot (target ~20% of the cap) and wildcard slot (~10%) are
    # wired into the taxonomy but need a quality signal / saved-item history
    # this pass doesn't have — see module docstring. They render empty
    # rather than filling on distance or recency alone.

    overflow = [c for c in candidates if c["item_id"] not in selected_ids]
    return selected, overflow


def _has_code_section(conn: sqlite3.Connection, story_ids: list[int]) -> list[dict]:
    """Filter, not a ranker: papers with a linked repo, in-area, sorted by relevance."""
    if not story_ids:
        return []
    placeholders = ",".join("?" for _ in story_ids)
    rows = conn.execute(
        f"""
        SELECT s.id AS story_id, i.id AS item_id, i.title, i.url, i.github_repo,
               tr.score, tr.reason, sm.summary, sm.why_it_matters
        FROM stories s
        JOIN items i ON i.id = s.canonical_item_id
        JOIN triage_results tr ON tr.content_hash = i.content_hash
        LEFT JOIN summaries sm ON sm.content_hash = i.content_hash
        WHERE i.github_repo IS NOT NULL AND tr.score >= ? AND s.id IN ({placeholders})
        ORDER BY tr.score DESC
        LIMIT ?
        """,
        (MIN_SCORE_FOR_SECTION, *story_ids, SECTION_CAP),
    ).fetchall()
    return [dict(r, slot="has-code") for r in rows]


def _parse_watchlist_authors() -> list[str]:
    text = INTERESTS_PATH.read_text(encoding="utf-8")
    if "## Watchlist" not in text:
        return []
    section = text.split("## Watchlist", 1)[1].split("\n## ", 1)[0]
    names = []
    for line in section.splitlines():
        line = line.strip()
        if line.startswith("- ") and "none yet" not in line.lower():
            names.append(line[2:].strip())
    return names


def _watchlist_section(conn: sqlite3.Connection, story_ids: list[int]) -> list[dict]:
    """Named authors surface regardless of score — no ranking, no threshold."""
    authors = set(_parse_watchlist_authors())
    if not authors or not story_ids:
        return []
    placeholders = ",".join("?" for _ in story_ids)
    rows = conn.execute(
        f"""
        SELECT s.id AS story_id, i.id AS item_id, i.title, i.url, i.authors_json,
               tr.score, tr.reason, sm.summary, sm.why_it_matters
        FROM stories s
        JOIN items i ON i.id = s.canonical_item_id
        LEFT JOIN triage_results tr ON tr.content_hash = i.content_hash
        LEFT JOIN summaries sm ON sm.content_hash = i.content_hash
        WHERE s.id IN ({placeholders})
        """,
        story_ids,
    ).fetchall()
    matched = [
        dict(r, slot="watchlist") for r in rows if set(json.loads(r["authors_json"])) & authors
    ]
    return matched[:SECTION_CAP]


def _render_item(row: dict) -> dict:
    sections_json = row.get("sections_json")
    return {
        "title": row.get("title"),
        "url": row.get("url"),
        "summary": row.get("summary"),
        "why_it_matters": row.get("why_it_matters"),
        "score": row.get("score"),
        "reason": row.get("reason"),
        "slot": row.get("slot", "relevance"),
        "sections": json.loads(sections_json) if sections_json else [],
    }


def build_digest(conn: sqlite3.Connection, story_ids: list[int]) -> dict:
    run_date = datetime.now(UTC).date().isoformat()
    taxonomy = _load_taxonomy()

    brief_row = conn.execute(
        "SELECT * FROM briefs WHERE run_date = ? ORDER BY created_at DESC LIMIT 1", (run_date,)
    ).fetchone()

    # Front page: one flat top-N list, shared budget across every section.
    flat_candidates = _flat_candidates(conn, story_ids)
    stories, stories_overflow = _select_top(flat_candidates, TOTAL_STORY_CAP)

    # Browse view: the full section-by-section breakdown, kept for archive.astro
    # and for reviving a per-section front page later — see module docstring.
    sections_out: dict[str, dict] = {}
    archive_out: dict[str, list[dict]] = {}

    for tag, meta in taxonomy.items():
        section_overflow: list[dict] = []
        if tag == "has-code":
            items = _has_code_section(conn, story_ids)
        elif tag == "watchlist":
            items = _watchlist_section(conn, story_ids)
        elif tag in _NO_SOURCE_SECTIONS:
            items = []
        else:
            candidates = _section_candidates(conn, tag, story_ids)
            items, section_overflow = _select_top(candidates, SECTION_CAP)

        sections_out[tag] = {
            "group": meta["group"],
            "description": meta["description"].strip(),
            "items": [_render_item(r) for r in items],
            "unavailable_reason": _NO_SOURCE_SECTIONS.get(tag),
        }
        if section_overflow:
            archive_out[tag] = [_render_item(r) for r in section_overflow]

    return {
        "date": run_date,
        "brief": dict(brief_row) if brief_row else None,
        "stories": [_render_item(r) for r in stories],
        "stories_overflow": [_render_item(r) for r in stories_overflow],
        "sections": sections_out,
        "archive": archive_out,
    }


def write_site_data(digest: dict, path: Path = SITE_DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(digest, indent=2, default=str), encoding="utf-8")


def render_email_html(digest: dict) -> str:
    if digest.get("brief"):
        brief_html = f"<p>{digest['brief']['content_md']}</p>"
    else:
        brief_html = "<p>No brief today.</p>"
    items_html = "".join(
        f"<li><a href='{it['url']}'>{it['title']}</a>"
        f"{' — ' + it['summary'] if it.get('summary') else ''}"
        f"{' <i>(' + ', '.join(it['sections']) + ')</i>' if it.get('sections') else ''}</li>"
        for it in digest.get("stories", [])
    )
    stories_html = f"<h3>Today's stories</h3><ul>{items_html}</ul>" if items_html else ""
    return (
        f"<html><body><h1>AI Digest — {digest['date']}</h1>"
        f"{brief_html}{stories_html}</body></html>"
    )


def _unquote(value: str, name: str) -> str:
    """Strip whitespace and any surrounding quote pair from an env value.

    .env quoting is a dotenv convention that dotenv itself removes, but the
    GitHub Actions secrets UI stores the literal bytes you paste. Copying a
    value straight out of .env therefore lands the quotes in the secret, and
    Resend 422s on the resulting unparseable address.
    """
    cleaned = value.strip()
    for quote in ('"', "'"):
        if len(cleaned) >= 2 and cleaned.startswith(quote) and cleaned.endswith(quote):
            cleaned = cleaned[1:-1].strip()
            print(f"[publish] {name} had surrounding {quote} quotes — stripped; fix the secret")
            break
    return cleaned


def send_email(digest: dict) -> None:
    """Send the digest email. Never raises — the site data is already written
    and the LLM work is already paid for by this point, so a mail failure must
    not take down the run and discard the night's output."""
    api_key = os.environ.get("RESEND_API_KEY")
    from_addr = os.environ.get("RESEND_FROM")
    to_addr = os.environ.get("RESEND_TO")
    if not (api_key and from_addr and to_addr):
        print("[publish] RESEND_API_KEY/RESEND_FROM/RESEND_TO not set — skipping email send")
        return

    # RESEND_TO may hold several comma-separated addresses; Resend wants them
    # as separate list entries and 422s on a single string containing commas.
    from_addr = _unquote(from_addr, "RESEND_FROM")
    recipients = [
        _unquote(addr, "RESEND_TO") for addr in to_addr.split(",") if _unquote(addr, "RESEND_TO")
    ]

    try:
        resp = httpx.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "from": from_addr.strip(),
                "to": recipients,
                "subject": f"AI Digest — {digest['date']}",
                "html": render_email_html(digest),
            },
            timeout=30.0,
        )
    except httpx.HTTPError as exc:
        print(f"[publish] email send failed: {exc} — digest still published")
        return

    if resp.is_error:
        # Resend names the offending field in the body; without it a 422 is
        # undiagnosable from the log alone.
        print(f"[publish] email send failed: HTTP {resp.status_code} {resp.text}")
        return

    print("[publish] email sent")


def run(conn: sqlite3.Connection, story_ids: list[int], send: bool = True) -> dict:
    digest = build_digest(conn, story_ids)
    write_site_data(digest)
    if send:
        send_email(digest)
    print(
        f"[publish] wrote site data for {digest['date']}: "
        f"{len(digest['stories'])} stories on the front page"
    )
    return digest
