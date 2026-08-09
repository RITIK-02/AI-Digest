"""SQLite state. data/digest.db, committed to the repo. It holds every item
ever seen, its embedding, its scores, and its summary — the dedup key store
and the cost cache. See CLAUDE.md "Non-negotiable architecture".

Schema init is idempotent: every statement is CREATE TABLE IF NOT EXISTS, so
calling init_schema() on an existing db is a no-op.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "digest.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    raw_id TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL DEFAULT '',
    authors_json TEXT NOT NULL DEFAULT '[]',
    published_at TEXT,
    fetched_at TEXT NOT NULL,
    arxiv_id TEXT,
    doi TEXT,
    openreview_id TEXT,
    github_repo TEXT,
    content_hash TEXT NOT NULL,
    raw_payload_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(source, raw_id)
);
CREATE INDEX IF NOT EXISTS idx_items_content_hash ON items(content_hash);
CREATE INDEX IF NOT EXISTS idx_items_arxiv_id ON items(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_items_doi ON items(doi);

CREATE TABLE IF NOT EXISTS stories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_item_id INTEGER NOT NULL REFERENCES items(id),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS story_items (
    story_id INTEGER NOT NULL REFERENCES stories(id),
    item_id INTEGER NOT NULL REFERENCES items(id),
    role TEXT NOT NULL DEFAULT 'discussion',
    PRIMARY KEY (story_id, item_id)
);

CREATE TABLE IF NOT EXISTS embeddings (
    item_id INTEGER PRIMARY KEY REFERENCES items(id),
    model TEXT NOT NULL,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS interest_vector (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    model TEXT NOT NULL,
    vector BLOB NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS triage_results (
    content_hash TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    score REAL NOT NULL,
    sections_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    cost_usd REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (content_hash, prompt_version, model)
);

CREATE TABLE IF NOT EXISTS summaries (
    content_hash TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    summary TEXT NOT NULL,
    why_it_matters TEXT NOT NULL,
    cost_usd REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (content_hash, prompt_version, model)
);

CREATE TABLE IF NOT EXISTS briefs (
    run_date TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    content_md TEXT NOT NULL,
    cost_usd REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_date, provider)
);

CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stage TEXT NOT NULL,
    provider TEXT NOT NULL,
    external_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'submitted',
    submitted_at TEXT NOT NULL,
    collected_at TEXT
);

CREATE TABLE IF NOT EXISTS cost_log (
    run_date TEXT NOT NULL,
    stage TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    usd REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS considered_items (
    run_date TEXT NOT NULL,
    item_id INTEGER NOT NULL REFERENCES items(id),
    score REAL NOT NULL,
    shown INTEGER NOT NULL DEFAULT 0,
    reason TEXT,
    PRIMARY KEY (run_date, item_id)
);
"""


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()
