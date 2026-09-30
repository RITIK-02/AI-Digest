"""Pure parser tests against a fixture — no network, per CLAUDE.md's own rule
against tests that hit live APIs."""

import json
from pathlib import Path

from src.sources.hn import parse_hits

FIXTURE = Path(__file__).parent / "fixtures" / "hn_sample_response.json"


def _load_hits() -> list[dict]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data["hits"]


def test_parse_hits_returns_expected_items():
    items = parse_hits(_load_hits())

    # Third fixture hit has no objectID and must be dropped, not crash.
    assert len(items) == 2

    arxiv_linked = items[0]
    assert arxiv_linked.source == "hn"
    assert arxiv_linked.raw_id == "12345678"
    assert arxiv_linked.url == "https://arxiv.org/abs/2401.12345"
    assert arxiv_linked.authors == ["someresearcher"]
    assert arxiv_linked.raw_payload["points"] == 245
    assert arxiv_linked.raw_payload["num_comments"] == 87
    assert arxiv_linked.raw_payload["hn_url"] == "https://news.ycombinator.com/item?id=12345678"

    self_post = items[1]
    # No external url -> the HN discussion page itself is the item's url.
    assert self_post.url == "https://news.ycombinator.com/item?id=23456789"
    assert self_post.abstract == (
        "A small side project for poking at attention patterns in open models."
    )


def test_parse_hits_handles_empty_list():
    assert parse_hits([]) == []
