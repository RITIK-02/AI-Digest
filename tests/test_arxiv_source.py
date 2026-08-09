"""Pure parser tests against a fixture — no network, per CLAUDE.md's own rule
against tests that hit live APIs."""

from pathlib import Path

from src.sources.arxiv import parse_atom_feed

FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_sample_response.xml"


def test_parse_atom_feed_returns_expected_items():
    xml_text = FIXTURE.read_text(encoding="utf-8")
    items = parse_atom_feed(xml_text)

    assert len(items) == 2

    first = items[0]
    assert first.source == "arxiv"
    assert first.raw_id == "2401.12345v2"
    assert first.url == "http://arxiv.org/abs/2401.12345v2"
    assert "Jailbreaking" in first.title
    assert first.authors == ["Alice Example", "Bob Researcher"]
    assert first.published_at is not None
    assert "github.com/example/jailbreak-search" in first.raw_payload["comment"]

    second = items[1]
    assert second.raw_payload["doi"] == "10.1234/tsfm.2026.001"
    assert "cs.LG" in second.raw_payload["categories"]


def test_parse_atom_feed_handles_empty_feed():
    empty_feed = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    )
    assert parse_atom_feed(empty_feed) == []
