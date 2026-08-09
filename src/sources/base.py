"""Shared contract and helpers for source modules.

Each source module is a plain module exporting `fetch() -> list[RawItem]` —
no base class, no registration mechanism. `ingest.py` imports the modules
listed as enabled in config/sources.yaml directly.
"""

from __future__ import annotations

import re
import time
from typing import Protocol

from src.models import RawItem

_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
_ARXIV_BARE_RE = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
_OPENREVIEW_RE = re.compile(r"openreview\.net/forum\?id=([A-Za-z0-9_-]+)", re.IGNORECASE)
_GITHUB_RE = re.compile(
    r"github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:[/)\s]|$)", re.IGNORECASE
)


class FetchFn(Protocol):
    def __call__(self) -> list[RawItem]: ...


def extract_identifiers(*texts: str | None) -> dict[str, str | None]:
    """Pull strong dedup keys out of arbitrary text — post bodies, comment
    text, anywhere a URL might appear, not just the item's own metadata."""
    blob = " ".join(t for t in texts if t)
    arxiv_match = _ARXIV_RE.search(blob) or _ARXIV_BARE_RE.search(blob)
    doi_match = _DOI_RE.search(blob)
    openreview_match = _OPENREVIEW_RE.search(blob)
    github_match = _GITHUB_RE.search(blob)
    return {
        "arxiv_id": arxiv_match.group(1) if arxiv_match else None,
        "doi": doi_match.group(0).rstrip(").,") if doi_match else None,
        "openreview_id": openreview_match.group(1) if openreview_match else None,
        "github_repo": github_match.group(1) if github_match else None,
    }


class RateLimiter:
    """Sleeps as needed to keep calls at most one per `seconds` apart."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self._last_call: float | None = None

    def wait(self) -> None:
        if self._last_call is not None:
            elapsed = time.monotonic() - self._last_call
            remaining = self.seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()
