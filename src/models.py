"""Core data schemas shared across sources, pipeline stages, and the LLM layer.

Two levels matter here: Item (one thing from one source) and Story (a cluster
of items about the same underlying thing). The digest displays stories, never
items directly. See CLAUDE.md's "Data model" section.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Section(StrEnum):
    LLM_SECURITY = "llm-security"
    ADVERSARIAL_ML = "adversarial-ml"
    TSFM = "tsfm"
    ALIGNMENT = "alignment"
    AGENTS = "agents"
    EFFICIENCY = "efficiency"
    EVALUATION = "evaluation"
    MODEL_RELEASES = "model-releases"
    POLICY = "policy"
    LONGFORM = "longform"
    DEADLINES = "deadlines"
    WATCHLIST = "watchlist"
    RISING = "rising"
    HAS_CODE = "has-code"
    JOBS = "jobs"


class RawItem(BaseModel):
    """One fetch result from a source module, before normalization."""

    source: str
    raw_id: str
    url: str
    fetched_at: datetime
    title: str | None = None
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    published_at: datetime | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class Item(BaseModel):
    """A normalized item: one thing from one source."""

    id: int | None = None
    source: str
    raw_id: str
    url: str
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    published_at: datetime | None = None
    fetched_at: datetime

    # Strong dedup keys. Extracted from the item itself or from any URL found
    # inside it (post text, HN comment bodies, etc).
    arxiv_id: str | None = None
    doi: str | None = None
    openreview_id: str | None = None
    github_repo: str | None = None

    def content_hash(self) -> str:
        """Stable key for the LLM cache tables: (content_hash, prompt_version, model)."""
        basis = self.arxiv_id or self.doi or self.openreview_id or f"{self.source}:{self.raw_id}"
        return hashlib.sha256(basis.encode()).hexdigest()[:16]


class Story(BaseModel):
    """A cluster of items about the same underlying thing.

    One canonical item (usually the paper) plus N discussion links.
    """

    id: int | None = None
    canonical_item_id: int
    discussion_item_ids: list[int] = Field(default_factory=list)
    created_at: datetime


class TriageResult(BaseModel):
    """Stage-1 output. Sections are tags, not hard filters."""

    item_hash: str
    prompt_version: str
    provider: str
    model: str
    score: float = Field(ge=0, le=10)
    sections: list[Section]
    reason: str
    cost_usd: float = 0.0


class SummaryResult(BaseModel):
    """Stage-2 output: three sentences plus one why-this-matters line."""

    item_hash: str
    prompt_version: str
    provider: str
    model: str
    summary: str
    why_it_matters: str
    cost_usd: float = 0.0


class Brief(BaseModel):
    """Stage-3 output: the daily prose brief."""

    date: str  # YYYY-MM-DD
    provider: str
    model: str
    content_md: str
    cost_usd: float = 0.0


class DedupeAdjudication(BaseModel):
    """LLM output for the borderline (0.78-0.85 cosine) dedupe band."""

    same_story: bool
    reason: str


class Cost(BaseModel):
    provider: str
    model: str
    stage: str
    tokens_in: int
    tokens_out: int
    usd: float


class LLMRequest(BaseModel):
    """One request inside a batch. custom_id round-trips through both vendors'
    batch APIs so responses can be matched back to their source item."""

    custom_id: str
    prompt: str
    system_prompt: str = ""
    json_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    custom_id: str
    parsed: dict[str, Any] | None = None
    raw_text: str
    cost: Cost


class BatchHandle(BaseModel):
    """Returned by submit_batch; persisted so a later process (collect.py) can poll it."""

    provider: str
    stage: str
    external_id: str
    submitted_at: datetime


class Pending(BaseModel):
    """Sentinel returned by poll_batch when a batch hasn't finished yet."""

    provider: str
    external_id: str


class ConsideredItem(BaseModel):
    """Every item scored in a run, not just the ones shown — the counterfactual
    set needed to evaluate ranking later."""

    run_date: str
    item_id: int
    score: float
    shown: bool
    reason: str | None = None
