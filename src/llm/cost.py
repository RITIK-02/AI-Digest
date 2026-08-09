"""Dollar cost accounting. Track dollars, not tokens — the two vendors'
tokenizers differ, so token counts are not comparable across providers and
any dashboard that sums them is lying (CLAUDE.md, "Rules").

PRICING is a snapshot at write time (2026-08). Verify against current vendor
pricing pages before trusting cost projections for a real run — this is
exactly the kind of thing that goes stale silently.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml

from src.models import Cost

LIMITS_PATH = Path(__file__).resolve().parents[2] / "config" / "limits.yaml"

# $ per million tokens: (input, output). Standard (non-batch) rates,
# confirmed against vendor docs 2026-08-08 — deliberately used as-is even
# though every call here goes through the ~50% cheaper Batch API, so the
# budget cap in check_budget() overestimates actual spend rather than under.
#
# claude-sonnet-5 is $2.00/$10.00 introductory through 2026-08-31, reverting
# to $3.00/$15.00 on 2026-09-01 — update this when that happens.
PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "gpt-5-mini-2025-08-07": (0.25, 2.00),
    "gpt-5-2025-08-07": (1.25, 10.00),
    # OpenRouter slugs — same underlying models, billed via OpenRouter's own
    # balance instead of a direct vendor account. Same list price as direct.
    "openai/gpt-5-mini": (0.25, 2.00),
    "openai/gpt-5": (1.25, 10.00),
}


class BudgetExceeded(RuntimeError):
    pass


def compute_cost(provider: str, model: str, stage: str, tokens_in: int, tokens_out: int) -> Cost:
    if model not in PRICING:
        raise KeyError(
            f"No pricing entry for model '{model}'. Add it to PRICING in src/llm/cost.py "
            "after checking the vendor's current rate."
        )
    price_in, price_out = PRICING[model]
    usd = (tokens_in / 1_000_000) * price_in + (tokens_out / 1_000_000) * price_out
    return Cost(
        provider=provider,
        model=model,
        stage=stage,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        usd=round(usd, 6),
    )


def record_cost(conn: sqlite3.Connection, cost: Cost, run_date: str | None = None) -> None:
    run_date = run_date or date.today().isoformat()
    conn.execute(
        """
        INSERT INTO cost_log (run_date, stage, provider, model, tokens_in, tokens_out, usd)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_date,
            cost.stage,
            cost.provider,
            cost.model,
            cost.tokens_in,
            cost.tokens_out,
            cost.usd,
        ),
    )
    conn.commit()


def run_total_usd(conn: sqlite3.Connection, run_date: str | None = None) -> float:
    run_date = run_date or date.today().isoformat()
    row = conn.execute(
        "SELECT COALESCE(SUM(usd), 0) FROM cost_log WHERE run_date = ?", (run_date,)
    ).fetchone()
    return float(row[0])


def check_budget(conn: sqlite3.Connection, cap_usd: float, run_date: str | None = None) -> None:
    """Raise loudly if the run has exceeded its hard cost cap.

    Call this after recording each cost, not just at the end — the point is
    to fail fast, not to spend the whole budget before noticing.
    """
    total = run_total_usd(conn, run_date)
    if total > cap_usd:
        raise BudgetExceeded(f"Run cost ${total:.4f} exceeded cap ${cap_usd:.4f}")


@lru_cache(maxsize=1)
def load_limits() -> dict:
    with open(LIMITS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def record_and_check(conn: sqlite3.Connection, cost: Cost, run_date: str | None = None) -> None:
    """The common pattern every stage uses: record the spend, then fail loudly
    if it pushed the run over its hard cap."""
    record_cost(conn, cost, run_date)
    check_budget(conn, load_limits()["per_run_usd_cap"], run_date)
