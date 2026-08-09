"""Stage -> provider/model routing, read from config/providers.yaml.

No pipeline stage imports anthropic.py or openai.py directly. A stage calls
`get_client(stage)` and gets back something satisfying LLMClient — it never
knows which vendor answered.
"""

from __future__ import annotations

import os
import sqlite3
import time
from functools import lru_cache
from pathlib import Path

import yaml

from src.llm.anthropic import AnthropicClient
from src.llm.base import LLMClient
from src.llm.openai import OpenAIClient
from src.llm.openrouter import OpenRouterClient
from src.models import BatchHandle, LLMRequest, LLMResponse, Pending

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "providers.yaml"

_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Response size varies wildly by stage — triage returns a JSON array covering
# a ~10-item batch, dedupe/summarize return one small object, brief returns a
# full prose paragraph. A single flat default starves the bigger ones.
_STAGE_MAX_TOKENS = {
    "dedupe_adjudication": 1024,
    "triage": 4096,
    "summarize": 1024,
    "brief": 2048,
}


class NoProviderAvailable(RuntimeError):
    """Both primary and fallback failed to submit a batch for a stage."""


@lru_cache(maxsize=1)
def _config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _model_for(provider: str, tier: str) -> str:
    return _config()["providers"][provider][tier]


def _build_client(provider: str, model: str, stage: str) -> LLMClient:
    opts = _config().get("provider_options", {}).get(provider, {})
    api_key = os.environ.get(_API_KEY_ENV[provider])
    if not api_key:
        raise RuntimeError(f"{_API_KEY_ENV[provider]} is not set")
    max_tokens = _STAGE_MAX_TOKENS.get(stage, 1024)

    if provider == "anthropic":
        return AnthropicClient(
            api_key=api_key,
            model=model,
            stage=stage,
            max_tokens=max_tokens,
            prompt_caching=opts.get("prompt_caching", True),
        )
    if provider == "openai":
        return OpenAIClient(
            api_key=api_key,
            model=model,
            stage=stage,
            max_tokens=max_tokens,
            strict_json_schema=opts.get("strict_json_schema", True),
            reasoning_effort=opts.get("reasoning_effort"),
        )
    if provider == "openrouter":
        return OpenRouterClient(
            api_key=api_key,
            model=model,
            stage=stage,
            max_tokens=max_tokens,
            strict_json_schema=opts.get("strict_json_schema", True),
            reasoning_effort=opts.get("reasoning_effort"),
        )
    raise ValueError(f"Unknown provider: {provider}")


def get_client(stage: str, *, use_fallback: bool = False) -> LLMClient:
    route = _config()["routing"][stage]
    slot = route["fallback"] if use_fallback else route["primary"]
    model = _model_for(slot["provider"], slot["tier"])
    return _build_client(slot["provider"], model, stage)


def get_client_for_batch(handle: BatchHandle) -> LLMClient:
    """Reconstruct the client that must poll a given batch — whichever of
    primary/fallback actually submitted it, looked up by provider rather than
    assumed, in case routing config changes between submit and collect."""
    route = _config()["routing"][handle.stage]
    for slot in (route["primary"], route["fallback"]):
        if slot["provider"] == handle.provider:
            model = _model_for(slot["provider"], slot["tier"])
            return _build_client(slot["provider"], model, handle.stage)
    raise ValueError(f"No routing slot for provider={handle.provider} stage={handle.stage}")


def submit_stage_batch(
    stage: str, requests: list[LLMRequest], conn: sqlite3.Connection
) -> BatchHandle:
    """Submit a batch for a stage, falling back once on provider failure.

    A digest built entirely on the backup provider is a fine outcome; a
    missing digest is not — so this never silently skips the stage.
    """
    errors = []
    for use_fallback in (False, True):
        try:
            client = get_client(stage, use_fallback=use_fallback)
            handle = client.submit_batch(requests)
            conn.execute(
                """
                INSERT INTO batches (stage, provider, external_id, status, submitted_at)
                VALUES (?, ?, ?, 'submitted', ?)
                """,
                (stage, handle.provider, handle.external_id, handle.submitted_at.isoformat()),
            )
            conn.commit()
            if use_fallback:
                print(f"[routing] stage={stage} primary failed, used fallback provider")
            return handle
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any provider failure falls back
            errors.append(str(exc))
            print(f"[routing] stage={stage} provider attempt failed: {exc}")

    raise NoProviderAvailable(f"stage={stage} primary and fallback both failed: {errors}")


def poll_until_done(
    client: LLMClient, handle: BatchHandle, max_wait_s: float = 600, interval_s: float = 15
) -> list[LLMResponse] | None:
    """Block, polling a batch until it finishes or max_wait_s elapses.

    Only reasonable for small single-request batches (the brief stage, which
    runs synchronously inside collect.py and needs a result before publish
    can proceed). Triage and summarize use the async submit/collect split
    across the two workflows instead — see submit_stage_batch.
    """
    waited = 0.0
    while waited <= max_wait_s:
        try:
            result = client.poll_batch(handle)
        except Exception as exc:  # noqa: BLE001 - a transient network blip mid-wait shouldn't
            # abort an otherwise-healthy batch; treat it like "not ready yet" and keep polling.
            print(f"[routing] poll_batch transient error, retrying: {exc}")
        else:
            if not isinstance(result, Pending):
                return result
        time.sleep(interval_s)
        waited += interval_s
    return None
