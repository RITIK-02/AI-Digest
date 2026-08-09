"""OpenRouter implementation of LLMClient, using OpenRouter's own Batch API
(https://openrouter.ai/docs/batch-quickstart) — not the OpenAI SDK. OpenRouter's
batch shape (inline request array, one model per batch, results returned
inline rather than via a file) differs from both OpenAI's (file upload) and
Anthropic's (per-request model), so this is a genuine third implementation,
not a thin wrapper around either existing client.

Bills against the OpenRouter account balance, independent of whatever vendor
actually serves the model underneath — useful as a stand-in when a direct
vendor account is rate- or billing-limited.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx

from src.llm.cost import compute_cost
from src.models import BatchHandle, LLMRequest, LLMResponse, Pending

_BATCHES_URL = "https://openrouter.ai/api/beta/batches"
_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        stage: str,
        max_tokens: int = 4096,
        strict_json_schema: bool = True,
        reasoning_effort: str | None = None,
    ):
        self.model = model
        self.stage = stage
        self.max_tokens = max_tokens
        self.strict_json_schema = strict_json_schema
        self.reasoning_effort = reasoning_effort
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _build_body(self, req: LLMRequest) -> dict:
        messages = []
        if req.system_prompt:
            messages.append({"role": "system", "content": req.system_prompt})
        messages.append({"role": "user", "content": req.prompt})
        body: dict = {"messages": messages, "max_tokens": self.max_tokens}
        # Same reasoning-token-starvation risk as OpenAIClient — see there.
        # OpenRouter normalizes this to its own `reasoning` object rather
        # than OpenAI's flat `reasoning_effort` field.
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        if req.json_schema and self.strict_json_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": req.json_schema, "strict": True},
            }
        return body

    def submit_batch(self, requests: list[LLMRequest]) -> BatchHandle:
        payload = {
            "endpoint": "/v1/chat/completions",
            "model": self.model,
            "requests": [{"custom_id": r.custom_id, "body": self._build_body(r)} for r in requests],
        }
        resp = httpx.post(_BATCHES_URL, json=payload, headers=self._headers, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        return BatchHandle(
            provider="openrouter",
            stage=self.stage,
            external_id=data["id"],
            submitted_at=datetime.now(UTC),
        )

    def poll_batch(self, handle: BatchHandle) -> list[LLMResponse] | Pending:
        url = f"{_BATCHES_URL}/{handle.external_id}"
        resp = httpx.get(url, headers=self._headers, timeout=30.0)
        if resp.status_code == 404:
            # The batch record can take a moment to propagate right after
            # submission — treat "not found yet" the same as "not done yet"
            # rather than raising, since a hard failure here would abort the
            # whole stage over a benign timing race.
            return Pending(provider="openrouter", external_id=handle.external_id)
        resp.raise_for_status()
        data = resp.json()

        if data["status"] not in _TERMINAL_STATUSES:
            return Pending(provider="openrouter", external_id=handle.external_id)
        if data["status"] != "completed":
            return []

        responses = []
        for item in data.get("results", []):
            if item.get("error") or "response" not in item:
                continue
            body = item["response"]["body"]
            text = body["choices"][0]["message"].get("content") or ""
            usage = body.get("usage", {})
            cost = compute_cost(
                provider="openrouter",
                model=self.model,
                stage=self.stage,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
            )
            responses.append(
                LLMResponse(
                    custom_id=item["custom_id"],
                    parsed=_try_parse_json(text),
                    raw_text=text,
                    cost=cost,
                )
            )
        return responses


def _try_parse_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
