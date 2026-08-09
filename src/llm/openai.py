"""OpenAI implementation of LLMClient, using the Batch API.

Batch mechanics: OpenAI wants a JSONL file upload, not a request list —
see anthropic.py for the direct-list counterpart. Hiding that difference
behind one interface is exactly what these two client implementations
are for.

Structured output uses strict JSON schema mode, which is the real
reliability advantage that routes triage here first (see
config/providers.yaml).
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime

import openai

from src.llm.cost import compute_cost
from src.models import BatchHandle, LLMRequest, LLMResponse, Pending

_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        stage: str,
        max_tokens: int = 4096,
        strict_json_schema: bool = True,
        reasoning_effort: str | None = None,
    ):
        self._client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.stage = stage
        self.max_tokens = max_tokens
        self.strict_json_schema = strict_json_schema
        self.reasoning_effort = reasoning_effort

    def _build_body(self, req: LLMRequest) -> dict:
        messages = []
        if req.system_prompt:
            messages.append({"role": "system", "content": req.system_prompt})
        messages.append({"role": "user", "content": req.prompt})

        body: dict = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            "messages": messages,
        }
        # GPT-5-family reasoning models spend max_completion_tokens on hidden
        # reasoning first — left uncapped, a well-specified classification
        # task can burn the whole budget on reasoning and return empty
        # content (finish_reason "length", zero visible output).
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if req.json_schema and self.strict_json_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "result",
                    "schema": req.json_schema,
                    "strict": True,
                },
            }
        return body

    def submit_batch(self, requests: list[LLMRequest]) -> BatchHandle:
        lines = []
        for req in requests:
            lines.append(
                json.dumps(
                    {
                        "custom_id": req.custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": self._build_body(req),
                    }
                )
            )
        jsonl_bytes = ("\n".join(lines) + "\n").encode("utf-8")
        uploaded = self._client.files.create(
            file=("batch_input.jsonl", io.BytesIO(jsonl_bytes)), purpose="batch"
        )
        batch = self._client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"stage": self.stage},
        )
        return BatchHandle(
            provider="openai",
            stage=self.stage,
            external_id=batch.id,
            submitted_at=datetime.now(UTC),
        )

    def poll_batch(self, handle: BatchHandle) -> list[LLMResponse] | Pending:
        batch = self._client.batches.retrieve(handle.external_id)
        if batch.status not in _TERMINAL_STATUSES:
            return Pending(provider="openai", external_id=handle.external_id)
        if batch.status != "completed" or not batch.output_file_id:
            return []

        content = self._client.files.content(batch.output_file_id).text
        responses = []
        for line in content.splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            body = entry.get("response", {}).get("body", {})
            if entry.get("error") or not body:
                continue
            choice = body["choices"][0]["message"]
            text = choice.get("content") or ""
            parsed = _try_parse_json(text)
            usage = body.get("usage", {})
            cost = compute_cost(
                provider="openai",
                model=self.model,
                stage=self.stage,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
            )
            responses.append(
                LLMResponse(
                    custom_id=entry["custom_id"], parsed=parsed, raw_text=text, cost=cost
                )
            )
        return responses


def _try_parse_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
