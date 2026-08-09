"""Anthropic implementation of LLMClient, using the Message Batches API.

Batch mechanics: Anthropic takes a request list directly (no file upload,
unlike OpenAI) — see openai.py for the JSONL-upload counterpart. Hiding that
difference behind one interface is exactly what these two client
implementations are for.

Structured output uses forced tool-use (Anthropic has no strict-JSON-schema
mode the way OpenAI does — that gap is *why* triage routes to OpenAI first;
see config/providers.yaml).
"""

from __future__ import annotations

from datetime import UTC, datetime

import anthropic

from src.llm.cost import compute_cost
from src.models import BatchHandle, LLMRequest, LLMResponse, Pending

_TERMINAL_STATUSES = {"ended"}
_TOOL_NAME = "emit_result"


class AnthropicClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        stage: str,
        max_tokens: int = 1024,
        prompt_caching: bool = True,
    ):
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.stage = stage
        self.max_tokens = max_tokens
        self.prompt_caching = prompt_caching

    def _build_request(self, req: LLMRequest) -> dict:
        params: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": req.prompt}],
        }
        if req.system_prompt:
            block: dict = {"type": "text", "text": req.system_prompt}
            if self.prompt_caching:
                block["cache_control"] = {"type": "ephemeral"}
            params["system"] = [block]
        if req.json_schema:
            params["tools"] = [
                {
                    "name": _TOOL_NAME,
                    "description": "Emit the structured result for this item.",
                    "input_schema": req.json_schema,
                }
            ]
            params["tool_choice"] = {"type": "tool", "name": _TOOL_NAME}
        return {"custom_id": req.custom_id, "params": params}

    def submit_batch(self, requests: list[LLMRequest]) -> BatchHandle:
        batch_requests = [self._build_request(r) for r in requests]
        batch = self._client.messages.batches.create(requests=batch_requests)
        return BatchHandle(
            provider="anthropic",
            stage=self.stage,
            external_id=batch.id,
            submitted_at=datetime.now(UTC),
        )

    def poll_batch(self, handle: BatchHandle) -> list[LLMResponse] | Pending:
        batch = self._client.messages.batches.retrieve(handle.external_id)
        if batch.processing_status not in _TERMINAL_STATUSES:
            return Pending(provider="anthropic", external_id=handle.external_id)

        responses = []
        for entry in self._client.messages.batches.results(handle.external_id):
            result = entry.result
            if result.type != "succeeded":
                continue
            message = result.message
            parsed, raw_text = _extract_output(message)
            cost = compute_cost(
                provider="anthropic",
                model=self.model,
                stage=self.stage,
                tokens_in=message.usage.input_tokens,
                tokens_out=message.usage.output_tokens,
            )
            responses.append(
                LLMResponse(custom_id=entry.custom_id, parsed=parsed, raw_text=raw_text, cost=cost)
            )
        return responses


def _extract_output(message) -> tuple[dict | None, str]:
    text_parts = []
    for block in message.content:
        if block.type == "tool_use" and block.name == _TOOL_NAME:
            return block.input, ""
        if block.type == "text":
            text_parts.append(block.text)
    return None, "".join(text_parts)
