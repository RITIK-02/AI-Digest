"""The one interface every LLM provider implements.

Deliberately narrow — kept at the intersection of the Anthropic and OpenAI
batch APIs, not the union. Reasoning effort, cache control, logprobs, service
tiers: none of that belongs here. Provider-specific behavior lives inside
each client and is configured via config/providers.yaml's `provider_options`.

No pipeline stage may import anthropic.py or openai.py directly — stages call
`get_client(stage)` from routing.py and know nothing about who answers.
"""

from __future__ import annotations

from typing import Protocol

from src.models import BatchHandle, LLMRequest, LLMResponse, Pending


class LLMClient(Protocol):
    def submit_batch(self, requests: list[LLMRequest]) -> BatchHandle: ...

    def poll_batch(self, handle: BatchHandle) -> list[LLMResponse] | Pending: ...
